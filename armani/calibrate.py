"""The pixel -> robot map: a plane homography between the C920 image and the table.

Everything the arm does with what it sees runs through this file, so it is built
to refuse rather than guess. A homography computed from sloppy correspondences
still produces confident-looking coordinates that are simply wrong, and the arm
would then drive precisely to the wrong place. So ``save`` reprojects every
calibration point and refuses to write a map whose mean error exceeds
``config.CALIB_MAX_REPROJECTION_PX``.

Two ways to get the correspondences:

* **ChArUco board** (primary, following google-gemini/robotics-pointing-sample).
  The board gives many sub-pixel-accurate image points in one capture. It cannot
  give ROBOT coordinates on its own, so the operator states where the board's
  origin corner sits in the robot frame and how it is rotated. That measurement
  is the method's weak link: a 5 mm slip of the ruler shifts the whole map 5 mm.

* **Gripper-tip 6-point** (fallback). The operator touches the gripper to a spot
  on the table, forward kinematics gives that spot's robot XY exactly, and they
  click the same spot in the image. Slower, needs the arm, and needs no ruler at
  all — the robot measures itself. See ``scripts/calibrate_camera.py``.

The homography is only valid for the camera pose and frame size it was computed
at. Move the tripod and it must be redone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from armani import config
from armani.logutil import get_logger, log_event

log = get_logger("calibrate")

Point = tuple[float, float]


class CalibrationError(RuntimeError):
    """Calibration could not be computed or is not good enough to save."""


@dataclass(frozen=True)
class Homography:
    """A saved pixel -> robot map, plus the evidence that it is any good."""

    matrix: np.ndarray  # 3x3, maps pixel (u, v) -> robot (x, y) in metres
    frame_size: tuple[int, int]
    method: str
    created: str
    mean_error_px: float
    max_error_px: float
    point_count: int
    table_polygon: tuple[Point, ...]

    def to_robot(self, pixel: Point) -> Point:
        return pixel_to_robot(self.matrix, pixel)

    def to_pixel(self, robot: Point) -> Point:
        return robot_to_pixel(self.matrix, robot)


# --- Core maths ----------------------------------------------------------


def _apply(matrix: np.ndarray, point: Point) -> Point:
    """Apply a 3x3 homography to one 2D point."""
    vector = np.array([float(point[0]), float(point[1]), 1.0])
    mapped = matrix @ vector
    w = mapped[2]
    if not np.isfinite(w) or abs(w) < 1e-12:
        # The point is on (or beyond) the horizon of the mapped plane. There is
        # no finite answer, and returning a huge number would look like a real
        # coordinate to every caller downstream.
        raise CalibrationError(
            f"point {point} maps to infinity — it is outside the calibrated plane"
        )
    return (float(mapped[0] / w), float(mapped[1] / w))


def pixel_to_robot(matrix: np.ndarray, pixel: Point) -> Point:
    """Image pixel (u, v) -> robot table coordinates (x, y) in metres."""
    return _apply(matrix, pixel)


def robot_to_pixel(matrix: np.ndarray, robot: Point) -> Point:
    """Robot table coordinates (x, y) -> image pixel (u, v)."""
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise CalibrationError(f"homography is singular and cannot be inverted: {exc}") from exc
    return _apply(inverse, robot)


def compute(pixels: list[Point], robots: list[Point]) -> tuple[np.ndarray, float, float]:
    """Fit pixel -> robot and measure how well it fits.

    Returns (matrix, mean_error_px, max_error_px). The error is measured in
    PIXELS by mapping each robot point back through the inverse homography and
    comparing with the pixel it came from — reporting it in metres would hide
    how bad a map is in the part of the image where the camera is oblique.
    """
    if len(pixels) != len(robots):
        raise CalibrationError(f"{len(pixels)} pixel points but {len(robots)} robot points")
    if len(pixels) < config.CALIB_MIN_POINTS:
        raise CalibrationError(
            f"need at least {config.CALIB_MIN_POINTS} correspondences, got {len(pixels)}"
        )

    import cv2

    source = np.array(pixels, dtype=np.float64).reshape(-1, 1, 2)
    destination = np.array(robots, dtype=np.float64).reshape(-1, 1, 2)

    # Plain least squares, NOT RANSAC: with a handful of hand-made points RANSAC
    # can silently discard the ones that disagree and report a beautiful error
    # over the survivors. We want every point to count, and a bad point to show
    # up as a bad score.
    matrix, _ = cv2.findHomography(source, destination, method=0)
    if matrix is None:
        raise CalibrationError(
            "findHomography failed — the points are probably collinear or duplicated. "
            "Spread them across the whole table, not along one line."
        )
    if not np.all(np.isfinite(matrix)):
        raise CalibrationError("findHomography produced a non-finite matrix")

    errors = reprojection_errors(matrix, pixels, robots)
    return matrix, float(np.mean(errors)), float(np.max(errors))


def reprojection_errors(matrix: np.ndarray, pixels: list[Point], robots: list[Point]) -> list[float]:
    """Per-point pixel error when each robot point is mapped back to the image."""
    errors = []
    for pixel, robot in zip(pixels, robots):
        back = robot_to_pixel(matrix, robot)
        errors.append(float(np.hypot(back[0] - pixel[0], back[1] - pixel[1])))
    return errors


# --- Table polygon -------------------------------------------------------


def polygon_from_points(robots: list[Point], margin: float | None = None) -> tuple[Point, ...]:
    """Convex hull of the calibrated robot points, pulled in by ``margin``.

    The hull is the region we have evidence for. A homography extrapolates
    happily and wrongly beyond its data, so the workspace polygon is shrunk
    toward the centroid — the arm may only be sent where calibration actually
    measured, with a little to spare.
    """
    if margin is None:
        margin = config.TABLE_MARGIN_M
    if len(robots) < 3:
        raise CalibrationError(f"need 3+ points for a table polygon, got {len(robots)}")

    import cv2

    points = np.array(robots, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) < 3:
        raise CalibrationError("calibration points are collinear; no table area to work with")

    centroid = hull.mean(axis=0)
    shrunk: list[Point] = []
    for vertex in hull:
        offset = vertex - centroid
        distance = float(np.hypot(*offset))
        if distance <= margin:
            # Pulling in by more than the radius would turn the polygon inside
            # out. A table this small is a calibration mistake, not a workspace.
            raise CalibrationError(
                f"table polygon is smaller than the {margin * 100:.0f} cm safety margin — "
                "spread the calibration points further apart"
            )
        scale = (distance - margin) / distance
        shrunk.append((float(centroid[0] + offset[0] * scale), float(centroid[1] + offset[1] * scale)))
    return tuple(shrunk)


def point_in_polygon(x: float, y: float, polygon: tuple[Point, ...] | None = None) -> bool:
    """Safety rule 3: is this robot (x, y) inside the calibrated table?

    Ray casting, written out rather than delegated to cv2, because this is on
    the path that decides whether the arm moves: it must be unit-testable
    without a camera stack and must fail closed. An empty polygon — which is
    what an uncalibrated system has — always returns False.
    """
    if polygon is None:
        polygon = config.TABLE_POLYGON
    if not polygon or len(polygon) < 3:
        return False
    if not (np.isfinite(x) and np.isfinite(y)):
        return False

    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        # Does the horizontal ray from (x, y) cross this edge?
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


# --- Persistence ---------------------------------------------------------


def save(
    matrix: np.ndarray,
    pixels: list[Point],
    robots: list[Point],
    frame_size: tuple[int, int],
    method: str,
    path: Path | None = None,
) -> Homography:
    """Write the map, or refuse to. A bad map is worse than no map.

    Refusing leaves any previous calibration in place untouched.
    """
    if path is None:
        path = config.HOMOGRAPHY_PATH

    errors = reprojection_errors(matrix, pixels, robots)
    mean_error, max_error = float(np.mean(errors)), float(np.max(errors))
    if mean_error > config.CALIB_MAX_REPROJECTION_PX:
        raise CalibrationError(
            f"mean reprojection error is {mean_error:.1f} px, over the "
            f"{config.CALIB_MAX_REPROJECTION_PX:.0f} px limit. NOT SAVED — a map this bad "
            "would send the arm confidently to the wrong place.\n"
            "  Likely causes: the board or camera moved between captures, a mis-clicked "
            "point, a wrong square size, or points spread along a line instead of over the area.\n"
            f"  Per-point errors (px): {', '.join(f'{e:.1f}' for e in errors)}"
        )

    polygon = polygon_from_points(robots)
    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "matrix": [[float(v) for v in row] for row in matrix],
        "table_polygon": [[round(x, 4), round(y, 4)] for x, y in polygon],
        "table_height_m": config.TABLE_HEIGHT_M,
        "reprojection_px": {
            "mean": round(mean_error, 2),
            "max": round(max_error, 2),
            "per_point": [round(e, 2) for e in errors],
        },
        "points": {
            "pixel": [[round(float(u), 1), round(float(v), 1)] for u, v in pixels],
            "robot": [[round(float(x), 4), round(float(y), 4)] for x, y in robots],
        },
        "note": (
            "Valid ONLY for this camera pose and frame size. If the tripod or the table "
            "is bumped, re-run scripts/calibrate_camera.py."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log_event(
        "homography_saved",
        method=method,
        points=len(pixels),
        mean_error_px=round(mean_error, 2),
        max_error_px=round(max_error, 2),
        polygon=[[round(x, 4), round(y, 4)] for x, y in polygon],
    )
    return Homography(
        matrix=matrix,
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        method=method,
        created=str(payload["created"]),
        mean_error_px=mean_error,
        max_error_px=max_error,
        point_count=len(pixels),
        table_polygon=polygon,
    )


def load(path: Path | None = None) -> Homography | None:
    """Read the saved map. None means "not calibrated yet", which is not an error."""
    if path is None:
        path = config.HOMOGRAPHY_PATH
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        matrix = np.array(payload["matrix"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"matrix is {matrix.shape}, expected a finite 3x3")
        width, height = payload["frame_size"]
        errors = payload.get("reprojection_px", {})
        polygon = tuple((float(x), float(y)) for x, y in payload.get("table_polygon", ()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.error("ignoring unreadable homography at %s: %s", path, exc)
        return None

    return Homography(
        matrix=matrix,
        frame_size=(int(width), int(height)),
        method=str(payload.get("method", "unknown")),
        created=str(payload.get("created", "unknown")),
        mean_error_px=float(errors.get("mean", float("nan"))),
        max_error_px=float(errors.get("max", float("nan"))),
        point_count=len(payload.get("points", {}).get("pixel", [])),
        table_polygon=polygon,
    )


# --- ChArUco -------------------------------------------------------------


def _board():
    """The ChArUco board described by config, plus its dictionary."""
    import cv2

    name = config.CHARUCO_DICT
    if not hasattr(cv2.aruco, name):
        raise CalibrationError(
            f"unknown ArUco dictionary {name!r}. Try one of: "
            + ", ".join(n for n in dir(cv2.aruco) if n.startswith("DICT_"))
        )
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    board = cv2.aruco.CharucoBoard(
        (config.CHARUCO_SQUARES_X, config.CHARUCO_SQUARES_Y),
        config.CHARUCO_SQUARE_M,
        config.CHARUCO_MARKER_M,
        dictionary,
    )
    return board, dictionary


def board_image(pixels_per_square: int = 120):
    """Render the configured board as an image the operator can print."""
    board, _ = _board()
    size = (
        config.CHARUCO_SQUARES_X * pixels_per_square,
        config.CHARUCO_SQUARES_Y * pixels_per_square,
    )
    return board.generateImage(size)


def charuco_correspondences(
    frame, origin: Point, rotation_deg: float = 0.0, mirror: bool = False
) -> tuple[list[Point], list[Point]]:
    """Detect the board and pair each corner's pixel with its robot XY.

    ``origin`` is the robot (x, y) of the board's FIRST chessboard corner — the
    inner corner nearest the board's origin, not the paper's edge — and
    ``rotation_deg`` is how far the board's +X axis is rotated from the robot's
    +X axis, counter-clockwise seen from above.

    ``mirror`` flips the board's Y axis before rotating. This is not paranoia:
    OpenCV's board coordinates and a sheet of paper laid face-up on a table do
    not necessarily agree on handedness, and the reference sample
    (google-gemini/robotics-pointing-sample) hardcodes a board->robot mapping
    whose determinant is -1 — a reflection, not a rotation. Which one is right
    depends on how the board is laid down, so the caller tries both and keeps
    whichever reprojects better. Both candidates still face the error ceiling in
    save(), so this can never quietly accept a wrong map.

    This is where the ChArUco method's accuracy is won or lost: the corner
    pixels are sub-pixel exact, but the robot coordinates are only as good as
    the operator's ruler.
    """
    import cv2

    board, _ = _board()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(grey)

    if corners is None or ids is None or len(ids) < config.CALIB_MIN_POINTS:
        found = 0 if ids is None else len(ids)
        raise CalibrationError(
            f"found {found} ChArUco corners, need {config.CALIB_MIN_POINTS}. "
            "Check the whole board is in frame, lit evenly and not glare-washed, and that "
            f"ARMANI_CHARUCO_* match the board you printed ({config.CHARUCO_SQUARES_X}x"
            f"{config.CHARUCO_SQUARES_Y}, {config.CHARUCO_SQUARE_M * 1000:.0f} mm squares)."
        )

    # Board-frame coordinates of every chessboard corner, in metres.
    board_points = board.getChessboardCorners()
    angle = np.radians(rotation_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    pixels: list[Point] = []
    robots: list[Point] = []
    for corner, corner_id in zip(corners.reshape(-1, 2), ids.reshape(-1)):
        bx, by = float(board_points[corner_id][0]), float(board_points[corner_id][1])
        if mirror:
            by = -by
        pixels.append((float(corner[0]), float(corner[1])))
        # float() rather than letting numpy scalars through: these end up in
        # JSON and in comparisons, and np.float64 serialises differently.
        robots.append(
            (
                float(origin[0] + bx * cos_a - by * sin_a),
                float(origin[1] + bx * sin_a + by * cos_a),
            )
        )
    log.info("detected %d ChArUco corners", len(pixels))
    return pixels, robots


# --- ChArUco → rigid transform (Spike S1) --------------------------------
#
# The stage-4 ChArUco path above needs the operator to MEASURE where the board
# sits in the robot frame with a ruler, and a 5 mm slip warps the whole map.
# This path removes the ruler: the pixel→board-mm homography is fit from the
# sub-pixel corners (well-conditioned), and board→robot is a RIGID 2D transform
# (rotation + translation, 3 DOF) fit from just three tip touches, so click
# error averages out instead of distorting an 8-DOF homography.
#
# Units discipline (stage 4 burned us on this once): the board is built with the
# MEASURED square length in METRES, so board.getChessboardCorners() is already
# in metres and the rigid fit is metres→metres with no scale. Gemini points are
# [y, x] (handled in eyes.py); every (x, y) here is pixels or robot metres.


def measured_square_m() -> float:
    """The printed square size to trust, in metres.

    ``ARMANI_CHARUCO_SQUARE_MM`` (the operator's ruler measurement) wins, because
    printers rescale and the intended size is only a hope. Falls back to the
    config default. Read at call time so exporting the env before a run works.
    """
    raw = os.getenv("ARMANI_CHARUCO_SQUARE_MM")
    if raw and raw.strip():
        try:
            return float(raw) / 1000.0
        except ValueError:
            log.warning("ARMANI_CHARUCO_SQUARE_MM=%r is not a number; using config", raw)
    return config.CHARUCO_SQUARE_M


def choose_aruco_api(version: str) -> str:
    """'modern' for OpenCV >= 4.7 (CharucoDetector), else 'legacy'. Pure, testable."""
    try:
        major, minor = (int(part) for part in version.split(".")[:2])
    except (ValueError, AttributeError):
        return "modern"  # unparseable version: assume the current API
    return "modern" if (major, minor) >= (4, 7) else "legacy"


def _charuco_detector(board):
    """A ChArUco detector for the installed OpenCV, or a clear error on legacy."""
    import cv2

    api = choose_aruco_api(cv2.__version__)
    if api == "legacy" or not hasattr(cv2.aruco, "CharucoDetector"):
        raise CalibrationError(
            f"OpenCV {cv2.__version__} predates the CharucoDetector API (needs >= 4.7). "
            "The legacy aruco path is not implemented in this spike — upgrade opencv-python "
            "in the detector venv, or use the stage-4 tip method."
        )
    return cv2.aruco.CharucoDetector(board)


def charuco_board(square_m: float | None = None, marker_m: float | None = None):
    """Build the configured ChArUco board with an explicit square length.

    ``square_m`` is the MEASURED printed square in metres (the operator's ruler
    is truth). The marker keeps the config's marker/square ratio so a detector
    built for it still matches the printed board.
    """
    import cv2

    if square_m is None:
        square_m = measured_square_m()
    if marker_m is None:
        ratio = config.CHARUCO_MARKER_M / config.CHARUCO_SQUARE_M
        marker_m = square_m * ratio
    if not hasattr(cv2.aruco, config.CHARUCO_DICT):
        raise CalibrationError(f"unknown ArUco dictionary {config.CHARUCO_DICT!r}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.CHARUCO_DICT))
    board = cv2.aruco.CharucoBoard(
        (config.CHARUCO_SQUARES_X, config.CHARUCO_SQUARES_Y),
        float(square_m), float(marker_m), dictionary,
    )
    return board


def calibration_corner_ids() -> tuple[int, int, int]:
    """Three well-spread chessboard-corner ids: two along an edge, one opposite.

    Indexed however OpenCV orders getChessboardCorners — we only ever use the id
    to look up the board-mm coordinate, so the choice just needs the three to be
    far apart for a stable rigid fit. make_charuco.py marks these exact ids on
    the printed board so the operator touches the ones we fit against.
    """
    cols = config.CHARUCO_SQUARES_X - 1
    rows = config.CHARUCO_SQUARES_Y - 1
    origin = 0
    along_x = cols - 1
    opposite = (rows - 1) * cols
    return origin, along_x, opposite


def detect_charuco(frame, board) -> tuple[list[Point], list[Point], list[int]]:
    """Detect the board. Returns (pixel corners, board-mm corners in metres, ids).

    Corners come back sub-pixel from ``detectBoard``. The board-frame coordinate
    of each is taken from the board geometry, so it is exact by construction.
    """
    import cv2

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _, _ = _charuco_detector(board).detectBoard(grey)
    if corners is None or ids is None or len(ids) < config.CALIB_MIN_POINTS:
        found = 0 if ids is None else len(ids)
        raise CalibrationError(
            f"found {found} ChArUco corners, need {config.CALIB_MIN_POINTS}. "
            "Check the whole board is in frame, flat, lit evenly and not glare-washed."
        )
    board_points = board.getChessboardCorners()
    pixels: list[Point] = []
    board_mm: list[Point] = []
    found_ids: list[int] = []
    for corner, corner_id in zip(corners.reshape(-1, 2), ids.reshape(-1)):
        cid = int(corner_id)
        pixels.append((float(corner[0]), float(corner[1])))
        board_mm.append((float(board_points[cid][0]), float(board_points[cid][1])))
        found_ids.append(cid)
    log.info("detected %d ChArUco corners", len(pixels))
    return pixels, board_mm, found_ids


def fit_board_homography(pixels: list[Point], board_m: list[Point]) -> tuple[np.ndarray, float]:
    """Fit pixel -> board-metres. Returns (H 3x3, RMS reprojection in PIXELS)."""
    import cv2

    if len(pixels) != len(board_m) or len(pixels) < 4:
        raise CalibrationError(f"need >=4 matched corners, got {len(pixels)}")
    src = np.array(pixels, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array(board_m, dtype=np.float64).reshape(-1, 1, 2)
    matrix, _ = cv2.findHomography(src, dst, method=0)
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise CalibrationError("board homography fit failed (corners collinear or degenerate)")

    # RMS in pixels: map each board-mm corner back to the image via the inverse
    # homography and compare to the sub-pixel corner it came from.
    inverse = np.linalg.inv(matrix)
    squared = []
    for (u, v), bm in zip(pixels, board_m):
        pu, pv = _apply(inverse, bm)
        squared.append((pu - u) ** 2 + (pv - v) ** 2)
    rms = float(np.sqrt(np.mean(squared)))
    return matrix, rms


def fit_rigid_2d(
    src: list[Point], dst: list[Point]
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Least-squares rigid 2D transform src -> dst (rotation + translation, no scale).

    Kabsch/Procrustes in 2D, with the reflection guard so a noisy 3-point fit
    cannot flip into a mirror. Returns (R 2x2, t 2, per-point residuals in the
    same units as the inputs — metres here). Both inputs MUST share units and
    scale (a rigid transform preserves distance).
    """
    if len(src) != len(dst) or len(src) < 2:
        raise CalibrationError(f"need >=2 matched points, got {len(src)} and {len(dst)}")
    s = np.array(src, dtype=np.float64)
    d = np.array(dst, dtype=np.float64)
    mu_s, mu_d = s.mean(axis=0), d.mean(axis=0)
    sc, dc = s - mu_s, d - mu_d
    covariance = sc.T @ dc
    u, _, vt = np.linalg.svd(covariance)
    reflect = np.diag([1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    rotation = vt.T @ reflect @ u.T
    translation = mu_d - rotation @ mu_s
    residuals = [float(np.hypot(*(point_d - (rotation @ point_s + translation))))
                 for point_s, point_d in zip(s, d)]
    return rotation, translation, residuals


def rigid_affine_3x3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Pack a 2D rigid (R, t) into a 3x3 homogeneous matrix."""
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = translation
    return matrix


def compose_pixel_to_robot(
    board_homography: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """pixel -> robot metres as ONE 3x3: rigid(board->robot) @ homography(pixel->board).

    The composition of a projective homography with an affine is still a single
    homography, so this drops straight into the existing `matrix` field and every
    downstream caller (Homography.to_robot, grasp.hover_over) works unchanged.
    """
    return rigid_affine_3x3(rotation, translation) @ board_homography


def save_charuco(
    matrix: np.ndarray,
    corner_pixels: list[Point],
    frame_size: tuple[int, int],
    *,
    board_rms_px: float,
    tip_board_m: list[Point],
    tip_robot_m: list[Point],
    rigid_residuals_m: list[float],
    path: Path | None = None,
) -> Homography:
    """Write a composed pixel->robot calibration in the existing file format.

    Refuses if the board homography RMS exceeds the pixel ceiling — a bad map is
    worse than none. The workspace polygon is the hull of ALL detected corners
    mapped to robot, so it covers the whole board area.
    """
    if path is None:
        path = config.HOMOGRAPHY_PATH
    if board_rms_px > config.CALIB_MAX_REPROJECTION_PX:
        raise CalibrationError(
            f"board homography RMS is {board_rms_px:.1f} px, over the "
            f"{config.CALIB_MAX_REPROJECTION_PX:.0f} px ceiling. NOT SAVED."
        )

    robot_corners = [pixel_to_robot(matrix, px) for px in corner_pixels]
    polygon = polygon_from_points(robot_corners)
    residual_mm = [round(r * 1000, 2) for r in rigid_residuals_m]
    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "charuco_rigid",
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "matrix": [[float(v) for v in row] for row in matrix],
        "table_polygon": [[round(x, 4), round(y, 4)] for x, y in polygon],
        "table_height_m": config.TABLE_HEIGHT_M,
        # Reuse the schema's error field so describe()/load() work; here it is the
        # pixel-plane fit quality (RMS), which is the meaningful board number.
        "reprojection_px": {"mean": round(board_rms_px, 2), "max": round(board_rms_px, 2)},
        # The board corners this was fit from, so load()/describe() report a real
        # point count (the robot side lives under "charuco.tip_robot_m").
        "points": {"pixel": [[round(float(u), 1), round(float(v), 1)] for u, v in corner_pixels]},
        "charuco": {
            "board_rms_px": round(board_rms_px, 3),
            "square_mm": round(measured_square_m() * 1000, 2),
            "corners_used": len(corner_pixels),
            "rigid_residual_mm": residual_mm,
            "rigid_residual_mean_mm": round(float(np.mean(residual_mm)), 2) if residual_mm else None,
            "tip_board_m": [[round(x, 4), round(y, 4)] for x, y in tip_board_m],
            "tip_robot_m": [[round(x, 4), round(y, 4)] for x, y in tip_robot_m],
        },
        "note": (
            "ChArUco pixel->board homography composed with a 3-touch rigid board->robot "
            "transform. Valid ONLY for this camera pose and frame size."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log_event(
        "homography_saved",
        method="charuco_rigid",
        board_rms_px=round(board_rms_px, 2),
        rigid_residual_mean_mm=payload["charuco"]["rigid_residual_mean_mm"],
        polygon=[[round(x, 4), round(y, 4)] for x, y in polygon],
    )
    return Homography(
        matrix=matrix,
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        method="charuco_rigid",
        created=str(payload["created"]),
        mean_error_px=board_rms_px,
        max_error_px=board_rms_px,
        point_count=len(corner_pixels),
        table_polygon=polygon,
    )


def roundtrip_pixel_errors(
    matrix: np.ndarray, corner_pixels: list[Point]
) -> list[float]:
    """pixel -> robot -> pixel self-consistency, in pixels.

    A sanity check on the composed matrix (catches a broken inverse), NOT an
    accuracy measure — with an invertible matrix this is ~0. Real accuracy comes
    from the rigid residuals at the tip corners and the operator's ruler.
    """
    errors = []
    for px in corner_pixels:
        robot = pixel_to_robot(matrix, px)
        back = robot_to_pixel(matrix, robot)
        errors.append(float(np.hypot(back[0] - px[0], back[1] - px[1])))
    return errors


def describe(homography: Homography | None) -> str:
    """One-paragraph human summary, for smoke tests and the doctor."""
    if homography is None:
        return (
            f"NOT CALIBRATED — no {config.HOMOGRAPHY_PATH.name}. "
            "Run: python scripts/calibrate_camera.py"
        )
    return (
        f"{homography.method} calibration from {homography.point_count} points, "
        f"{homography.created}, frame {homography.frame_size[0]}x{homography.frame_size[1]}, "
        f"reprojection mean {homography.mean_error_px:.1f} px / max {homography.max_error_px:.1f} px, "
        f"table polygon {len(homography.table_polygon)} corners"
    )
