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
