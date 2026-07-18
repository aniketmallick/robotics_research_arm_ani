"""Homography maths and the workspace check. No camera, no arm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import calibrate, config  # noqa: E402

# A synthetic but realistic map: a 40x30 cm table region seen by a 640x480
# camera. Built by picking robot points and their pixels so the fit is exact.
SQUARE_PIXELS = [(100.0, 100.0), (540.0, 100.0), (540.0, 380.0), (100.0, 380.0),
                 (320.0, 100.0), (320.0, 380.0)]
SQUARE_ROBOT = [(0.10, 0.15), (0.10, -0.15), (0.40, -0.15), (0.40, 0.15),
                (0.10, 0.0), (0.40, 0.0)]


def fitted():
    return calibrate.compute(SQUARE_PIXELS, SQUARE_ROBOT)


def test_exact_correspondences_fit_with_no_error():
    # Sub-thousandth of a pixel: exact to the limits of the float arithmetic,
    # which is what "no error" can mean for a least-squares fit.
    _, mean_error, max_error = fitted()
    assert mean_error < 1e-3
    assert max_error < 1e-3


def test_round_trip_pixel_to_robot_and_back():
    matrix, _, _ = fitted()
    for pixel in SQUARE_PIXELS:
        robot = calibrate.pixel_to_robot(matrix, pixel)
        back = calibrate.robot_to_pixel(matrix, robot)
        assert back == pytest.approx(pixel, abs=1e-6)


def test_interpolated_point_lands_where_geometry_says():
    matrix, _, _ = fitted()
    # Centre of the image is the centre of the mapped rectangle.
    x, y = calibrate.pixel_to_robot(matrix, (320.0, 240.0))
    assert x == pytest.approx(0.25, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_too_few_points_is_refused():
    with pytest.raises(calibrate.CalibrationError, match="at least"):
        calibrate.compute(SQUARE_PIXELS[:3], SQUARE_ROBOT[:3])


def test_mismatched_lists_are_refused():
    with pytest.raises(calibrate.CalibrationError, match="but"):
        calibrate.compute(SQUARE_PIXELS, SQUARE_ROBOT[:-1])


def test_collinear_points_are_refused():
    pixels = [(float(i * 50), 100.0) for i in range(6)]
    robots = [(0.1 + i * 0.02, 0.0) for i in range(6)]
    with pytest.raises(calibrate.CalibrationError):
        calibrate.compute(pixels, robots)


def test_a_bad_point_shows_up_as_error_rather_than_being_discarded():
    """Least squares, not RANSAC: one bad click must be visible, not silently dropped."""
    pixels = list(SQUARE_PIXELS)
    pixels[2] = (pixels[2][0] + 60.0, pixels[2][1] + 60.0)
    _, mean_error, _ = calibrate.compute(pixels, SQUARE_ROBOT)
    assert mean_error > 5.0


def test_save_refuses_a_bad_map(tmp_path):
    pixels = list(SQUARE_PIXELS)
    pixels[2] = (pixels[2][0] + 200.0, pixels[2][1] + 200.0)
    matrix, _, _ = calibrate.compute(pixels, SQUARE_ROBOT)
    target = tmp_path / "homography.json"
    with pytest.raises(calibrate.CalibrationError, match="NOT SAVED"):
        calibrate.save(matrix, pixels, SQUARE_ROBOT, (640, 480), "test", path=target)
    assert not target.exists(), "a rejected calibration must not leave a file behind"


def test_save_then_load_round_trip(tmp_path):
    matrix, _, _ = fitted()
    target = tmp_path / "homography.json"
    saved = calibrate.save(matrix, SQUARE_PIXELS, SQUARE_ROBOT, (640, 480), "test", path=target)
    loaded = calibrate.load(path=target)
    assert loaded is not None
    assert loaded.method == "test"
    assert loaded.frame_size == (640, 480)
    assert loaded.point_count == len(SQUARE_PIXELS)
    assert np.allclose(loaded.matrix, saved.matrix)
    assert len(loaded.table_polygon) >= 3


def test_load_missing_file_is_none(tmp_path):
    assert calibrate.load(path=tmp_path / "nope.json") is None


def test_load_garbage_is_none_not_an_exception(tmp_path):
    target = tmp_path / "homography.json"
    target.write_text("{ not json")
    assert calibrate.load(path=target) is None


def test_load_wrong_shaped_matrix_is_none(tmp_path):
    target = tmp_path / "homography.json"
    target.write_text('{"matrix": [[1,2],[3,4]], "frame_size": [640,480]}')
    assert calibrate.load(path=target) is None


# --- table polygon / workspace check (safety rule 3) ---------------------

SQUARE = ((0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4))


def test_point_inside_and_outside():
    assert calibrate.point_in_polygon(0.2, 0.2, SQUARE)
    assert not calibrate.point_in_polygon(0.5, 0.2, SQUARE)
    assert not calibrate.point_in_polygon(0.2, -0.1, SQUARE)


def test_empty_polygon_fails_closed():
    """An uncalibrated system must refuse everything, not allow everything."""
    assert not calibrate.point_in_polygon(0.2, 0.2, ())
    assert not calibrate.point_in_polygon(0.0, 0.0, ())


def test_degenerate_polygon_fails_closed():
    assert not calibrate.point_in_polygon(0.1, 0.1, ((0.0, 0.0), (0.4, 0.4)))


def test_non_finite_coordinates_fail_closed():
    assert not calibrate.point_in_polygon(float("nan"), 0.2, SQUARE)
    assert not calibrate.point_in_polygon(float("inf"), 0.2, SQUARE)


def test_polygon_is_shrunk_inside_the_measured_points():
    polygon = calibrate.polygon_from_points(list(SQUARE_ROBOT), margin=0.02)
    # Every calibrated corner should now sit OUTSIDE the shrunken polygon,
    # because extrapolating to the very edge of the data is what we are avoiding.
    assert not calibrate.point_in_polygon(0.10, 0.15, polygon)
    # ...while the middle of the table stays available.
    assert calibrate.point_in_polygon(0.25, 0.0, polygon)


def test_polygon_smaller_than_the_margin_is_refused():
    tiny = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)]
    with pytest.raises(calibrate.CalibrationError, match="safety margin"):
        calibrate.polygon_from_points(tiny, margin=0.05)


# --- ChArUco, end to end on a synthetic board ---------------------------
# Renders the configured board, detects it, and fits a map. This exercises the
# whole primary calibration path — board geometry, detection, board->robot
# placement and the fit — without a camera, a printer or the operator. If this
# breaks, calibration day breaks.


def _synthetic_board_frame():
    import cv2
    import numpy as np

    board = calibrate.board_image(pixels_per_square=90)
    frame = np.full((config.CAMERA_HEIGHT, config.CAMERA_WIDTH), 255, np.uint8)
    height, width = board.shape[:2]
    scale = min((config.CAMERA_HEIGHT - 60) / height, (config.CAMERA_WIDTH - 80) / width)
    small = cv2.resize(board, (int(width * scale), int(height * scale)))
    top = (config.CAMERA_HEIGHT - small.shape[0]) // 2
    left = (config.CAMERA_WIDTH - small.shape[1]) // 2
    frame[top : top + small.shape[0], left : left + small.shape[1]] = small
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def test_charuco_board_is_detected_and_fits_cleanly():
    pixels, robots = calibrate.charuco_correspondences(
        _synthetic_board_frame(), origin=(0.20, -0.06)
    )
    assert len(pixels) >= config.CALIB_MIN_POINTS
    assert all(isinstance(v, float) for point in robots for v in point)
    _, mean_error, _ = calibrate.compute(pixels, robots)
    assert mean_error < 1.0, "a flat, undistorted board must fit to well under a pixel"


def test_charuco_origin_offset_shifts_every_robot_point():
    frame = _synthetic_board_frame()
    _, base = calibrate.charuco_correspondences(frame, origin=(0.0, 0.0))
    _, shifted = calibrate.charuco_correspondences(frame, origin=(0.10, -0.05))
    for (bx, by), (sx, sy) in zip(base, shifted):
        assert sx == pytest.approx(bx + 0.10)
        assert sy == pytest.approx(by - 0.05)


def test_charuco_rotation_rotates_the_board_frame():
    frame = _synthetic_board_frame()
    _, base = calibrate.charuco_correspondences(frame, origin=(0.0, 0.0), rotation_deg=0.0)
    _, turned = calibrate.charuco_correspondences(frame, origin=(0.0, 0.0), rotation_deg=90.0)
    # A 90 deg CCW turn maps board (bx, by) to robot (-by, bx).
    for (bx, by), (tx, ty) in zip(base, turned):
        assert tx == pytest.approx(-by, abs=1e-9)
        assert ty == pytest.approx(bx, abs=1e-9)


def test_charuco_on_a_blank_frame_is_refused():
    import numpy as np

    blank = np.full((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), 255, np.uint8)
    with pytest.raises(calibrate.CalibrationError, match="ChArUco"):
        calibrate.charuco_correspondences(blank, origin=(0.2, 0.0))


def test_config_polygon_is_a_tuple_of_pairs():
    """Whatever calibration state the machine is in, the config must be well formed."""
    for point in config.TABLE_POLYGON:
        assert len(point) == 2
        assert all(isinstance(v, float) for v in point)
