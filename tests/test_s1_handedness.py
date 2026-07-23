"""S1 fix 2: board-frame handedness + rigid-residual save gate. No hardware.

The ChArUco rigid path fit y-DOWN board coordinates (OpenCV's getChessboardCorners)
straight against the y-UP robot frame — opposite handedness. fit_rigid_2d's
reflection guard then produced garbage residuals (the operator saw 92/77/19 mm and
it SAVED anyway, hover off by 3–4 cm). The fix normalizes the board frame to
right-handed at the source, keeps the guard, and adds a max-residual save gate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import calibrate, config  # noqa: E402


def _rot(theta_deg: float) -> np.ndarray:
    a = math.radians(theta_deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _yflip(points):
    ys = [p[1] for p in points]
    height = min(ys) + max(ys)
    return [(x, height - y) for x, y in points]


# --- fit_rigid_2d reflection handling ------------------------------------
def test_reflection_permitted_collapses_opposite_handed_fit():
    # Opposite-handed inputs: guard-on gives huge residuals, reflection-permitted small.
    board = [(0.03, 0.03), (0.12, 0.03), (0.03, 0.18)]              # CCW
    robot = [(0.3318, 0.0766), (0.3385, -0.0121), (0.1655, 0.0621)]  # CW (real operator data)
    _, _, forbidden = calibrate.fit_rigid_2d(board, robot, allow_reflection=False)
    _, _, permitted = calibrate.fit_rigid_2d(board, robot, allow_reflection=True)
    assert max(forbidden) > 0.05                    # >50 mm garbage
    assert max(permitted) < 0.015                   # collapses to ~noise
    assert max(forbidden) > 5 * max(permitted)      # reflection-class error


def test_yflip_recovers_a_pure_rotation_with_the_guard_ON():
    # True relationship: robot = R(37 deg) @ yup(board) + t, a proper rotation of the
    # right-handed board. Fitting the y-DOWN board fights the guard; fitting the
    # y-FLIPPED board recovers it exactly.
    board_ydown = [(0.03, 0.03), (0.12, 0.03), (0.03, 0.18), (0.12, 0.18)]
    board_yup = _yflip(board_ydown)
    R, t = _rot(37.0), np.array([0.25, -0.05])
    robot = [tuple(R @ np.array(p) + t) for p in board_yup]

    _, _, res_nofix = calibrate.fit_rigid_2d(board_ydown, robot)  # guard on, wrong frame
    _, _, res_fix = calibrate.fit_rigid_2d(board_yup, robot)      # guard on, fixed frame
    assert max(res_nofix) > 0.05
    assert max(res_fix) < 1e-9


# --- chessboard_corners_yup ----------------------------------------------
class _FakeBoard:
    def __init__(self, corners):
        self._c = np.asarray(corners, dtype=np.float32)

    def getChessboardCorners(self):
        return self._c


def test_chessboard_corners_yup_flips_y_about_the_corner_centre():
    board = _FakeBoard([[0.03, 0.03, 0.0], [0.12, 0.03, 0.0], [0.03, 0.18, 0.0]])
    out = calibrate.chessboard_corners_yup(board)
    # y about (0.03+0.18)=0.21: 0.03->0.18, 0.18->0.03; x untouched; z dropped.
    assert np.allclose(out, [[0.03, 0.18], [0.12, 0.18], [0.03, 0.03]])


def test_chessboard_corners_yup_flip_constant_is_over_the_full_set():
    # Discriminating: the flip must reflect about the FULL corner set's centre, not a
    # subset's. Full y-extent [0.0, 0.20] -> constant 0.20; the y=0.05 corner must map
    # to 0.15. A flip about the first two corners (extent 0.05) would give 0.0 instead.
    board = _FakeBoard([[0.0, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.10, 0.0], [0.0, 0.20, 0.0]])
    out = calibrate.chessboard_corners_yup(board)
    assert out[1][1] == pytest.approx(0.15)
    assert out[0][1] == pytest.approx(0.20) and out[3][1] == pytest.approx(0.0)


def _rendered_board():
    board = calibrate.charuco_board(square_m=config.CHARUCO_SQUARE_M)
    image = board.generateImage(
        (config.CHARUCO_SQUARES_X * 140, config.CHARUCO_SQUARES_Y * 140), marginSize=20
    )
    return board, image


def test_detect_charuco_returns_the_yflipped_frame():
    # Pins the load-bearing consistency property: detect_charuco's board_m is the
    # y-UP helper's frame (not raw getChessboardCorners), so it shares the tip
    # targets' frame. A regression reverting detect_charuco to raw corners fails here.
    board, image = _rendered_board()
    _, board_m, ids = calibrate.detect_charuco(image, board)
    yup = calibrate.chessboard_corners_yup(board)
    raw = np.asarray(board.getChessboardCorners(), dtype=float)[:, :2]
    assert len(board_m) >= 6
    for (bx, by), cid in zip(board_m, ids):
        assert (bx, by) == pytest.approx(tuple(yup[cid]), abs=1e-9)   # matches y-up helper
        if not np.isclose(by, raw[cid][1]):                            # inverted vs raw (off-centre)
            assert by == pytest.approx(raw[:, 1].min() + raw[:, 1].max() - raw[cid][1], abs=1e-9)


# --- distance_table -------------------------------------------------------
def test_distance_table_math_and_names_the_mis_touched_corner():
    board = [(0.03, 0.03), (0.12, 0.03), (0.03, 0.18)]  # d01=90 d02=150 d12=175 mm
    robot = [(0.0, 0.0), (0.09, 0.0), (0.0, 0.165)]     # corner 2 pulled 15 mm short in y
    rows, suspect = calibrate.distance_table(board, robot)
    by_pair = {(i, j): (db, dr, delta) for i, j, db, dr, delta in rows}
    assert by_pair[(0, 1)][0] == pytest.approx(90.0, abs=0.1)
    assert by_pair[(0, 1)][2] == pytest.approx(0.0, abs=0.1)   # 0-1 agrees
    assert by_pair[(0, 2)][2] == pytest.approx(15.0, abs=0.1)  # 0-2 off by 15 mm
    assert suspect == 2                                        # corner 2 is the culprit


def test_distance_table_no_suspect_when_all_consistent():
    board = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)]
    robot = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)]  # identical -> all deltas 0
    _, suspect = calibrate.distance_table(board, robot)
    assert suspect is None


# --- residual save gate ---------------------------------------------------
def _valid_save_kwargs(tmp_path, residuals_m, tip_board=None, tip_robot=None):
    # A minimal but non-degenerate calibration: identity-ish pixel->robot on a square.
    board = tip_board or [(0.03, 0.03), (0.12, 0.03), (0.03, 0.18)]
    robot = tip_robot if tip_robot is not None else list(board)  # identical -> distance gate passes
    corner_pixels = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)]
    return dict(
        matrix=np.eye(3), corner_pixels=corner_pixels, frame_size=(640, 480), board_rms_px=0.5,
        tip_board_m=board, tip_robot_m=robot, rigid_residuals_m=residuals_m,
        path=tmp_path / "homography.json",
    )


def test_save_gate_refuses_a_high_max_residual(tmp_path):
    kwargs = _valid_save_kwargs(tmp_path, residuals_m=[0.001, 0.002, 0.0925])  # 92.5 mm
    with pytest.raises(calibrate.CalibrationError, match="residual"):
        calibrate.save_charuco(**kwargs)
    assert not (tmp_path / "homography.json").exists()  # nothing written


def test_save_allows_a_low_max_residual(tmp_path):
    kwargs = _valid_save_kwargs(tmp_path, residuals_m=[0.001, 0.002, 0.004])  # 4 mm
    calibrate.save_charuco(**kwargs)  # must not raise on either gate
    assert (tmp_path / "homography.json").exists()


def test_save_allows_residual_exactly_at_the_gate(tmp_path):
    # The gate is strict '>' so exactly MAX_RIGID_RESIDUAL_MM must be ALLOWED.
    at = config.MAX_RIGID_RESIDUAL_MM / 1000.0
    calibrate.save_charuco(**_valid_save_kwargs(tmp_path, residuals_m=[0.001, 0.002, at]))
    assert (tmp_path / "homography.json").exists()


def test_save_refuses_empty_residuals_fail_closed(tmp_path):
    # No accuracy evidence must fail closed, not be treated as a perfect fit.
    with pytest.raises(calibrate.CalibrationError, match="no rigid residuals"):
        calibrate.save_charuco(**_valid_save_kwargs(tmp_path, residuals_m=[]))
    assert not (tmp_path / "homography.json").exists()


def test_genuine_single_mistouch_is_gated_even_when_residual_hides_it(tmp_path):
    # Same-handedness single-corner mis-touch: corner 2 pulled 15 mm short in y.
    board = [(0.03, 0.03), (0.12, 0.03), (0.03, 0.18)]
    robot = [(0.03, 0.03), (0.12, 0.03), (0.03, 0.165)]
    _, _, res = calibrate.fit_rigid_2d(board, robot)  # guard ON, correct handedness
    # Procrustes SPREADS the 15 mm error, so the max residual slips UNDER the gate...
    assert max(res) * 1000.0 < config.MAX_RIGID_RESIDUAL_MM
    # ...but the distance-consistency gate refuses it (a rigid touch preserves distance).
    with pytest.raises(calibrate.CalibrationError, match="distance"):
        calibrate.save_charuco(**_valid_save_kwargs(tmp_path, list(res), tip_board=board, tip_robot=robot))
    assert not (tmp_path / "homography.json").exists()


def test_pairing_swap_is_gated(tmp_path):
    # Swapping two robot labels is a gross error; the gates refuse it regardless of
    # whether distance_table's suspect hint localizes it.
    board = [(0.03, 0.03), (0.12, 0.03), (0.03, 0.18)]
    swapped = [board[0], board[2], board[1]]
    _, _, res = calibrate.fit_rigid_2d(board, swapped)
    with pytest.raises(calibrate.CalibrationError):
        calibrate.save_charuco(**_valid_save_kwargs(tmp_path, list(res), tip_board=board, tip_robot=swapped))
    assert not (tmp_path / "homography.json").exists()


def test_config_rigid_residual_gate_is_sane():
    assert isinstance(config.MAX_RIGID_RESIDUAL_MM, float)
    assert 0.0 < config.MAX_RIGID_RESIDUAL_MM <= 20.0
