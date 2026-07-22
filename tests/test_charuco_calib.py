"""Spike S1 — the ChArUco/rigid calibration math. No camera, no arm.

The rendered-board tests use cv2.aruco (present in the lerobot env); everything
else is pure numpy. These cover the traps stage 4 burned us on — units (mm vs m)
and transform composition — with synthetic inputs that have known answers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import calibrate, config, eyes  # noqa: E402


def rotation(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# --- rigid 2D fit --------------------------------------------------------


def test_rigid_fit_recovers_a_known_transform_exactly():
    src = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.08)]
    R, t = rotation(25.0), np.array([0.3, -0.12])
    dst = [tuple(R @ np.array(p) + t) for p in src]
    R_fit, t_fit, resid = calibrate.fit_rigid_2d(src, dst)
    assert np.allclose(R_fit, R, atol=1e-9)
    assert np.allclose(t_fit, t, atol=1e-9)
    assert max(resid) < 1e-9


def test_rigid_fit_is_a_proper_rotation_not_a_reflection():
    """A noisy 3-point fit must not flip into a mirror (det must be +1)."""
    src = [(0.0, 0.0), (0.1, 0.0), (0.05, 0.09)]
    R = rotation(40.0)
    dst = [tuple(R @ np.array(p) + np.array([0.2, 0.05])) for p in src]
    rng = np.random.default_rng(3)
    dst = [(x + rng.normal(0, 3e-4), y + rng.normal(0, 3e-4)) for x, y in dst]
    R_fit, _, _ = calibrate.fit_rigid_2d(src, dst)
    assert np.linalg.det(R_fit) == pytest.approx(1.0, abs=1e-6)


def test_rigid_fit_preserves_distance_no_scale():
    """Rigid means no scale: inter-point distances are unchanged by the fit."""
    src = [(0.0, 0.0), (0.10, 0.0), (0.0, 0.10)]
    dst = [tuple(rotation(15.0) @ np.array(p) + np.array([0.5, 0.5])) for p in src]
    R_fit, t_fit, _ = calibrate.fit_rigid_2d(src, dst)
    mapped = [R_fit @ np.array(p) + t_fit for p in src]
    assert np.hypot(*(mapped[1] - mapped[0])) == pytest.approx(0.10, abs=1e-9)


def test_rigid_fit_rejects_too_few_points():
    with pytest.raises(calibrate.CalibrationError):
        calibrate.fit_rigid_2d([(0.0, 0.0)], [(1.0, 1.0)])


# --- composition ---------------------------------------------------------


def test_compose_equals_rigid_of_homography():
    # A homography pixel->board (here a simple affine is enough to check algebra).
    H = np.array([[0.0004, 0.0, 0.01], [0.0, 0.0004, -0.02], [0.0, 0.0, 1.0]])
    R, t = rotation(33.0), np.array([0.25, -0.05])
    M = calibrate.compose_pixel_to_robot(H, R, t)
    for px in [(100.0, 120.0), (400.0, 300.0), (10.0, 470.0)]:
        board = np.array(calibrate.pixel_to_robot(H, px))
        expected = R @ board + t
        got = np.array(calibrate.pixel_to_robot(M, px))
        assert np.allclose(got, expected, atol=1e-9)


def test_rigid_affine_3x3_packs_correctly():
    R, t = rotation(90.0), np.array([1.0, 2.0])
    A = calibrate.rigid_affine_3x3(R, t)
    assert A.shape == (3, 3)
    assert np.allclose(A[:2, :2], R)
    assert np.allclose(A[:2, 2], t)
    assert np.allclose(A[2], [0, 0, 1])


# --- aruco version dispatch ----------------------------------------------


@pytest.mark.parametrize("version,expected", [
    ("4.6.0", "legacy"),
    ("4.7.0", "modern"),
    ("4.13.0", "modern"),
    ("5.0.0", "modern"),
    ("4.11.0-dev", "modern"),
    ("garbage", "modern"),
])
def test_aruco_api_dispatch(version, expected):
    assert calibrate.choose_aruco_api(version) == expected


# --- measured square (mm vs m) -------------------------------------------


def test_measured_square_prefers_the_env_mm(monkeypatch):
    monkeypatch.setenv("ARMANI_CHARUCO_SQUARE_MM", "31.4")
    assert calibrate.measured_square_m() == pytest.approx(0.0314)


def test_measured_square_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("ARMANI_CHARUCO_SQUARE_MM", raising=False)
    assert calibrate.measured_square_m() == pytest.approx(config.CHARUCO_SQUARE_M)


def test_measured_square_ignores_garbage(monkeypatch):
    monkeypatch.setenv("ARMANI_CHARUCO_SQUARE_MM", "not-a-number")
    assert calibrate.measured_square_m() == pytest.approx(config.CHARUCO_SQUARE_M)


def test_calibration_corner_ids_are_distinct_and_in_range():
    ids = calibrate.calibration_corner_ids()
    inner = (config.CHARUCO_SQUARES_X - 1) * (config.CHARUCO_SQUARES_Y - 1)
    assert len(set(ids)) == 3
    assert all(0 <= i < inner for i in ids)


# --- full pixel->board fit on a rendered board ---------------------------


def _rendered_board():
    board = calibrate.charuco_board(square_m=config.CHARUCO_SQUARE_M)
    image = board.generateImage(
        (config.CHARUCO_SQUARES_X * 140, config.CHARUCO_SQUARES_Y * 140), marginSize=20
    )
    return board, image


def test_detect_and_fit_a_clean_board_is_subpixel():
    board, image = _rendered_board()
    pixels, board_m, ids = calibrate.detect_charuco(image, board)
    inner = (config.CHARUCO_SQUARES_X - 1) * (config.CHARUCO_SQUARES_Y - 1)
    assert len(pixels) == inner
    _, rms = calibrate.fit_board_homography(pixels, board_m)
    assert rms < 1.0, f"a clean render should fit sub-pixel, got {rms:.2f} px"


def test_board_mm_corners_are_in_metres():
    """getChessboardCorners must be metres (board built with square in metres).
    A 30 mm square board's corners span < 0.25 m, never hundreds (that would be
    the mm-vs-m bug)."""
    board, image = _rendered_board()
    _, board_m, _ = calibrate.detect_charuco(image, board)
    xs = [p[0] for p in board_m]
    assert max(xs) < 0.25


def test_board_homography_fit_rejects_too_few():
    with pytest.raises(calibrate.CalibrationError):
        calibrate.fit_board_homography([(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (0.1, 0.1)])


# --- save / load a composed charuco calibration --------------------------

# A synthetic affine pixel->robot that spreads a 640x480 frame across ~0.4 m.
SYNTH_M = np.array([[0.001, 0.0, 0.15], [0.0, 0.001, -0.20], [0.0, 0.0, 1.0]])
CORNERS = [(100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)]


def test_save_charuco_round_trip(tmp_path):
    path = tmp_path / "homography.json"
    saved = calibrate.save_charuco(
        SYNTH_M, CORNERS, (640, 480),
        board_rms_px=2.3,
        tip_board_m=[(0.0, 0.0), (0.09, 0.0), (0.0, 0.15)],
        tip_robot_m=[(0.15, -0.20), (0.24, -0.20), (0.15, -0.05)],
        rigid_residuals_m=[0.0004, 0.0006, 0.0003],
        path=path,
    )
    assert saved.method == "charuco_rigid"
    loaded = calibrate.load(path=path)
    assert loaded is not None
    assert np.allclose(loaded.matrix, SYNTH_M)
    assert len(loaded.table_polygon) >= 3
    payload = json.loads(path.read_text())
    assert payload["charuco"]["board_rms_px"] == 2.3
    assert payload["charuco"]["rigid_residual_mm"] == [0.4, 0.6, 0.3]


def test_save_charuco_refuses_a_bad_board_rms(tmp_path):
    path = tmp_path / "homography.json"
    with pytest.raises(calibrate.CalibrationError, match="NOT SAVED"):
        calibrate.save_charuco(
            SYNTH_M, CORNERS, (640, 480),
            board_rms_px=config.CALIB_MAX_REPROJECTION_PX + 5,
            tip_board_m=[(0.0, 0.0)], tip_robot_m=[(0.0, 0.0)], rigid_residuals_m=[0.0],
            path=path,
        )
    assert not path.exists()


def test_roundtrip_pixel_errors_are_near_zero_for_an_invertible_matrix():
    errs = calibrate.roundtrip_pixel_errors(SYNTH_M, CORNERS)
    assert max(errs) < 1e-6


# --- eyes contact-point prompt (additive kwarg) --------------------------


def test_contact_prompt_variants_ask_for_the_table_contact():
    for template in eyes.CONTACT_PROMPT_VARIANTS:
        rendered = eyes.render_prompt(template, "red block")
        assert "TABLE" in rendered.upper()
        assert "red block" in rendered
        assert "[y, x]" in rendered  # still the same point format


def test_contact_and_centre_prompts_differ():
    assert set(eyes.CONTACT_PROMPT_VARIANTS) != set(eyes.PROMPT_VARIANTS)


def test_locate_still_rejects_empty_name():
    with pytest.raises(ValueError):
        eyes.locate("  ", contact_point=True)
