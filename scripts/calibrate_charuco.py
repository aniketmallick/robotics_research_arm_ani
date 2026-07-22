#!/usr/bin/env python
"""Spike S1 — ChArUco recalibration: pixel -> board homography + 3-touch rigid board -> robot.

    python scripts/calibrate_charuco.py            # DRY RUN (default): camera + RMS, no arm, no save
    python scripts/calibrate_charuco.py --run      # full calibration (OPERATOR + ARM), saves
    python scripts/calibrate_charuco.py --check     # validate the saved map (round trip)
    python scripts/calibrate_charuco.py --check --live-corners   # hover 2 corners for a ruler check

THE C920 MUST NOT MOVE once this starts — it is locked to the arm's workspace.

Flow: detect the board's sub-pixel corners -> fit pixel->board-mm homography
(gate: RMS <= 10 px) -> operator jogs the gripper tip to the 3 marked corners,
FK reads robot XY -> fit a RIGID 2D board->robot transform -> compose to one
pixel->robot map -> save in the existing calibration file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import _bootstrap  # noqa: E402,F401  (verifies the interpreter, exits if wrong)

from armani import calibrate, config, eyes, kinematics, safety  # noqa: E402

RMS_GATE_PX = 10.0  # time-box rule: above this, stop and report — no tip-click fallback


def banner() -> None:
    print("=== ChArUco recalibration (Spike S1) ===")
    print(f"  board  : {config.CHARUCO_SQUARES_X}x{config.CHARUCO_SQUARES_Y}, "
          f"square {calibrate.measured_square_m()*1000:.2f} mm (ARMANI_CHARUCO_SQUARE_MM)")
    print("  *** DO NOT MOVE THE C920 from here on — it is locked to the workspace. ***")


def detect_and_fit(frame):
    """Detect the board and fit pixel->board. Returns (pixels, board_m, ids, H, rms)."""
    board = calibrate.charuco_board()
    pixels, board_m, ids = calibrate.detect_charuco(frame, board)
    homography, rms = calibrate.fit_board_homography(pixels, board_m)
    return board, pixels, board_m, ids, homography, rms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="do the full calibration (arm + save)")
    parser.add_argument("--check", action="store_true", help="validate the saved calibration")
    parser.add_argument("--live-corners", action="store_true",
                        help="with --check: hover over 2 board corners for a ruler measurement")
    args = parser.parse_args()

    banner()
    if args.check:
        return run_check(args.live_corners)
    if args.run:
        return run_calibration()
    return run_dry()


# --- dry run: camera + RMS only, no arm, no save -------------------------


def run_dry() -> int:
    print("\n[dry-run] camera + homography fit only. No arm, no save. Use --run to calibrate.")
    if config.CAMERA_INDEX is None:
        print("  (set ARMANI_CAMERA_INDEX so a frame can be captured)")
    try:
        frame = eyes.capture_frame()
    except eyes.EyesError as exc:
        print(f"  no camera frame: {exc}")
        print(f"  would then ask you to touch marked corners {calibrate.calibration_corner_ids()}.")
        return 0
    try:
        _, _, _, _, _, rms = detect_and_fit(frame)
    except calibrate.CalibrationError as exc:
        print(f"  board not usable: {exc}")
        return 1
    verdict = "OK, proceed with --run" if rms <= RMS_GATE_PX else f"TOO HIGH (>{RMS_GATE_PX:.0f} px) — STOP"
    print(f"\n  board homography RMS = {rms:.2f} px  ->  {verdict}")
    print(f"  --run would then ask you to touch marked corners {calibrate.calibration_corner_ids()}.")
    return 0 if rms <= RMS_GATE_PX else 1


# --- full calibration ----------------------------------------------------


def run_calibration() -> int:
    from armani import motion

    print("\nStep 1/3 — capture the board. Make sure the WHOLE board is in view and the")
    print("           arm is NOT blocking it. Press ENTER to capture...")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return 1
    try:
        frame = eyes.capture_frame()
        board, pixels, board_m, ids, homography, rms = detect_and_fit(frame)
    except (eyes.EyesError, calibrate.CalibrationError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    frame_size = (frame.shape[1], frame.shape[0])
    print(f"  detected {len(pixels)} corners, board homography RMS = {rms:.2f} px")
    if rms > RMS_GATE_PX:
        print(f"\nSTOP: RMS {rms:.2f} px exceeds the {RMS_GATE_PX:.0f} px gate. Do NOT proceed.",
              file=sys.stderr)
        print("  Improve lighting/flatness/framing and re-run. No tip-click fallback (that failed).",
              file=sys.stderr)
        return 1

    if not kinematics.available():
        print("inverse kinematics unavailable, cannot read the tip via FK.", file=sys.stderr)
        return 1
    if not safety.require_operator("release torque so you can touch the board corners by hand"):
        print("Not confirmed — nothing done.")
        return 1
    safety.clear_stop()
    safety.install_kill_switch()

    print("\nStep 2/3 — touch the 3 marked corners. Connecting to the arm...")
    try:
        arm = motion.connect()
    except Exception as exc:
        print(f"could not connect: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    corner_ids = calibrate.calibration_corner_ids()
    board_points = board.getChessboardCorners()
    tip_board = [(float(board_points[c][0]), float(board_points[c][1])) for c in corner_ids]
    try:
        tip_robot = _touch_corners(arm, corner_ids, tip_board)
    except KeyboardInterrupt:
        print("\ninterrupted — nothing saved. Torque is back on; support the arm.", file=sys.stderr)
        return 1
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)
    if tip_robot is None:
        return 1

    print("\nStep 3/3 — fit and save.")
    rotation, translation, residuals = calibrate.fit_rigid_2d(tip_board, tip_robot)
    residual_mm = [r * 1000 for r in residuals]
    print(f"  rigid residuals (mm): {', '.join(f'{r:.2f}' for r in residual_mm)}")
    if max(residual_mm) > 5.0:
        print("  WARNING: a residual over 5 mm suggests a mis-touched corner or a board that moved.")

    matrix = calibrate.compose_pixel_to_robot(homography, rotation, translation)
    try:
        saved = calibrate.save_charuco(
            matrix, pixels, frame_size,
            board_rms_px=rms, tip_board_m=tip_board, tip_robot_m=tip_robot,
            rigid_residuals_m=residuals,
        )
    except calibrate.CalibrationError as exc:
        print(f"NOT SAVED: {exc}", file=sys.stderr)
        return 1

    print(f"\nSAVED: {calibrate.describe(saved)}")
    print("  Remove the board, place your objects, then: python scripts/hover_object.py --live")
    print("  Validate first with: python scripts/calibrate_charuco.py --check")
    return 0


def _touch_corners(arm, corner_ids, tip_board) -> list | None:
    """Release torque; operator moves the tip to each marked corner; FK reads XY."""
    import termios

    print("SUPPORT THE ARM — torque is about to release and it will go limp.")
    try:
        input("Press ENTER when you are holding it... ")
    except (EOFError, KeyboardInterrupt):
        return None
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass

    tip_robot: list = []
    with arm.torque_disabled():
        print("\nTorque OFF. Move the gripper TIP to each red-circled corner, then press ENTER.\n")
        for label, (cid, (bx, by)) in enumerate(zip(corner_ids, tip_board), start=1):
            try:
                input(f"  corner {label} (id {cid}, board ~({bx*1000:.0f},{by*1000:.0f}) mm): "
                      "tip on it, press ENTER... ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            pose = arm.read_positions()
            x, y, z = kinematics.tool_position(pose)
            tip_robot.append((float(x), float(y)))
            print(f"    robot XY = ({x:+.4f}, {y:+.4f}) m  (tip z {z:+.3f} m)")
        arm.send(arm.read_positions())  # park the goal before torque returns
        print("\nTorque coming back on — keep supporting the arm.")

    if len(tip_robot) != len(corner_ids):
        print(f"only {len(tip_robot)}/{len(corner_ids)} corners captured; not enough.", file=sys.stderr)
        return None
    return tip_robot


# --- validation ----------------------------------------------------------


def run_check(live_corners: bool) -> int:
    homography = calibrate.load()
    if homography is None:
        print("no saved calibration to check. Run --run first.", file=sys.stderr)
        return 1
    print(f"\n{calibrate.describe(homography)}")

    try:
        frame = eyes.capture_frame()
        board = calibrate.charuco_board()
        pixels, _, ids = calibrate.detect_charuco(frame, board)
    except (eyes.EyesError, calibrate.CalibrationError) as exc:
        print(f"could not re-detect the board for the check: {exc}", file=sys.stderr)
        return 1
    if homography.frame_size != (frame.shape[1], frame.shape[0]):
        print("WARNING: frame size differs from calibration; the map does not apply.", file=sys.stderr)

    errs = calibrate.roundtrip_pixel_errors(homography.matrix, pixels)
    print(f"\n  pixel->robot->pixel self-consistency: max {max(errs):.2e} px (should be ~0)")
    print("  (this only proves the matrix is invertible; accuracy is the rigid residuals + ruler)")

    if live_corners:
        return _live_corner_check(homography, board, pixels, ids)
    print("\n  For the honest end-to-end number, re-run with --live-corners (operator + arm).")
    return 0


def _live_corner_check(homography, board, pixels, ids) -> int:
    """Hover over 2 board corners from their pixel coords; operator measures the offset."""
    from armani import grasp, motion

    if not safety.require_operator("hover the arm over 2 board corners for a ruler check"):
        return 1
    safety.clear_stop()
    safety.install_kill_switch()
    try:
        arm = motion.connect()
    except Exception as exc:
        print(f"could not connect: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    try:
        chosen = calibrate.calibration_corner_ids()[:2]
        by_id = {cid: px for px, cid in zip(pixels, ids)}
        for cid in chosen:
            if cid not in by_id:
                continue
            x, y = homography.to_robot(by_id[cid])
            print(f"\ncorner id {cid}: pixel {by_id[cid]} -> robot ({x:+.3f},{y:+.3f}) m; hovering...")
            result = grasp.hover_over(arm, x, y)
            print(f"  {result.reason or 'hovering'}; measure the horizontal offset from the corner.")
            try:
                input("  press ENTER for the next corner... ")
            except (EOFError, KeyboardInterrupt):
                break
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)
    print("\nLog the ruler offsets in docs/spike_s1_results.md — that is the honest error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
