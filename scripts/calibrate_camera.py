#!/usr/bin/env python
"""Build the pixel -> robot map. OPERATOR REQUIRED.

Two methods:

  --method charuco   Lay the printed ChArUco board flat on the table, tell this
                     script where its origin corner sits in robot coordinates,
                     and one capture builds the map. Fast. Its accuracy is
                     limited by how well you can measure the board's position
                     with a ruler.

  --method tip       Touch the closed gripper to a spot on the table, press
                     ENTER (forward kinematics reads the robot XY exactly), then
                     click that same spot in the photo. Repeat 6+ times spread
                     across the table. Slower, MOVES NOTHING BY ITSELF — you
                     pose the arm by hand with torque released — and needs no
                     ruler at all, because the robot measures itself.

  --print-board      Write a printable board PNG and exit. Print it at 100%
                     scale (no "fit to page"), then MEASURE a square with a
                     ruler and set ARMANI_CHARUCO_SQUARE_M to what you measured.

Nothing is saved if the reprojection check fails — a bad map is worse than none.
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

from armani import calibrate, config, eyes, kinematics  # noqa: E402
from armani.logutil import log_event  # noqa: E402

WINDOW = "ARM-ANI calibration — click the gripper tip, or press s to skip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("charuco", "tip"), default="charuco")
    parser.add_argument("--dry-run", action="store_true", help="explain the plan, touch nothing")
    parser.add_argument("--print-board", action="store_true", help="write a printable board and exit")
    parser.add_argument("--points", type=int, default=8, help="tip method: how many points to collect")
    args = parser.parse_args()

    print("=== ARM-ANI camera calibration ===")
    print(f"method     : {args.method}")
    print(f"output     : {config.HOMOGRAPHY_PATH}")
    print(f"error limit: {config.CALIB_MAX_REPROJECTION_PX:.0f} px mean reprojection")

    existing = calibrate.load()
    if existing is not None:
        print(f"\nexisting   : {calibrate.describe(existing)}")

    if args.print_board:
        return _print_board(args.dry_run)

    if args.dry_run:
        print("\n[dry-run] would:")
        print("[dry-run]   capture a frame from the C920")
        if args.method == "charuco":
            print(f"[dry-run]   detect a {config.CHARUCO_SQUARES_X}x{config.CHARUCO_SQUARES_Y} "
                  f"ChArUco board ({config.CHARUCO_SQUARE_M * 1000:.0f} mm squares)")
            print("[dry-run]   ask for the board origin's robot XY and rotation")
        else:
            print(f"[dry-run]   release torque and collect {args.points} tip points by hand")
            print("[dry-run]   ask you to click each tip position in the photo")
        print("[dry-run]   fit a homography, check reprojection error, and save only if it passes")
        print("[dry-run] nothing written — that would falsely mark the system calibrated.")
        return 0

    try:
        if args.method == "charuco":
            pixels, robots, frame = _collect_charuco()
        else:
            pixels, robots, frame = _collect_tip(args.points)
    except KeyboardInterrupt:
        print("\nInterrupted — nothing saved.", file=sys.stderr)
        return 1
    except (calibrate.CalibrationError, eyes.EyesError, kinematics.KinematicsUnavailable) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    height, width = frame.shape[:2]
    try:
        matrix, mean_error, max_error = calibrate.compute(pixels, robots)
        print(f"\nfit: {len(pixels)} points, mean {mean_error:.1f} px, max {max_error:.1f} px")
        homography = calibrate.save(matrix, pixels, robots, (width, height), args.method)
    except calibrate.CalibrationError as exc:
        print(f"\nNOT SAVED: {exc}", file=sys.stderr)
        log_event("homography_rejected", method=args.method, error=str(exc))
        return 1

    _report(homography)
    return 0


def _print_board(dry_run: bool) -> int:
    out_path = config.TEST_OUT_DIR / "charuco_board.png"
    if dry_run:
        print(f"[dry-run] would write {out_path}")
        return 0
    import cv2

    config.TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = calibrate.board_image()
    if not cv2.imwrite(str(out_path), image):
        print(f"could not write {out_path}", file=sys.stderr)
        return 1
    print(f"\nWrote {out_path}")
    print(f"  Board: {config.CHARUCO_SQUARES_X}x{config.CHARUCO_SQUARES_Y}, "
          f"{config.CHARUCO_SQUARE_M * 1000:.0f} mm squares, dictionary {config.CHARUCO_DICT}")
    print("  Print at 100% scale — NOT 'fit to page', which silently rescales it.")
    print("  Then MEASURE one square with a ruler. If it is not "
          f"{config.CHARUCO_SQUARE_M * 1000:.0f} mm, set ARMANI_CHARUCO_SQUARE_M to the real value:")
    print("  a wrong square size scales the entire map linearly and every reach will be off.")
    return 0


def _collect_charuco():
    """One capture, many corners, plus the operator's measurement of the board."""
    print(
        "\nLay the printed board FLAT on the table, fully inside the camera view.\n"
        "Its origin corner is the inner chessboard corner closest to the board's\n"
        "top-left as printed (the first corner, id 0).\n"
    )
    origin_x = _ask_float("Robot X of the board's origin corner (metres, forward from base)")
    origin_y = _ask_float("Robot Y of the board's origin corner (metres, +Y to the robot's left)")
    rotation = _ask_float("Board rotation from robot +X (degrees CCW, 0 if aligned)", default=0.0)

    input("\nPress ENTER to capture... ")
    frame = eyes.capture_frame()

    # Try both board handednesses and keep the one that actually fits. A sheet
    # of paper laid face-up and OpenCV's board coordinates do not always agree
    # on which way Y runs, and guessing wrong produces a plausible-looking map
    # that is quietly mirrored. The fit itself settles it; the error ceiling in
    # save() still has the final word either way.
    best = None
    for mirror in (False, True):
        pixels, robots = calibrate.charuco_correspondences(
            frame, (origin_x, origin_y), rotation, mirror=mirror
        )
        try:
            _, mean_error, _ = calibrate.compute(pixels, robots)
        except calibrate.CalibrationError as exc:
            print(f"  mirror={mirror}: could not fit ({exc})")
            continue
        print(f"  mirror={mirror}: {len(pixels)} corners, mean {mean_error:.2f} px")
        if best is None or mean_error < best[0]:
            best = (mean_error, mirror, pixels, robots)

    if best is None:
        raise calibrate.CalibrationError("neither board orientation produced a usable fit")

    mean_error, mirror, pixels, robots = best
    print(f"using board handedness mirror={mirror} ({mean_error:.2f} px)")
    return pixels, robots, frame


def _collect_tip(wanted: int):
    """Hand-posed gripper tip points. The robot supplies its own ground truth."""
    from armani import motion, safety

    if not kinematics.available():
        raise kinematics.KinematicsUnavailable(
            "the tip method needs forward kinematics to know where the gripper is"
        )
    if not safety.require_operator("release torque so you can move the gripper by hand"):
        raise calibrate.CalibrationError("operator did not confirm presence")

    print(
        f"\nYou will place the gripper on {wanted} spots spread across the table.\n"
        "  * SPREAD THEM OUT — corners and middle. Points along a line cannot define a plane.\n"
        "  * Keep the gripper roughly VERTICAL when touching down: the maths uses the\n"
        "    gripper frame's XY, which only sits above the fingertip when it is upright.\n"
        "  * Do not move the camera or the table at any point.\n"
    )

    arm = motion.connect()
    pixels: list[tuple[float, float]] = []
    robots: list[tuple[float, float]] = []
    frame = None
    try:
        print("SUPPORT THE ARM — torque is about to release and it will go limp.")
        input("Press ENTER when you are holding it... ")
        _drain_stdin()

        with arm.torque_disabled():
            print("\nTorque OFF. Move the gripper by hand.\n")
            for index in range(wanted):
                print(f"--- point {index + 1} of {wanted} ---")
                input("  Touch the gripper tip to a spot on the table, hold it, press ENTER... ")
                pose = arm.read_positions()
                x, y, z = kinematics.tool_position(pose)
                tilt = kinematics.tilt_from_down(kinematics.forward(pose))
                print(f"  robot XY = ({x:+.4f}, {y:+.4f}) m   z = {z:+.4f} m   lean = {tilt:.0f} deg")
                if tilt > TIP_TILT_WARN_DEG:
                    print(f"  WARNING: the gripper is leaning {tilt:.0f} deg. Its frame origin is "
                          "then NOT above the fingertip and this point will be off. Re-do it upright.")

                frame = eyes.capture_frame()
                pixel = _click_point(frame, index + 1, wanted)
                if pixel is None:
                    print("  skipped.")
                    continue
                print(f"  pixel = {pixel}")
                pixels.append(pixel)
                robots.append((x, y))

            # Park the goal where the arm actually is before torque returns, or
            # the servos snap back to a stale target the moment the block exits.
            arm.send(arm.read_positions())
            print("\nTorque coming back on — keep supporting the arm.")
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)

    if frame is None or len(pixels) < config.CALIB_MIN_POINTS:
        raise calibrate.CalibrationError(
            f"collected {len(pixels)} usable points, need {config.CALIB_MIN_POINTS}"
        )
    return pixels, robots, frame


# A gripper leaning more than this when touching down puts its frame origin
# measurably away from the fingertip the operator is about to click.
TIP_TILT_WARN_DEG = 20.0


def _click_point(frame, index: int, total: int):
    """Show the frame and return the clicked pixel, or None if skipped."""
    import cv2

    clicked: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((float(x), float(y)))

    canvas = frame.copy()
    cv2.putText(
        canvas, f"point {index}/{total}: click the GRIPPER TIP  (s = skip)",
        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
    )
    cv2.imshow(WINDOW, canvas)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while not clicked:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("s"), 27):  # s or ESC
                return None
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyWindow(WINDOW)
        # macOS needs a few event-loop turns before the window actually goes.
        for _ in range(4):
            cv2.waitKey(1)
    return clicked[0]


def _drain_stdin() -> None:
    """Discard buffered newlines so a stray ENTER does not auto-advance."""
    try:
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:  # not a tty, or not a termios platform
        pass


def _ask_float(prompt: str, default: float | None = None) -> float:
    suffix = "" if default is None else f" [{default:g}]"
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("    not a number")


def _report(homography) -> None:
    print("\n" + "=" * 68)
    print("  CALIBRATION SAVED")
    print("=" * 68)
    print(f"  {calibrate.describe(homography)}")
    print(f"\n  Table polygon ({len(homography.table_polygon)} corners, robot metres):")
    for x, y in homography.table_polygon:
        print(f"    ({x:+.3f}, {y:+.3f})")
    print(
        "\n  DO NOT MOVE THE CAMERA OR THE TABLE.\n"
        "  This map is only valid for the tripod position and table position it was\n"
        "  measured at. If either is bumped — even slightly — every coordinate the arm\n"
        "  computes from vision is wrong, and it will reach confidently to the wrong\n"
        "  place. Re-run this script (about 5 minutes) if that happens.\n"
    )
    print("  Next: python tests/smoke_10_hover.py")


if __name__ == "__main__":
    raise SystemExit(main())
