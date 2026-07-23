#!/usr/bin/env python
"""Spike S1 — hover the gripper over a named object using the ChArUco calibration.

Reuses the stage-4 motion path (grasp.hover_over) unchanged: Gemini locates the
object's CONTACT point with the table -> pixel -> robot XY via the calibration ->
workspace polygon check -> IK hover at the safe hover height -> interp move.

    python scripts/hover_object.py --pixel 320 240        # DRY RUN: chain for a synthetic pixel
    python scripts/hover_object.py --object "red block"    # DRY RUN: real vision, no motion
    python scripts/hover_object.py --object "red block" --live   # OPERATOR + ARM: really hover

--dry-run is the default. Nothing moves without --live.
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

from armani import calibrate, config, eyes, grasp, kinematics, safety  # noqa: E402

DEFAULT_OBJECT = next(iter(config.OBJECT_CATALOG))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", default=DEFAULT_OBJECT, help="what to hover over")
    parser.add_argument("--pixel", nargs=2, type=int, metavar=("U", "V"),
                        help="use this pixel instead of the camera (dry-run only)")
    parser.add_argument("--live", action="store_true", help="OPERATOR + ARM: actually hover")
    parser.add_argument("--centre", action="store_true",
                        help="point at the visual centre instead of the table contact point")
    args = parser.parse_args()

    print("=== hover over object (Spike S1) ===")
    print("  *** the C920 must be where it was during calibration ***")

    homography = calibrate.load()
    if homography is None:
        print("not calibrated — run scripts/calibrate_charuco.py --run first.", file=sys.stderr)
        return 1
    print(f"  calibration: {calibrate.describe(homography)}")
    if not config.TABLE_POLYGON:
        print("the saved calibration has no table polygon; re-run calibration.", file=sys.stderr)
        return 1
    if not kinematics.available():
        print("inverse kinematics unavailable (placo/URDF). Cannot plan a hover.", file=sys.stderr)
        return 1

    # --- get the pixel ---
    if args.pixel is not None:
        pixel = (int(args.pixel[0]), int(args.pixel[1]))
        print(f"\npixel (synthetic): {pixel}")
    else:
        if config.api_key("GOOGLE_API_KEY") is None:
            print("GOOGLE_API_KEY not set; pass --pixel U V for a headless dry run.", file=sys.stderr)
            return 1
        try:
            frame = eyes.capture_frame()
        except eyes.EyesError as exc:
            print(f"no camera frame: {exc}", file=sys.stderr)
            return 1
        if homography.frame_size != (frame.shape[1], frame.shape[0]):
            print(f"frame {frame.shape[1]}x{frame.shape[0]} != calibration "
                  f"{homography.frame_size}; the map does not apply.", file=sys.stderr)
            return 1
        detection = eyes.locate(args.object, frame=frame, contact_point=not args.centre)
        if detection is None:
            print(f"could not see a {args.object}.", file=sys.stderr)
            return 1
        pixel = detection.point
        print(f"\n{args.object!r} {'(contact point)' if not args.centre else '(centre)'}: "
              f"pixel {pixel}, vision confidence {detection.confidence:.2f}")

    # --- pixel -> robot ---
    try:
        x, y = homography.to_robot(pixel)
    except calibrate.CalibrationError as exc:
        print(f"could not map pixel to the table: {exc}", file=sys.stderr)
        return 1
    print(f"robot XY: ({x:+.4f}, {y:+.4f}) m   hover z: {config.hover_z():.3f} m")
    on_table = calibrate.point_in_polygon(x, y, margin_m=config.POLYGON_MARGIN_M)
    print(f"on table: {'yes' if on_table else 'NO — outside the polygon'}")

    # --- plan (IK) ---
    plan = grasp.plan_hover(x, y, config.HOME_POSE)
    if plan.joints:
        print("\nIK joint targets:")
        for joint in config.IK_JOINTS:
            print(f"  {joint:<15} {plan.joints[joint]:+8.2f} deg")
        print(f"  position error: {plan.position_error_m*1000:.1f} mm   lean: {plan.tilt_deg:.1f} deg")
    print(f"reachable: {plan.ok}" + ("" if plan.ok else f"  ({plan.reason})"))

    if not args.live:
        print("\n[dry-run] chain printed above; nothing moved. Add --live to hover for real.")
        return 0

    return _run_live(args.object, x, y, plan)


def _run_live(object_name: str, x: float, y: float, plan) -> int:
    from armani import motion

    if not plan.ok:
        print(f"not reachable, refusing to move: {plan.reason}", file=sys.stderr)
        return 1
    if not safety.require_operator(f"hover over {object_name!r} at ({x:+.3f}, {y:+.3f}) m"):
        return 1
    safety.clear_stop()
    safety.install_kill_switch()
    safety.warn_kill_switch_untrusted()
    print("kill switch armed: ESC / Ctrl-C freezes the arm.")
    try:
        arm = motion.connect()
    except Exception as exc:
        print(f"could not connect: {type(exc).__name__}: {exc}", file=sys.stderr)
        safety.release_kill_switch()  # connect failed before the finally below could run
        return 1

    try:
        start_pose = arm.read_positions()
        result = grasp.hover_over(arm, x, y)
        print(f"\n{'HOVERED' if result.ok else 'did not hover'}: {result.reason or 'ok'}")
        if result.ok:
            print("  measure the tip's horizontal offset from the object and log it (target <=1.5 cm).")
            try:
                input("  press ENTER to return to the start pose... ")
            except (EOFError, KeyboardInterrupt):
                print()
            motion.goto(arm, {j: start_pose[j] for j in config.IK_JOINTS},
                        duration=config.RECOVERY_DURATION_S, profile="recorded")
        return 0 if result.ok else 1
    except KeyboardInterrupt:
        print("\ninterrupted — the arm holds where it is.")
        return 1
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)
        # Stop the ESC listener + release cv2 handles before exit (macOS teardown
        # segfault guard). No-op if the listener was never installed.
        safety.release_kill_switch()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
