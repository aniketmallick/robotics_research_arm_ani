#!/usr/bin/env python
"""Smoke 10 — hover over a named object. OPERATOR + HARDWARE. THE ARM MOVES.

The whole stage-4 chain on one object:

    camera frame -> Gemini points at it -> homography maps the pixel to the
    table -> IK -> hover HOVER_HEIGHT_M above it -> hold -> return to start

Requires: a saved homography, a verified home pose, and you watching the arm.

The descent is stage 5, and this test proves it cannot happen yet: every action
actually sent to the motors is run through forward kinematics, and the test
FAILS if the commanded gripper height ever drops below where it started or where
it was going — whichever is lower.

    python tests/smoke_10_hover.py --dry-run
    python tests/smoke_10_hover.py --object "red block"
"""

from __future__ import annotations

import argparse
import time

from _bootstrap import EXIT_SKIP, banner, fail, ok, skip

from armani import calibrate, config, eyes, grasp, kinematics, motion, safety
from armani.logutil import log_event

HOLD_SECONDS = 2.0

# The joint-space path between two poses is not monotonic in tool height, so a
# legal move can dip slightly below both endpoints on its way. This is the
# allowance for that — small enough that a real descent (10 cm) cannot hide in it.
Z_PATH_TOLERANCE_M = 0.02

DEFAULT_OBJECT = next(iter(config.OBJECT_CATALOG))


class ZWatchArm:
    """Wraps the arm and forward-kinematics every commanded action.

    This is the coding guard that stage 4 never descends. It watches what is
    actually SENT, not what was planned, so a bug between the plan and the
    motors still gets caught.
    """

    def __init__(self, inner, start_pose: dict[str, float]) -> None:
        self._inner = inner
        self._pose = dict(start_pose)
        self.min_z = kinematics.tool_position(start_pose)[2]
        self.sends = 0

    @property
    def label(self) -> str:
        return self._inner.label

    def read_positions(self):
        pose = self._inner.read_positions()
        self._pose.update(pose)
        return pose

    def send(self, action):
        sent = self._inner.send(action)
        self._pose.update(sent)
        self.sends += 1
        self.min_z = min(self.min_z, kinematics.tool_position(self._pose)[2])
        return sent

    def disconnect(self) -> None:
        self._inner.disconnect()

    def disable_torque(self) -> None:
        self._inner.disable_torque()

    def torque_disabled(self):
        return self._inner.torque_disabled()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="explain the plan, move nothing")
    parser.add_argument("--object", default=DEFAULT_OBJECT, help="what to hover over")
    parser.add_argument("--duration", type=float, default=grasp.HOVER_DURATION_S)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    banner("Smoke 10: hover over a named object (MOTION)")

    hover_z = config.hover_z()
    print(f"table height : {config.TABLE_HEIGHT_M:+.3f} m (robot frame)")
    print(f"hover height : {config.HOVER_HEIGHT_M:.3f} m above it  ->  z = {hover_z:.3f} m")
    print(f"object       : {args.object!r}")

    if args.dry_run:
        print("\n[dry-run] would, in order:")
        print("[dry-run]   1. connect to the follower")
        print("[dry-run]   2. capture a frame and ask Gemini to point at the object")
        print("[dry-run]   3. map that pixel to robot XY through the homography")
        print("[dry-run]   4. check the point is inside the calibrated table polygon")
        print(f"[dry-run]   5. solve IK for a hover at z={hover_z:.3f} m and clamp it to 'policy'")
        print(f"[dry-run]   6. move there over {args.duration:.0f}s, hold {HOLD_SECONDS:.0f}s, return to start")
        print("[dry-run]   7. assert no commanded pose ever went below the hover plane")
        print("[dry-run] the gripper is never actuated in stage 4.")
        return ok("dry run complete")

    # --- preconditions, checked before anything is energised ---
    if not kinematics.available():
        return skip(
            "inverse kinematics is unavailable (placo or the URDF is missing). "
            "See docs/env_report.md, 'IK ladder'."
        )
    homography = calibrate.load()
    if homography is None:
        return skip(
            "not calibrated — no armani/data/homography.json. "
            "Run: python scripts/calibrate_camera.py"
        )
    if not config.TABLE_POLYGON:
        return skip("the saved calibration has no table polygon; re-run scripts/calibrate_camera.py")
    if not config.HOME_VERIFIED:
        return skip(
            "home pose is not verified — run scripts/capture_home.py first (safety rule 4)"
        )
    if config.api_key("GOOGLE_API_KEY") is None:
        return skip("GOOGLE_API_KEY is not set")

    print(f"calibration  : {calibrate.describe(homography)}")

    # --- vision, before the motors are touched ---
    try:
        frame = eyes.capture_frame()
    except eyes.EyesError as exc:
        return skip(f"no camera frame: {exc}")
    if (frame.shape[1], frame.shape[0]) != homography.frame_size:
        return fail(
            f"frame is {frame.shape[1]}x{frame.shape[0]} but the homography was calibrated at "
            f"{homography.frame_size[0]}x{homography.frame_size[1]}. The map does not apply."
        )

    try:
        detection = eyes.locate(args.object, frame=frame)
    except eyes.EyesError as exc:
        return fail(f"vision call failed: {exc}")
    if detection is None:
        return skip(f"Gemini did not see {args.object!r}. Put it on the table, in view, and retry.")

    try:
        x, y = homography.to_robot(detection.point)
    except calibrate.CalibrationError as exc:
        return fail(f"could not map pixel {detection.point}: {exc}")

    print("\n--- what it sees ---")
    print(f"  pixel      : {detection.point}")
    print(f"  robot XY   : ({x:+.4f}, {y:+.4f}) m")
    print(f"  confidence : {detection.confidence:.2f}")
    if not calibrate.point_in_polygon(x, y):
        return skip(
            f"({x:+.3f}, {y:+.3f}) m is outside the calibrated table polygon. "
            "Move the object further onto the calibrated area."
        )

    # Write the annotated frame now, so the operator can check the detection
    # before deciding whether to let the arm move to it.
    config.TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    import cv2

    detect_path = config.TEST_OUT_DIR / "hover_target.jpg"
    cv2.imwrite(str(detect_path), eyes.annotate(frame, [detection]))
    print(f"  annotated  : {detect_path}  (check the marker is on the object BEFORE approving)")

    if not safety.require_operator(f"move the arm to hover over {args.object!r}"):
        return skip("operator did not confirm presence")

    return _run_hover(args, x, y, detection)


def _run_hover(args, x: float, y: float, detection) -> int:
    safety.install_kill_switch()
    try:
        arm = motion.connect()
    except Exception as exc:
        return fail(f"could not connect: {type(exc).__name__}: {exc}")

    try:
        start_pose = arm.read_positions()
        start_xyz = kinematics.tool_position(start_pose)
        print(f"\nstart pose  : gripper at ({start_xyz[0]:+.3f}, {start_xyz[1]:+.3f}, {start_xyz[2]:+.3f}) m")

        watched = ZWatchArm(arm, start_pose)
        plan = grasp.plan_hover(x, y, start_pose)
        print("\n--- IK ---")
        if plan.joints:
            for joint in config.IK_JOINTS:
                print(f"  {joint:<15} {plan.joints[joint]:+8.2f} deg")
            print(f"  position error   {plan.position_error_m * 1000:.1f} mm")
            print(f"  approach lean    {plan.tilt_deg:.1f} deg off vertical")
        if not plan.ok:
            return skip(f"not reachable, so nothing moved: {plan.reason}")

        result = grasp.hover_over(watched, x, y, duration=args.duration)
        if not result.ok:
            return fail(f"hover failed after planning succeeded: {result.reason}")

        print(f"\nHOLDING for {HOLD_SECONDS:.0f}s — look at the arm.")
        time.sleep(HOLD_SECONDS)

        reached = kinematics.tool_position(arm.read_positions())
        print(f"reached     : ({reached[0]:+.3f}, {reached[1]:+.3f}, {reached[2]:+.3f}) m")
        print(f"asked for   : ({x:+.3f}, {y:+.3f}, {config.hover_z():+.3f}) m")

        verdict, miss_cm = _ask_operator()

        print("\nReturning to the starting pose...")
        # Through `watched`, not `arm`: the descent guard below has to cover the
        # WHOLE session. Checking only the outbound move would leave the return
        # path unwatched, which is exactly where a stage-5 mistake would land.
        # The gripper is deliberately excluded — stage 4 never actuates it, so it
        # has not moved and commanding it would be the one thing this stage bans.
        motion.goto(
            watched,
            {joint: start_pose[joint] for joint in config.IK_JOINTS},
            duration=config.RECOVERY_DURATION_S,
            profile="recorded",
        )

        # --- the no-descent guard, over every action commanded this session ---
        floor = min(start_xyz[2], config.hover_z()) - Z_PATH_TOLERANCE_M
        print(f"\nlowest commanded gripper height: {watched.min_z:+.4f} m over {watched.sends} sends")
        print(f"stage-4 floor                  : {floor:+.4f} m")
        descended = watched.min_z < floor

        log_event(
            "smoke_10",
            object=args.object,
            pixel=list(detection.point),
            robot_xy=[round(x, 4), round(y, 4)],
            reached_xyz=[round(v, 4) for v in reached],
            position_error_mm=round(result.position_error_m * 1000, 1),
            tilt_deg=round(result.tilt_deg, 1),
            vision_confidence=round(detection.confidence, 3),
            combined_confidence=grasp.combined_confidence(detection.confidence, result),
            min_commanded_z=round(watched.min_z, 4),
            descended=descended,
            operator_verdict=verdict,
            operator_miss_cm=miss_cm,
        )

        if descended:
            return fail(
                f"DESCENT DETECTED: something commanded the gripper down to {watched.min_z:.4f} m, "
                f"below the stage-4 floor of {floor:.4f} m. Stage 4 must never descend."
            )
        if verdict != "yes":
            return fail(
                "the operator did not confirm the gripper was over the object. "
                "That is the real measurement — record the miss in cm and check the homography."
            )
        return ok(
            f"hovered over {args.object!r} at ({x:+.3f}, {y:+.3f}) m, "
            f"{result.tilt_deg:.0f} deg lean, and returned to start"
        )
    except safety.OutsideEnvelopeError as exc:
        return fail(str(exc))
    except KeyboardInterrupt:
        print("\ninterrupted — the arm holds where it is.")
        return EXIT_SKIP
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})")


def _ask_operator() -> tuple[str, str]:
    """The only measurement that matters: is the gripper actually over it?

    Returns (verdict, miss_cm). The miss distance is the number stage 5's grasp
    reliability rides on, so it is asked for either way and written to the
    decision log — not just printed and forgotten.
    """
    print("\n" + "=" * 68)
    print("  OPERATOR: is the gripper visibly over the object?")
    print("  Estimate the miss in centimetres — stage 5's grasp reliability rides on it.")
    print("=" * 68)
    try:
        answer = input("  Over the object? [y/N] ").strip().lower()
        miss = input("  Roughly how far off, in cm? [0] ").strip() or "0"
    except (EOFError, KeyboardInterrupt):
        print()
        return "unknown", "unknown"

    verdict = "yes" if answer in ("y", "yes") else "no"
    print(f"  recorded: {'over the object' if verdict == 'yes' else 'NOT over the object'}, ~{miss} cm off")
    return verdict, miss


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(EXIT_SKIP) from None
