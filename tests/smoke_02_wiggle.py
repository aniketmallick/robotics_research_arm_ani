#!/usr/bin/env python
"""Smoke 02 — the smallest possible real motion. OPERATOR MUST BE PRESENT.

Moves ONE joint by +/-5 degrees relative to where it already is, then returns
it to exactly where it started.

Two deliberate choices, both about not trusting unverified constants:

* The wiggle is RELATIVE to the current pose. Commanding an absolute target
  would move the arm from wherever it is to that target, which is not a 5
  degree move.
* "Slow home" at the end returns to the STARTING pose, which is known to be
  safe because the arm was already there. config.HOME_POSE is still a
  placeholder that has never been verified on hardware, so moving to it is
  offered separately, behind its own confirmation.

Default joint is wrist_roll: it rotates the gripper in place and cannot drive
the arm into the table.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import banner, fail, ok, skip

from armani import config, motion, safety
from armani.logutil import log_event

WIGGLE_DEGREES = 5.0
LEG_SECONDS = 2.0
DEFAULT_JOINT = "wrist_roll"


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One joint, +/-5 degrees, then back.")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, touch no hardware")
    parser.add_argument(
        "--joint",
        default=DEFAULT_JOINT,
        choices=[j for j in config.JOINTS if j != config.GRIPPER_JOINT],
        help=f"which joint to wiggle (default: {DEFAULT_JOINT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse()
    banner(f"Smoke 02: {WIGGLE_DEGREES:.0f} degree wiggle of {args.joint}")

    if args.dry_run:
        print(f"[dry-run] would ask the operator to confirm, then move {args.joint}")
        print(f"[dry-run]   +{WIGGLE_DEGREES}deg over {LEG_SECONDS}s")
        print(f"[dry-run]   -{WIGGLE_DEGREES}deg back to start over {LEG_SECONDS}s")
        print("[dry-run] all targets pass through safety.clamp_action() and are interpolated")
        arm = motion.connect(dry_run=True)
        start = arm.read_positions()
        _wiggle(arm, args.joint, start)
        arm.disconnect()
        return ok("dry run complete")

    if not safety.require_operator(f"wiggle {args.joint} by +/-{WIGGLE_DEGREES:.0f} degrees"):
        return skip("operator did not confirm presence")

    try:
        arm = motion.connect()
    except Exception as exc:
        return fail(f"could not connect: {type(exc).__name__}: {exc}")

    # Everything from here on is inside the try, so no failure can leave the arm
    # connected and torqued.
    try:
        safety.clear_stop()
        safety.install_kill_switch()
        home_pose = ", ".join(f"{k}={v:.0f}" for k, v in sorted(config.HOME_POSE.items()))
        print(
            "\nKill switch armed: Ctrl-C (ESC too, if Input Monitoring is granted).\n"
            "  Per safety rule 7 the kill switch slow-homes the arm. HOME_POSE is still the\n"
            f"  UNVERIFIED placeholder ({home_pose}), so if you trigger it the arm will move\n"
            "  there, which may be further than this test's 5 degrees. Keep that path clear.\n"
            "  A second Ctrl-C aborts immediately, leaving the arm where it is."
        )

        start = arm.read_positions()
        print(f"Start pose: {{{', '.join(f'{k}={v:.1f}' for k, v in sorted(start.items()))}}}")

        with safety.SafeMotion(arm, description=f"wiggle {args.joint}"):
            _wiggle(arm, args.joint, start)
    except safety.OutsideEnvelopeError as exc:
        return skip(f"{exc}")
    except Exception as exc:
        return fail(f"motion failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)

    if safety.stop_requested():
        return skip("kill switch fired during the test; arm returned to start")

    print(
        "\nNOTE: on the normal path this test returns to the pose the arm started in and never\n"
        "      drives to config.HOME_POSE, which is still an unverified placeholder. Only the\n"
        "      kill-switch path homes, because safety rule 7 requires it."
    )
    return ok(f"{args.joint} moved +/-{WIGGLE_DEGREES:.0f} degrees and returned to start")


def _wiggle(arm: motion.Arm, joint: str, start: dict[str, float]) -> None:
    """Out and back, both legs clamped and interpolated."""
    if joint not in start:
        raise RuntimeError(f"arm did not report a position for {joint!r}")

    limit_low, limit_high = config.JOINT_LIMITS[joint]

    def fits(value: float) -> bool:
        return limit_low <= value <= limit_high

    # Near a limit the clamp would silently shrink the move, so pick a direction
    # that actually fits. If NEITHER fits, the joint has less than 5 degrees of
    # room and we must not pretend the test ran.
    if fits(start[joint] + WIGGLE_DEGREES):
        target_value = start[joint] + WIGGLE_DEGREES
    elif fits(start[joint] - WIGGLE_DEGREES):
        target_value = start[joint] - WIGGLE_DEGREES
        print(f"  (near the {joint} upper limit, wiggling the other direction)")
    else:
        raise RuntimeError(
            f"{joint} is at {start[joint]:.1f} with limits {limit_low:g}..{limit_high:g}; "
            f"there is no room for a {WIGGLE_DEGREES:.0f} degree move in either direction. "
            "Move the arm nearer the middle of its range and re-run."
        )
    out_target = {joint: target_value}

    print(f"  -> {joint} to {out_target[joint]:+.1f} over {LEG_SECONDS:.0f}s")
    motion.goto(arm, out_target, duration=LEG_SECONDS)

    print(f"  -> {joint} back to {start[joint]:+.1f} over {LEG_SECONDS:.0f}s")
    motion.goto(arm, {joint: start[joint]}, duration=LEG_SECONDS)

    log_event("smoke_02", joint=joint, degrees=WIGGLE_DEGREES, returned_to=round(start[joint], 2))


if __name__ == "__main__":
    raise SystemExit(main())
