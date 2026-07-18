#!/usr/bin/env python
"""Smoke 02 — the smallest possible real motion. OPERATOR MUST BE PRESENT.

Moves ONE joint by +/-5 degrees relative to where it already is, then returns
it to exactly where it started.

Two deliberate choices, both about not trusting unverified constants:

* The wiggle is RELATIVE to the current pose. Commanding an absolute target
  would move the arm from wherever it is to that target, which is not a 5
  degree move.
* The return leg goes to the STARTING pose, which is known safe because the arm
  was just there. Nothing here drives to config.HOME_POSE — safety rule 4 bars
  that until capture_home has verified it on hardware.

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
# Wiggle targets come from the measured pose, so they use the recorded profile.
WIGGLE_PROFILE = "recorded"


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
        home_state = "available" if config.HOME_VERIFIED else "NOT offered (home is unverified)"
        print(
            "\nKill switch armed: Ctrl-C (ESC too, if Input Monitoring is granted).\n"
            "  Safety rule 7: the first press FREEZES the arm — it stops commanding and holds\n"
            "  position. Nothing moves until you choose:\n"
            "    [s] return slowly to where this move started\n"
            f"    [h] go to home — {home_state}\n"
            "    [t] torque off (support the arm first, it will drop)\n"
            "    [l] leave it exactly as-is\n"
            "  A second Ctrl-C aborts hard, leaving the arm where it is."
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
        return skip("kill switch fired during the test; the arm was frozen and you chose what next")

    print(
        "\nNOTE: this test returns to the pose the arm started in. Nothing here ever drives to\n"
        "      config.HOME_POSE — the kill switch only offers it, and only once capture_home\n"
        "      has verified it (safety rules 4 and 7)."
    )
    return ok(f"{args.joint} moved +/-{WIGGLE_DEGREES:.0f} degrees and returned to start")


def _wiggle(arm: motion.Arm, joint: str, start: dict[str, float]) -> None:
    """Out and back, both legs clamped and interpolated."""
    if joint not in start:
        raise RuntimeError(f"arm did not report a position for {joint!r}")

    # Both legs are derived from the MEASURED pose, not from an LLM or IK, so
    # they are clamped against the "recorded" profile — the policy envelope
    # would clip a legitimately parked joint by tens of degrees.
    limit_low, limit_high = config.LIMIT_PROFILES[WIGGLE_PROFILE][joint]

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
    motion.goto(arm, out_target, duration=LEG_SECONDS, profile=WIGGLE_PROFILE)

    print(f"  -> {joint} back to {start[joint]:+.1f} over {LEG_SECONDS:.0f}s")
    motion.goto(arm, {joint: start[joint]}, duration=LEG_SECONDS, profile=WIGGLE_PROFILE)

    log_event("smoke_02", joint=joint, degrees=WIGGLE_DEGREES, returned_to=round(start[joint], 2))


if __name__ == "__main__":
    raise SystemExit(main())
