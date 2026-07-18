#!/usr/bin/env python
"""Smoke 07 — recorded gesture macros.

Dry run: load every configured episode from the local dataset, report frame
counts, timing, the poses it starts and ends at, and how much the `recorded`
clamp would alter it. No hardware, no network.

Live: the operator watches one gesture replay (default `bow`).
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import banner, fail, ok, skip

from armani import config, gestures, motion, safety

DEFAULT_LIVE_GESTURE = "bow"


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and optionally replay gesture macros.")
    parser.add_argument("--dry-run", action="store_true", help="load and inspect only, no hardware")
    parser.add_argument(
        "--gesture",
        default=DEFAULT_LIVE_GESTURE,
        choices=gestures.list_gestures(),
        help=f"which gesture to replay live (default: {DEFAULT_LIVE_GESTURE})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse()
    banner("Smoke 07: gesture macros")

    print(f"Configured gestures ({len(config.GESTURES)}): {', '.join(gestures.list_gestures())}")
    print(f"Dataset: {config.GESTURE_DATASET_REPO_ID}")
    print(f"Root   : {config.GESTURE_DATASET_ROOT}")

    if not gestures.dataset_available():
        return skip(
            "gesture dataset not recorded yet. The operator records it once — "
            "see docs/recording_gestures.md. Everything else in stage 2 is independent of it."
        )

    loaded, problems = _inspect_all()
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return fail(f"{len(problems)} of {len(config.GESTURES)} gestures could not be loaded")

    if args.dry_run:
        return ok(f"all {len(loaded)} gestures loaded and are playable")

    # --- live replay ---
    if not config.HOME_VERIFIED:
        print(
            "\nNote: home is not verified, so the arm will be left at the gesture's last\n"
            "      frame instead of returning home. Run scripts/capture_home.py first."
        )

    if not safety.require_operator(f"replay the {args.gesture!r} gesture"):
        return skip("operator did not confirm presence")

    try:
        arm = motion.connect()
    except Exception as exc:
        return fail(f"could not connect: {type(exc).__name__}: {exc}")

    try:
        safety.clear_stop()
        safety.install_kill_switch()
        print("Kill switch armed: Ctrl-C freezes the arm and asks what to do next.")
        gestures.play_gesture(arm, args.gesture)
    except safety.OutsideEnvelopeError as exc:
        return skip(str(exc))
    except Exception as exc:
        return fail(f"replay failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)

    if safety.stop_requested():
        return skip("kill switch fired during the replay")
    return ok(f"replayed {args.gesture!r}")


def _inspect_all() -> tuple[list[gestures.Gesture], list[str]]:
    """Load every configured gesture, reporting rather than raising."""
    loaded: list[gestures.Gesture] = []
    problems: list[str] = []

    print(f"\n{'gesture':<14}{'ep':>3}{'frames':>8}{'sec':>7}  {'max clamp':>10}  start -> end (shoulder_lift)")
    for name in gestures.list_gestures():
        try:
            gesture = gestures.load_gesture(name)
        except Exception as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"{name:<14}{config.GESTURES[name]:>3}{'  FAILED':>8}")
            continue

        deviation = gestures.frame_clamp_deviation(gesture)
        worst = max(deviation.values(), default=0.0)
        loaded.append(gesture)
        print(
            f"{name:<14}{gesture.episode:>3}{len(gesture.frames):>8}{gesture.seconds:>7.1f}"
            f"  {worst:>10.2f}  {gesture.first.get('shoulder_lift', float('nan')):+.1f}"
            f" -> {gesture.last.get('shoulder_lift', float('nan')):+.1f}"
        )

        # The recorded profile is the physical range minus a 2 degree standoff,
        # so a take that pressed against a stop is trimmed slightly. That is the
        # margin doing its job; a large deviation means the wrong profile or a
        # dataset from a different robot.
        if worst > config.RECORDED_MARGIN + config.PHYSICAL_TOLERANCE:
            problems.append(
                f"{name}: 'recorded' clamp would alter frames by up to {worst:.2f} deg "
                f"(joint {max(deviation, key=deviation.get)}) — more than the "
                f"{config.RECORDED_MARGIN:g} deg standoff allows"
            )

    if loaded:
        _report_chainability(loaded)
    return loaded, problems


def _report_chainability(loaded: list[gestures.Gesture]) -> None:
    """Every gesture should start and end at the same pose, so they chain.

    Advisory only — it is a property of how the operator recorded, not something
    this code can fix, and stage 3 chains gestures back to back.
    """
    spread = 0.0
    for gesture in loaded:
        for joint, value in gesture.last.items():
            spread = max(spread, abs(value - loaded[0].first.get(joint, value)))
    verdict = "consistent" if spread <= 10.0 else "INCONSISTENT — replays may not chain cleanly"
    print(f"\nStart/end pose spread across all gestures: {spread:.1f} deg ({verdict})")


if __name__ == "__main__":
    raise SystemExit(main())
