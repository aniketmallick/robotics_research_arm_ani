#!/usr/bin/env python
"""Ask Claude for a move, validate it, print it — and optionally perform it.

    python scripts/improvise_cli.py "do a slow clap" --dry-run   # no hardware
    python scripts/improvise_cli.py "take a proud bow"           # OPERATOR REQUIRED
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

from armani import config, improvise, motion, safety  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Claude-choreographed arm move.")
    parser.add_argument("description", help='what the move should be, e.g. "a slow clap"')
    parser.add_argument("--dry-run", action="store_true", help="plan and print only; no hardware")
    args = parser.parse_args()

    print(f'=== Improvise: "{args.description}" ===')
    print(f"model: {config.ANTHROPIC_MODEL}\n")

    try:
        keyframes = improvise.request_plan(args.description)
    except improvise.ImproviseError as exc:
        print(f"No usable plan: {exc}", file=sys.stderr)
        return 1

    print("Validated and clamped plan (policy profile):")
    print(improvise.describe_plan(keyframes))

    if args.dry_run:
        print("\n[dry-run] simulating the moves; no serial port is opened.")
        arm = motion.connect(dry_run=True)
        improvise.perform(arm, keyframes)
        arm.disconnect()
        return 0

    if not safety.require_operator(f'perform an improvised move: "{args.description}"'):
        print("Not confirmed — nothing moved.")
        return 1

    try:
        arm = motion.connect()
    except Exception as exc:
        print(f"could not connect: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        safety.clear_stop()
        safety.install_kill_switch()
        print("Kill switch armed: Ctrl-C freezes the arm and asks what to do.")
        improvise.perform(arm, keyframes)
    except safety.OutsideEnvelopeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"move failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
