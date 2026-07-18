#!/usr/bin/env python
"""Run every ARM-ANI smoke test in order and summarise what works.

Each test runs as its own subprocess so a crash or a hung driver in one cannot
take the others down, and so each test's exit code is read cleanly:
    0 PASS   1 FAIL   2 SKIP

The motion test (02) is never run without the operator confirming presence —
that prompt lives inside the test itself, and stdin is inherited so it works.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"

PASS, FAIL, SKIP = 0, 1, 2
STATUS = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}


@dataclass(frozen=True)
class Check:
    script: str
    title: str
    needs_hardware: bool
    note: str


CHECKS: tuple[Check, ...] = (
    Check("smoke_01_ports.py", "Serial ports + follower connection", True, "arm plugged in, USB data cable"),
    Check("smoke_02_wiggle.py", "5 degree wiggle (MOTION)", True, "OPERATOR MUST WATCH THE ARM"),
    Check("smoke_03_camera.py", "Camera frame @640x480", True, "C920 on its tripod"),
    Check("smoke_04_mic.py", "Microphone record + playback", True, "wired headset selected as input"),
    Check("smoke_05_keys.py", "OpenAI / Gemini / Anthropic keys", False, "network + .env filled in"),
    Check("smoke_06_ptt.py", "Global spacebar (push-to-talk)", False, "Input Monitoring granted"),
)


def run(check: Check, dry_run: bool) -> int:
    print("\n" + "=" * 72)
    print(f"  {check.script}  —  {check.title}")
    print(f"  needs: {check.note}")
    print("=" * 72)

    command = [sys.executable, str(TESTS_DIR / check.script)]
    if dry_run:
        command.append("--dry-run")
    # The child writes straight to fd 1. Without this flush our buffered header
    # lands after the child's output whenever doctor's output is piped to a file.
    sys.stdout.flush()

    try:
        # stdin is inherited on purpose: the tests prompt the operator.
        process = subprocess.Popen(command, cwd=str(TESTS_DIR))
    except OSError as exc:
        print(f"  could not run {check.script}: {exc}")
        return FAIL

    # Deliberately NOT subprocess.run(): on KeyboardInterrupt it calls
    # process.kill(), which SIGKILLs the child. Ctrl-C already reached the child
    # via the terminal's process group, and the child's kill switch responds by
    # slow-homing the arm. SIGKILLing it mid-trajectory would abandon the arm
    # torqued and half way through a move. So we wait for it to finish instead.
    interrupts = 0
    while True:
        try:
            code = process.wait()
            break
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts == 1:
                print("\n  Ctrl-C — letting the test stop the arm safely. Ctrl-C again to give up on it.")
                continue
            print("\n  second Ctrl-C — abandoning the child process without waiting.")
            raise

    return code if code in STATUS else FAIL


def main() -> int:
    parser = argparse.ArgumentParser(description="ARM-ANI dependency doctor.")
    parser.add_argument("--dry-run", action="store_true", help="run every test in dry-run mode")
    parser.add_argument("--skip-motion", action="store_true", help="skip smoke 02 (the only test that moves)")
    parser.add_argument("--only", metavar="N", help="run one check by number, e.g. --only 3")
    args = parser.parse_args()

    from armani import config

    print("ARM-ANI doctor")
    print(f"python : {sys.executable}")
    print(f"repo   : {REPO_ROOT}")
    if args.dry_run:
        print("mode   : DRY RUN — no hardware or network is touched")
    elif config.DRY_RUN:
        # Otherwise a stale ARMANI_DRY_RUN=1 makes every hardware test quietly
        # run against the simulated arm and still report PASS, and the summary
        # would claim the foundation is proven having touched nothing.
        print(
            "\nERROR: ARMANI_DRY_RUN is set in the environment or .env, but --dry-run was not\n"
            "       passed. A live run would silently use the simulated arm and report PASS\n"
            "       without touching hardware. Unset ARMANI_DRY_RUN, or run with --dry-run.",
            file=sys.stderr,
        )
        return 1
    else:
        print("mode   : LIVE — smoke 02 will move the arm after you confirm")

    checks = list(CHECKS)
    if args.only:
        if not args.only.isdigit() or not 1 <= int(args.only) <= len(CHECKS):
            print(f"--only must be 1..{len(CHECKS)}", file=sys.stderr)
            return 2
        checks = [CHECKS[int(args.only) - 1]]

    results: list[tuple[Check, int]] = []
    for check in checks:
        if args.skip_motion and check.script.startswith("smoke_02"):
            print(f"\n  skipping {check.script} (--skip-motion)")
            results.append((check, SKIP))
            continue
        try:
            results.append((check, run(check, args.dry_run)))
        except KeyboardInterrupt:
            # The operator gave up on this check; abandon the whole run rather
            # than marching on into the next one (which may move the arm).
            print("\n  aborting the remaining checks.")
            results.append((check, SKIP))
            break

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    width = max(len(c.script) for c, _ in results)
    for check, code in results:
        print(f"  {STATUS[code]:<5} {check.script:<{width}}  {check.title}")

    failed = [c.script for c, code in results if code == FAIL]
    skipped = [c.script for c, code in results if code == SKIP]
    print()
    if failed:
        print(f"  {len(failed)} FAILED: {', '.join(failed)}")
        print("  Fix these before stage 2 — read each test's output above for the exact reason.")
    if skipped:
        print(f"  {len(skipped)} skipped: {', '.join(skipped)}")
    if args.dry_run:
        # Never let a dry run read as proof: it exercised the code paths and
        # touched no hardware, no microphone and no API.
        print("  DRY RUN only — this proves the code runs, NOT that the hardware works.")
        print("  Re-run without --dry-run, with the operator present, to prove the foundation.")
    elif not failed and not skipped:
        print("  All checks passed. Foundation is proven.")
    elif not failed:
        print("  No failures. Review the skips above and decide if each is acceptable.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
