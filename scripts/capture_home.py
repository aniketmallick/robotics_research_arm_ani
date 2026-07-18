#!/usr/bin/env python
"""Record the home pose by physically posing the arm. OPERATOR REQUIRED.

Torque is released so the arm can be moved by hand, live joint positions stream
at ~2 Hz, and ENTER captures wherever it is. Torque comes back on automatically.

Until this has been run, safety rule 4 forbids the code from ever auto-driving
to HOME_POSE — the kill switch will not offer "home" and motion.home() refuses.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import _bootstrap  # noqa: E402,F401  (verifies the interpreter, exits if wrong)

from armani import config, motion, safety  # noqa: E402
from armani.logutil import log_event  # noqa: E402

STREAM_HZ = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a verified home pose.")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, touch no hardware")
    args = parser.parse_args()

    print("=== Capture home pose ===")
    if config.HOME_VERIFIED:
        print(f"A verified home already exists at {config.HOME_POSE_PATH}:")
        print("  " + ", ".join(f"{j}={v:.1f}" for j, v in sorted(config.HOME_POSE.items())))
        if not args.dry_run and input("Replace it? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Keeping the existing home pose.")
            return 0

    if args.dry_run:
        print(f"[dry-run] would release torque, stream positions at {STREAM_HZ:.0f} Hz,")
        print("[dry-run] capture on ENTER, re-enable torque, and write:")
        print(f"[dry-run]   {config.HOME_POSE_PATH}")
        print("[dry-run] no file is written in dry-run — that would falsely mark home verified.")
        return 0

    if not safety.require_operator("release torque so you can pose the arm by hand"):
        print("Not confirmed — nothing done.")
        return 1

    print(
        "\nPose the arm somewhere it can safely rest:\n"
        "  * upright-ish, not folded against itself\n"
        "  * gripper roughly mid-open\n"
        "  * nothing overhanging the table edge\n"
        "  * a pose every gesture can start and end at\n"
    )

    try:
        arm = motion.connect()
    except Exception as exc:
        print(f"could not connect: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        pose = _pose_by_hand(arm)
    except Exception as exc:
        print(f"capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)

    return _save(pose)


def _pose_by_hand(arm: motion.Arm) -> dict[str, float]:
    """Stream positions with torque off until the operator presses ENTER."""
    captured = threading.Event()

    def wait_for_enter() -> None:
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        captured.set()

    joints = list(config.JOINTS)
    print("SUPPORT THE ARM — torque is about to release and it will go limp.")
    input("Press ENTER when you are holding it... ")

    # torque_disabled() guarantees torque comes back on however this block exits.
    with arm.torque_disabled():
        print("\nTorque OFF. Move the arm into position, then press ENTER to capture.\n")
        print("  " + "".join(f"{j:>15}" for j in joints))
        threading.Thread(target=wait_for_enter, daemon=True).start()
        pose: dict[str, float] = {}
        while not captured.is_set():
            pose = arm.read_positions()
            print("  " + "".join(f"{pose.get(j, float('nan')):>15.2f}" for j in joints), end="\r")
            captured.wait(1.0 / STREAM_HZ)
        # Re-read after ENTER: the operator may have nudged it while reaching.
        pose = arm.read_positions()

        # CRITICAL: park Goal_Position at where the arm actually is before torque
        # returns. lerobot's enable_torque writes only Torque_Enable and Lock —
        # it never touches Goal_Position — so each servo would otherwise drive
        # back to whatever goal was set BEFORE the operator posed it by hand,
        # snapping the arm out of their grip the instant this block exits.
        arm.send(pose)
        print("\n\nCaptured, and the arm is set to hold this pose.")
        print("Torque coming back on — keep supporting it.")

    return pose


def _save(pose: dict[str, float]) -> int:
    missing = [j for j in config.JOINTS if j not in pose]
    if missing:
        print(f"arm did not report joint(s) {missing}; not saving", file=sys.stderr)
        return 1

    print("\nCaptured pose:")
    for joint in config.JOINTS:
        print(f"  {joint:<15} {pose[joint]:+8.2f}")

    try:
        safety.check_start_pose(pose, config.JOINTS)
    except safety.OutsideEnvelopeError as exc:
        print(f"\nREFUSING TO SAVE: {exc}", file=sys.stderr)
        return 1

    outside = [
        f"{j}={pose[j]:.1f} (policy {config.JOINT_LIMITS[j][0]:g}..{config.JOINT_LIMITS[j][1]:g})"
        for j in config.JOINTS
        if not config.JOINT_LIMITS[j][0] <= pose[j] <= config.JOINT_LIMITS[j][1]
    ]
    if outside:
        # Not fatal — home() clamps with the "recorded" profile — but the
        # operator should know the pose sits outside the conservative envelope.
        print("\nWARNING: this pose is outside the policy envelope:\n  " + "\n  ".join(outside))
        print("  home() uses the 'recorded' profile so it still works, but a home pose")
        print("  inside the policy envelope is preferable. Re-pose and re-run to change it.")
        if input("Save anyway? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Not saved.")
            return 1

    payload = {
        "pose": {j: round(float(pose[j]), 3) for j in config.JOINTS},
        "verified": True,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Captured by hand via scripts/capture_home.py with torque released.",
    }
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.HOME_POSE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    log_event("home_captured", pose=payload["pose"])

    print(f"\nSaved {config.HOME_POSE_PATH}")
    print("Home is now VERIFIED: motion.home() works and the kill switch offers [h].")
    print("Record every gesture starting and ending at this pose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
