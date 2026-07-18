#!/usr/bin/env python
"""Smoke 01 — serial port + follower connection. Commands no trajectory.

Connects to the follower using the existing calibration and prints live joint
positions for a few seconds. No target is ever commanded.

It still asks the operator to confirm presence, because connecting is not
passive: lerobot's connect() calls configure(), which runs inside
bus.torque_disabled() — a context manager that RE-ENABLES torque on exit. The
arm therefore becomes energised and stiff the moment it connects, and if a
stale Goal_Position is left in the servos it can twitch. Safety rule 1 applies.
"""

from __future__ import annotations

import sys
import time

from _bootstrap import banner, fail, ok, parse_args, skip

from armani import config, motion, safety
from armani.logutil import log_event

READ_SECONDS = 3.0
READ_HZ = 5


def main() -> int:
    args = parse_args(__doc__ or "")
    banner("Smoke 01: serial ports and follower connection")

    ports = motion.find_serial_ports()
    print(f"Serial ports matching {config.SERIAL_PORT_GLOB}:")
    for port in ports:
        print(f"  {port}")
    if not ports:
        print("  (none)")

    if args.dry_run:
        print("\n[dry-run] would connect to the follower and stream joint positions.")
        print(f"[dry-run] expected calibration id: {config.FOLLOWER_ID}")
        return ok("dry run complete")

    if not ports:
        return fail(
            "No serial ports found. Plug in the SO-101 follower over USB and make sure "
            "the cable carries data (charge-only cables enumerate nothing)."
        )

    if not safety.require_operator("connect to the follower, which energises the motors"):
        return skip("operator did not confirm presence")

    try:
        arm = motion.connect()
    except Exception as exc:
        return fail(f"could not connect: {type(exc).__name__}: {exc}")

    try:
        print(f"\nConnected: {arm.label}")
        print("The motors are now energised and holding position.")
        print(f"Streaming positions for {READ_SECONDS:.0f}s (no target commanded)...\n")
        joints = list(config.JOINTS)
        print("  " + "".join(f"{j:>15}" for j in joints))
        deadline = time.perf_counter() + READ_SECONDS
        last: dict[str, float] = {}
        while time.perf_counter() < deadline:
            last = arm.read_positions()
            print("  " + "".join(f"{last.get(j, float('nan')):>15.2f}" for j in joints))
            time.sleep(1.0 / READ_HZ)

        missing = [j for j in joints if j not in last]
        if missing:
            return fail(f"follower did not report joint(s): {missing}")
        log_event("smoke_01", positions={k: round(v, 2) for k, v in last.items()})

        # Warn here rather than letting smoke_02 discover it: interp_move
        # refuses to move an arm that is resting outside its policy limits.
        outside = [
            f"{j}={last[j]:.1f} (limit {config.JOINT_LIMITS[j][0]:g}..{config.JOINT_LIMITS[j][1]:g})"
            for j in joints
            if not config.JOINT_LIMITS[j][0] <= last[j] <= config.JOINT_LIMITS[j][1]
        ]
        if outside:
            print(
                "\nWARNING: these joints rest OUTSIDE config.JOINT_LIMITS:\n  "
                + "\n  ".join(outside)
                + "\n  smoke_02 will refuse to move them. Move the arm back by hand, or ask\n"
                  "  the architect to widen the limit deliberately."
            )
            return skip(f"{len(outside)} joint(s) outside the policy envelope")

        return ok(f"follower responded on all {len(joints)} joints, all inside limits")
    except Exception as exc:
        return fail(f"error while reading positions: {type(exc).__name__}: {exc}")
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
