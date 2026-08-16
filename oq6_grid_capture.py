#!/usr/bin/env python3
"""OQ-6 — capture the placement grid in robot-frame coordinates. READ-ONLY: this
script never commands motion. Torque is released; YOU move the arm by hand.

Run from the Interactive Robot repo root, inside the lerobot conda env:

    conda activate lerobot
    python oq6_grid_capture.py                # real arm
    python oq6_grid_capture.py --dry-run      # rehearse the flow, no hardware

Procedure per point: close the gripper jaws on nothing (so the tip is defined),
hand-guide the TIP of the closed gripper to touch the marked point, hold it
still, press Enter. The script reads joints and computes FK via armani's
verified kinematics (placo + SO-101 URDF). Repeat for all points; 'r' redoes
the last point, 'q' finishes early. Output: grid_points.csv.

Safety: uses armani.motion.connect() (existing calibration only, never
recreates) and holds torque OFF for the whole capture via torque_disabled().
No targets are ever sent. Operator present throughout (your rule 1)."""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

DEFAULT_OUT = Path(
    "/Users/Aniket.Mallick/Documents/Claude/Projects/Robo_Research_Data/ego2so101/grid_points.csv"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=int, default=12, help="number of grid points to capture")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from armani import kinematics, motion  # inside main: import errors print cleanly

    if not args.dry_run and not kinematics.available():
        print("FATAL: kinematics unavailable (placo/URDF). Fix per docs/env_report.md; nothing captured.")
        return 1

    arm = motion.connect(dry_run=args.dry_run)
    rows: list[dict] = []
    try:
        print(f"\nConnected: {arm.label}")
        print("Releasing torque — the arm will go LIMP. Support it before continuing.")
        input("Press Enter when you are holding the arm... ")
        with arm.torque_disabled():
            i = 1
            while i <= args.points:
                cmd = input(
                    f"\n[point {i}/{args.points}] close jaws, touch tip to mark #{i}, hold still, "
                    "then Enter (r=redo last, q=quit): "
                ).strip().lower()
                if cmd == "q":
                    break
                if cmd == "r" and rows:
                    rows.pop()
                    i -= 1
                    print("  last point discarded — redo it.")
                    continue
                pose = arm.read_positions()  # degrees (+ gripper 0-100)
                if args.dry_run:
                    x = y = z = 0.0
                else:
                    x, y, z = kinematics.tool_position(pose)
                row = {"point_id": i, "x_m": round(x, 4), "y_m": round(y, 4), "z_m": round(z, 4)}
                row.update({f"{j}_deg": round(float(v), 2) for j, v in pose.items()})
                rows.append(row)
                print(f"  captured P{i}: x={x:+.3f} y={y:+.3f} z={z:+.3f} m")
                i += 1
        print("\nTorque restored (context exit). Take the arm's weight is back on the motors.")
    finally:
        arm.disconnect()

    if not rows:
        print("Nothing captured; no file written.")
        return 1

    # Sanity: warn on duplicate/implausible points before writing.
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            d = ((rows[a]["x_m"] - rows[b]["x_m"]) ** 2 + (rows[a]["y_m"] - rows[b]["y_m"]) ** 2) ** 0.5
            if d < 0.02 and not args.dry_run:
                print(f"  WARNING: P{rows[a]['point_id']} and P{rows[b]['point_id']} are {d*100:.1f} cm apart — same mark touched twice?")
    zs = [r["z_m"] for r in rows]
    if not args.dry_run and (max(zs) - min(zs)) > 0.02:
        print(f"  WARNING: z spread {max(zs)-min(zs):.3f} m across a flat table — check tip contact consistency.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\nWrote {len(rows)} points -> {args.out}  ({stamp})")
    print("Next: photograph the marked table from the C920, then OQ-7 mini-capture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
