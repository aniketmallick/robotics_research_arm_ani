"""Coverage map — where the demos were, where the eval trials went, and what they scored.

The S3 verdict's first finding was "coverage predicts capability": the fine-tuned policy
failed where the 50 demonstrations were thinnest. This script turns that sentence into a
figure and a number, from the same artifacts the verdict cites — no hand-placed dots.

    python experiments/s3_finetune/coverage_map.py \
        --root <s3 dataset dir> --logs experiments/s2_zero_shot/logs \
        --trials experiments/s2_zero_shot/trials.csv --out docs/coverage_map.png

What it reads
  * dataset parquet: per-frame action for all 50 episodes -> the arm's pose at the moment
    each demo's gripper closed (the "grasp pose"). Two joints locate the block on the table
    well enough for this purpose: shoulder_pan (left/right) and shoulder_lift (near/far).
  * eval per-step JSONL (episode_ft_<tag>_*.jsonl): the pose the policy was at when it
    first commanded the gripper closed (its grasp attempt); if it never did, its last pose.
  * trials.csv: the 0-4 score per official tag (A2..A11, B1..B5, C1..C3).

What it writes
  * the PNG: demo grasp poses (grey x) on a 5x5 cell grid with per-cell counts, eval attempts
    overlaid and coloured by score.
  * stdout: per-trial nearest-demo distance (in joint-space degrees, pan/lift), and the
    median distance for successes (4) vs non-grasps (<=1). Copy that block into the docs.

Limitations (say them out loud): joint space is a proxy for table position; the grasp
pose is a proxy for the placement; the mapping is per-rig. It is the honest version of
the finding, not a calibrated workspace map.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

OFFICIAL_TAGS = (
    [f"ft_A{i}" for i in range(2, 12)] + [f"ft_B{i}" for i in range(1, 6)] + [f"ft_C{i}" for i in range(1, 4)]
)
GRIPPER_CLOSED_BELOW = 20.0  # gripper.pos in %; rests ~2, opens to 30-63, holds the 40 mm block at 1-15
PAN, LIFT, GRIP = 0, 1, 5  # indices in the 6-D action vector (see meta/info.json)


def demo_grasp_poses(root: Path) -> np.ndarray:
    import pandas as pd  # local import: keeps the module importable without pandas

    frames = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in frames], ignore_index=True)
    poses = []
    for _, ep in df.groupby("episode_index"):
        act = np.stack(ep["action"].values)
        grip = act[:, GRIP]
        opened = np.where(grip > 30.0)[0]
        if len(opened) == 0:
            continue
        # first closure after the gripper was open = the grasp
        after = np.where(grip[opened[0] :] < GRIPPER_CLOSED_BELOW)[0]
        idx = opened[0] + after[0] if len(after) else int(np.argmin(grip))
        poses.append(act[idx, [PAN, LIFT]])
    poses = np.array(poses)
    # A table grasp on this rig has shoulder_lift > 0. A "close" with lift < 0 is a transient
    # dip near the home pose, not a grasp; 5 of 50 episodes trip this and are dropped here.
    return poses[poses[:, 1] > 0]


def eval_attempt_pose(path: Path) -> tuple[np.ndarray, bool]:
    """(pan, lift) at the policy's first close-after-open command — its grasp attempt.

    The gripper rests closed (~2%) at the start pose, so "first close" alone would return
    step 0. Same rule as the demos: wait for the gripper to open (>30%), then take the
    first close (<20%). If it never opened-then-closed, return the pose the policy
    ended at, flagged False.
    """
    last = None
    opened = False
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            sent = rec.get("sent") or rec.get("clamped") or rec.get("raw")
            if not sent:
                continue
            last = sent
            if sent["gripper"] > 30.0:
                opened = True
            elif opened and sent["gripper"] < GRIPPER_CLOSED_BELOW:
                return np.array([sent["shoulder_pan"], sent["shoulder_lift"]]), True
    return np.array([last["shoulder_pan"], last["shoulder_lift"]]), False


def read_scores(trials_csv: Path) -> dict[str, int]:
    scores: dict[str, int] = {}
    with open(trials_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            tag = row["episode_tag"]
            if tag in OFFICIAL_TAGS and tag not in scores:  # first row per official tag
                scores[tag] = int(row["score"])
    return scores


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--logs", default="experiments/s2_zero_shot/logs")
    p.add_argument("--trials", default="experiments/s2_zero_shot/trials.csv")
    p.add_argument("--out", default="docs/coverage_map.png")
    a = p.parse_args(argv)

    demos = demo_grasp_poses(Path(a.root))
    scores = read_scores(Path(a.trials))
    rows = []
    for tag in OFFICIAL_TAGS:
        files = sorted(glob.glob(str(Path(a.logs) / f"episode_{tag}_*.jsonl")))
        if not files or tag not in scores:
            continue
        pose, closed = eval_attempt_pose(Path(files[-1]))  # last file = the scored run
        d = np.min(np.linalg.norm(demos - pose, axis=1))
        rows.append((tag, scores[tag], pose, closed, d))

    print(f"demos: {len(demos)} grasp poses  pan [{demos[:,0].min():.1f}, {demos[:,0].max():.1f}]"
          f"  lift [{demos[:,1].min():.1f}, {demos[:,1].max():.1f}]")
    print(f"{'trial':7s} {'score':5s} {'pan':>7s} {'lift':>7s} {'closed':6s} {'nearest_demo_deg':>16s}")
    for tag, s, pose, closed, d in rows:
        print(f"{tag:7s} {s:5d} {pose[0]:7.1f} {pose[1]:7.1f} {str(closed):6s} {d:16.1f}")
    succ = [d for _, s, _, _, d in rows if s == 4]
    fail = [d for _, s, _, _, d in rows if s <= 1]
    mid = [d for _, s, _, _, d in rows if s in (2, 3)]
    print(f"median nearest-demo distance: score 4 -> {np.median(succ):.1f} deg (n={len(succ)}) | "
          f"score 2-3 -> {np.median(mid):.1f} deg (n={len(mid)}) | score <=1 -> {np.median(fail):.1f} deg (n={len(fail)})")
    print(f"max nearest-demo distance among score-4 trials: {max(succ):.1f} deg; "
          f"min among score<=1 trials: {min(fail):.1f} deg")

    # Second test: local demo density — how many demo grasps sit in the same 5x5 cell of
    # (pan, lift) space as the attempt. Coverage-predicts-capability would show low counts
    # for failures and high counts for successes.
    pe = np.linspace(demos[:, 0].min() - 0.1, demos[:, 0].max() + 0.1, 6)
    le = np.linspace(demos[:, 1].min() - 0.1, demos[:, 1].max() + 0.1, 6)
    H, _, _ = np.histogram2d(demos[:, 0], demos[:, 1], bins=[pe, le])
    print("\ndemos per 5x5 cell (columns = pan bins left->right; rows = lift bins far->near):")
    print(H.T[::-1].astype(int))
    cell = {}
    for tag, s, pose, closed, d in rows:
        i = min(max(int(np.searchsorted(pe, pose[0])) - 1, 0), 4)
        j = min(max(int(np.searchsorted(le, pose[1])) - 1, 0), 4)
        cell[tag] = int(H[i, j])
    for label, pred in (("score 4", lambda s: s == 4), ("score 2-3", lambda s: s in (2, 3)), ("score <=1", lambda s: s <= 1)):
        c = [cell[t] for t, s, *_ in rows if pred(s)]
        print(f"demos in attempt's cell, {label:9s}: {sorted(c)}  (median {np.median(c):.1f})")
    print("\nREAD THE NUMBERS, NOT THE HEADLINE: if the medians above do not separate, this proxy does not\n"
          "support 'coverage predicts capability' at n=18. See docs/coverage_map.md for the reading.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(demos[:, 0], demos[:, 1], s=55, c="#9a9a9a", marker="x", linewidths=1.2,
               label=f"demo grasp poses (n={len(demos)})", zorder=2)
    # 5x5 cell grid used for the density test, drawn faintly so the counts are readable
    for x in pe:
        ax.axvline(x, color="#dddddd", lw=0.8, zorder=1)
    for y in le:
        ax.axhline(y, color="#dddddd", lw=0.8, zorder=1)
    for i in range(5):
        for j in range(5):
            ax.text((pe[i] + pe[i + 1]) / 2, le[j] + 1.0, str(int(H[i, j])), ha="center", va="bottom",
                    fontsize=7, color="#b0b0b0", zorder=1)
    cmap = {4: "#1a9641", 3: "#a6d96a", 2: "#fdae61", 1: "#d7191c", 0: "#7b1113"}
    for tag, s, pose, closed, d in rows:
        ax.scatter(pose[0], pose[1], s=140, c=cmap[s], edgecolors="black", linewidths=1.2,
                   marker="o" if tag.startswith("ft_A") else ("s" if tag.startswith("ft_B") else "^"), zorder=3)
        ax.annotate(tag.replace("ft_", ""), (pose[0], pose[1]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    for s, c in cmap.items():
        ax.scatter([], [], c=c, edgecolors="black", s=80, label=f"score {s}")
    ax.scatter([], [], c="white", edgecolors="black", marker="o", s=80, label="A: trained positions")
    ax.scatter([], [], c="white", edgecolors="black", marker="s", s=80, label="B: novel positions")
    ax.scatter([], [], c="white", edgecolors="black", marker="^", s=80, label="C: novel objects")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.95)
    ax.set_xlabel("shoulder_pan at grasp (deg)  ←left · right→")
    ax.set_ylabel("shoulder_lift at grasp (deg)")
    ax.set_title("S3 coverage map: demo grasp poses (grey x, count per cell) vs 18 eval grasp attempts by score", fontsize=10)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
