"""Sanity-check the S3 dataset before training — headless, no GPU, no arm.

Confirms the dataset is what SmolVLA needs: a CAMERA (images), 6-D state/action,
enough episodes, and similar episode lengths (a length outlier is usually a fumbled
take that should be discarded — see the SOP). Reports the LeRobotDataset
`codebase_version` so the Colab training env can be matched to it.

    python experiments/s3_finetune/check_dataset.py
    python experiments/s3_finetune/check_dataset.py --root <path/to/dataset>
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.s3_finetune import s3_config  # noqa: E402

EXPECTED_DIM = 6  # SmolVLA on our SO-101: 6 joints (state) and 6 action dims
MIN_EPISODES = 30


def check_features(features: dict) -> list[str]:
    """Problems with the dataset's feature spec (empty list = fine). Pure/testable."""
    problems: list[str] = []
    action = features.get("action")
    if not action:
        problems.append("no 'action' feature")
    elif list(action.get("shape", [])) != [EXPECTED_DIM]:
        problems.append(f"action shape {action.get('shape')} != [{EXPECTED_DIM}]")
    state = features.get("observation.state")
    if not state:
        problems.append("no 'observation.state' feature")
    elif list(state.get("shape", [])) != [EXPECTED_DIM]:
        problems.append(f"observation.state shape {state.get('shape')} != [{EXPECTED_DIM}]")
    cameras = [k for k in features if k.startswith("observation.images.")]
    if not cameras:
        problems.append(
            "NO camera (observation.images.*) — SmolVLA is a vision policy and cannot train "
            "on this. Was --robot.cameras set during recording? (record_picks.py adds it.)"
        )
    return problems


def length_outliers(lengths: list[int], tol: float = 0.4) -> list[int]:
    """Indices of episodes whose length deviates from the median by more than ``tol``.

    Similar-length demos are an SOP requirement; an outlier is usually a fumbled or
    mis-timed take. Pure/testable.
    """
    if not lengths:
        return []
    median = statistics.median(lengths)
    if median <= 0:
        return list(range(len(lengths)))
    return [i for i, n in enumerate(lengths) if abs(n - median) > tol * median]


def episode_lengths(root: Path) -> list[int]:
    """Per-episode frame counts, in episode order, from meta/episodes/*.parquet."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    pairs: list[tuple[int, int]] = []
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "length"])
        pairs += [
            (int(ep), int(n))
            for ep, n in zip(table.column("episode_index").to_pylist(), table.column("length").to_pylist())
        ]
    pairs.sort()
    return [n for _, n in pairs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="check_dataset", description="Sanity-check the S3 dataset.")
    parser.add_argument("--root", default=None, help="dataset dir (default: the S3 config's dataset_root)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root) if args.root else s3_config.dataset_root()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        print(f"no dataset at {root} (no meta/info.json). Record it first with record_picks.py --go.",
              file=sys.stderr)
        return 1

    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    total_ep = info.get("total_episodes")
    problems = check_features(features)
    if isinstance(total_ep, int) and total_ep < MIN_EPISODES:
        problems.append(f"only {total_ep} episodes — aim for ~{s3_config.NUM_EPISODES} clean demos")

    try:
        lengths = episode_lengths(root)
    except Exception as exc:  # pragma: no cover - depends on the on-disk parquet
        lengths = []
        print(f"  (could not read episode lengths: {exc})")
    outliers = length_outliers(lengths)

    cams = [k for k in features if k.startswith("observation.images.")]
    print(f"dataset: {root}")
    print(f"  codebase_version : {info.get('codebase_version')}   (match the Colab lerobot to this)")
    print(f"  episodes / frames: {total_ep} / {info.get('total_frames')}   fps: {info.get('fps')}")
    print(f"  cameras          : {cams or 'NONE'}")
    print(f"  action / state   : {features.get('action', {}).get('shape')} / "
          f"{features.get('observation.state', {}).get('shape')}")
    if lengths:
        print(f"  episode lengths  : min {min(lengths)} / median {int(statistics.median(lengths))} / "
              f"max {max(lengths)} frames")
        if outliers:
            print(f"  ⚠ length OUTLIERS at episode index {outliers} — likely fumbled takes; re-record "
                  "or drop them (SOP: train on clean successes only).")
    print()

    if problems:
        print("PROBLEMS — fix before training:")
        for problem in problems:
            print("  ✗ " + problem)
        return 1
    print("READY: camera present, 6-D state/action, episodes consistent.")
    print("  ALWAYS visualize before training (push to the Hub, then):")
    print("  https://huggingface.co/spaces/lerobot/visualize_dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
