#!/usr/bin/env python
"""Back up the recorded datasets. Run this before you travel, and after re-recording.

The gesture and pick recordings are the only genuinely irreplaceable artefacts in
this project. They live in the HuggingFace cache OUTSIDE the repo:

    ~/.cache/huggingface/lerobot/anikmall/...

Nothing in git protects them. A cache clear, a `huggingface-cli delete-cache`, a
new laptop or a full disk destroys the demo, and re-recording needs the arm, the
leader, the table and half an hour you will not have at a venue.

This copies both datasets into armani/data/dataset_backup/<name>/<timestamp>/,
which is gitignored (they are tens of MB of binary and do not belong in git —
put that directory on a USB stick or in cloud storage).

    python scripts/backup_datasets.py            # back up
    python scripts/backup_datasets.py --list     # what is already backed up
    python scripts/backup_datasets.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import _bootstrap  # noqa: E402,F401  (verifies the interpreter, exits if wrong)

from armani import config  # noqa: E402
from armani.logutil import log_event  # noqa: E402


def datasets() -> list[tuple[str, Path]]:
    """The datasets the demo actually depends on, as (label, root)."""
    return [
        ("gestures", config.GESTURE_DATASET_ROOT),
        ("picks", config.PICK_DATASET_ROOT),
    ]


def episode_count(root: Path) -> int | None:
    info = root / "meta" / "info.json"
    try:
        return int(json.loads(info.read_text()).get("total_episodes", 0))
    except (OSError, ValueError, TypeError):
        return None


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="say what would be copied")
    parser.add_argument("--list", action="store_true", help="show existing backups and exit")
    args = parser.parse_args()

    print("=== ARM-ANI dataset backup ===")
    print(f"destination: {config.DATASET_BACKUP_DIR}")

    if args.list:
        return _list()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    failures = 0
    copied: list[str] = []

    for label, root in datasets():
        print(f"\n--- {label} ---")
        print(f"  source : {root}")
        if not (root / "meta" / "info.json").is_file():
            print(f"  MISSING — nothing at {root}. Has it been recorded?", file=sys.stderr)
            failures += 1
            continue

        episodes = episode_count(root)
        size = directory_size(root)
        print(f"  {episodes} episodes, {human(size)}")

        target = config.DATASET_BACKUP_DIR / label / f"{root.name}_{stamp}"
        if args.dry_run:
            print(f"  [dry-run] would copy to {target}")
            continue

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # dirs_exist_ok=False on purpose: a timestamped target should never
            # already exist, and silently merging into one would produce a
            # backup that is a blend of two recordings.
            shutil.copytree(root, target)
        except OSError as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue

        backed_up = directory_size(target)
        if backed_up != size:
            # A short copy is worse than no copy, because you would trust it.
            print(f"  FAILED: copied {human(backed_up)} but the source is {human(size)}", file=sys.stderr)
            failures += 1
            continue

        print(f"  copied -> {target}")
        copied.append(f"{label}:{episodes}ep")
        log_event("dataset_backup", dataset=label, source=str(root),
                  target=str(target), episodes=episodes, bytes=size)

    if args.dry_run:
        print("\n[dry-run] nothing was written.")
        return 0

    print()
    if failures:
        print(f"{failures} dataset(s) FAILED to back up — fix this before travelling.", file=sys.stderr)
        return 1

    print(f"Backed up: {', '.join(copied)}")
    print(
        "\n  These are gitignored on purpose (tens of MB of binary).\n"
        "  COPY armani/data/dataset_backup/ TO A USB STICK OR CLOUD DRIVE.\n"
        "  A backup that only exists on the laptop you might spill coffee on is not a backup.\n"
    )
    print("  To restore: copy a timestamped directory back to the source path printed above,")
    print("  or point ARMANI_GESTURE_ROOT / ARMANI_PICK_ROOT straight at the backup.")
    return 0


def _list() -> int:
    if not config.DATASET_BACKUP_DIR.is_dir():
        print("\nNo backups yet. Run: python scripts/backup_datasets.py")
        return 1
    found = 0
    for label_dir in sorted(config.DATASET_BACKUP_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        print(f"\n{label_dir.name}:")
        for backup in sorted(label_dir.iterdir()):
            if backup.is_dir():
                episodes = episode_count(backup)
                print(f"  {backup.name}  ({episodes} episodes, {human(directory_size(backup))})")
                found += 1
    if not found:
        print("\nNo backups yet. Run: python scripts/backup_datasets.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
