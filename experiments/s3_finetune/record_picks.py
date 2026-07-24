"""Record the S3 fine-tune dataset — a thin wrapper over lerobot 0.5.2 `lerobot-record`.

Runs in the DEMO `lerobot` env (the one with calibrated teleop). It reuses the exact
flags verified working in docs/recording_picks.md and ADDS the C920 camera in the
format from lerobot's AGENT_GUIDE, because SmolVLA needs images and the taught-zone
datasets had none.

    python experiments/s3_finetune/record_picks.py            # DRY RUN: print the command
    python experiments/s3_finetune/record_picks.py --go       # OPERATOR + ARM: record
    python experiments/s3_finetune/record_picks.py --go --resume        # add more episodes
    python experiments/s3_finetune/record_picks.py --num-episodes 10 --go

Ports come from ARMANI_FOLLOWER_PORT / ARMANI_LEADER_PORT (they change per plug-in on
macOS — set them for the session). READ the SOP before recording: consistency is the
whole point of this spike.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.s3_finetune import s3_config  # noqa: E402


def camera_arg(index: int) -> str:
    """The `--robot.cameras` value in lerobot's draccus dict syntax (AGENT_GUIDE format)."""
    return (
        f"{{{s3_config.CAMERA_NAME}: {{type: opencv, index_or_path: {index}, "
        f"width: {s3_config.CAMERA_W}, height: {s3_config.CAMERA_H}, fps: {s3_config.CAMERA_FPS}}}}}"
    )


def build_record_argv(
    *,
    follower_port: str,
    leader_port: str,
    follower_id: str,
    leader_id: str,
    repo_id: str,
    task: str,
    root: str,
    camera_index: int,
    num_episodes: int,
    episode_time_s: int,
    reset_time_s: int,
    fps: int,
    resume: bool = False,
) -> list[str]:
    """Assemble the lerobot-record argv. Pure — testable without hardware.

    Mirrors docs/recording_picks.md (use_degrees, pinned --dataset.root, push_to_hub
    off) and adds --robot.cameras so the dataset captures images for SmolVLA.
    """
    argv = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={follower_id}",
        "--robot.use_degrees=true",
        f"--robot.cameras={camera_arg(camera_index)}",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={leader_id}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={root}",
        f"--dataset.single_task={task}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.fps={fps}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
        "--dataset.push_to_hub=false",
        "--display_data=true",
    ]
    if resume:
        argv.append("--resume=true")
    return argv


def _ports() -> tuple[str, str, str, str]:
    """(follower_port, leader_port, follower_id, leader_id) from armani.config/env."""
    from armani import config

    if not config.FOLLOWER_PORT or not config.LEADER_PORT:
        raise SystemExit(
            "ARMANI_FOLLOWER_PORT and ARMANI_LEADER_PORT must be set (they change per USB "
            "plug-in on macOS). Find them with `ls /dev/tty.usbmodem*` / `lerobot-find-port`, "
            "then export them, and re-run."
        )
    return config.FOLLOWER_PORT, config.LEADER_PORT, config.FOLLOWER_ID, config.LEADER_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="record_picks", description="Record the S3 pick dataset.")
    parser.add_argument("--go", action="store_true", help="actually record (default is a dry-run print)")
    parser.add_argument("--num-episodes", type=int, default=s3_config.NUM_EPISODES)
    parser.add_argument("--resume", action="store_true", help="append episodes to an existing dataset")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    follower_port, leader_port, follower_id, leader_id = _ports()
    record_argv = build_record_argv(
        follower_port=follower_port, leader_port=leader_port,
        follower_id=follower_id, leader_id=leader_id,
        repo_id=s3_config.REPO_ID, task=s3_config.TASK, root=str(s3_config.dataset_root()),
        camera_index=s3_config.CAMERA_INDEX, num_episodes=args.num_episodes,
        episode_time_s=s3_config.EPISODE_TIME_S, reset_time_s=s3_config.RESET_TIME_S,
        fps=s3_config.FPS, resume=args.resume,
    )
    print("=== S3 record (dataset: %s, task: %r) ===" % (s3_config.REPO_ID, s3_config.TASK))
    print("  *** READ experiments/s3_finetune/SOP.md FIRST — consistency is the point. ***")
    print("  *** the C920 must be at the same fixed pose for every demo. ***\n")
    print("command:\n  " + " \\\n    ".join(record_argv) + "\n")
    if not args.go:
        print("[dry-run] nothing recorded. Add --go to record (operator + arm, keys: → next, ← redo, ESC finish).")
        return 0
    print("Recording. Keys: → next episode, ← redo current, ESC finish. Hand near the arm.\n")
    return subprocess.run(record_argv).returncode


if __name__ == "__main__":
    raise SystemExit(main())
