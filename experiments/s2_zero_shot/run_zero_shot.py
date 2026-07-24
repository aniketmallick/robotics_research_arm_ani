"""Spike S2 runner — zero-shot SmolVLA on the SO-101, one explicit control loop.

Deliberately NOT ``lerobot-record``: we want our own loop so the policy clamp
(CLAUDE.md safety rule 2) sits visibly in the send path and we can log raw vs
clamped every step. Flow per step:

    get state + camera frame  ->  policy predicts a joint action  ->
    POLICY CLAMP (+ log raw vs clamped)  ->  [--live only] send to the arm.

Observe-only is the default. ``--live`` is the ONLY path that sends to a motor,
and it first: confirms the operator is present, installs the kill switch, and
wraps the loop in ``armani.safety.SafeMotion`` so any error returns the arm to
where the episode began. Episodes are hard-capped (``--seconds``, default 20,
never above 30). Nothing here runs unattended.

Run ``python -m experiments.s2_zero_shot.run_zero_shot --help`` for the flags,
or see the README for the observe-only -> single --live -> 10-trial sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from experiments.s2_zero_shot import clamp

# --- constants -----------------------------------------------------------
DEFAULT_TASK = "Pick up the red block"
DEFAULT_SECONDS = 20.0
MAX_EPISODE_SECONDS = 30.0  # hard cap; the spike says episodes ~20 s, never unattended
DEFAULT_HZ = 10.0

_HERE = Path(__file__).resolve().parent
LOG_DIR = _HERE / "logs"
TRIALS_CSV = _HERE / "trials.csv"
TRIALS_HEADER = (
    "timestamp",
    "episode_tag",
    "task",
    "device",
    "live",
    "n_steps",
    "clamp_bit_rate",
    "score",
    "note",
)


# --- injectable surfaces (kept tiny so the loop is testable with fakes) ---
class ArmLike(Protocol):
    def read_positions(self) -> dict[str, float]: ...
    def send(self, action: dict[str, float]) -> dict[str, float]: ...


# infer_fn(state, frame_bgr, task) -> raw joint action dict (unclamped).
InferFn = Callable[[dict[str, float], np.ndarray, str], dict[str, float]]
# read_frame(step) -> BGR frame.
FrameFn = Callable[[int], np.ndarray]
# sink(record) -> None. Appends one JSONL record (or list.append in tests).
Sink = Callable[[dict], None]


@dataclass
class EpisodeStats:
    n_steps: int = 0
    elapsed_s: float = 0.0
    live: bool = False
    clamp_bit_steps: int = 0
    invalid_steps: int = 0  # NaN / unknown-joint predictions the clamp rejected
    per_joint_bit_counts: Counter = field(default_factory=Counter)
    clamp_source: str = ""

    @property
    def clamp_bit_rate(self) -> float:
        return self.clamp_bit_steps / self.n_steps if self.n_steps else 0.0


def run_episode(
    *,
    arm: ArmLike,
    read_frame: FrameFn,
    infer_fn: InferFn,
    task: str,
    live: bool,
    hz: float,
    seconds: float,
    sink: Sink,
    clamp_profile: str = "policy",
    now: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    stop: Callable[[], bool] = lambda: False,
) -> EpisodeStats:
    """Run one capped episode. Returns stats. Sends to the arm ONLY when ``live``.

    Every predicted action is clamped and logged before any send. A prediction
    the clamp rejects (NaN / unknown joint) is logged and dropped — never sent.
    """
    seconds = float(seconds)
    # A non-finite or non-positive cap would defeat the loop guard entirely
    # (min(nan, 30) is nan, and `elapsed >= nan` is always False -> unbounded,
    # unattended loop). The hard cap must be structural, so reject it outright.
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"seconds must be a positive finite number, got {seconds!r}")
    seconds = min(seconds, MAX_EPISODE_SECONDS)
    period = 1.0 / hz if hz > 0 else 0.0
    stats = EpisodeStats(live=live, clamp_source=clamp.clamp_source(clamp_profile))

    start = now()
    step = 0
    while True:
        elapsed = now() - start
        if elapsed >= seconds or stop():
            break

        step_t = now()
        state = arm.read_positions()
        frame = read_frame(step)

        infer_start = now()
        raw = infer_fn(state, frame, task)
        infer_ms = (now() - infer_start) * 1000.0

        record: dict = {
            "step": step,
            "t_rel_s": round(elapsed, 4),
            "task": task,
            "live": live,
            "infer_ms": round(infer_ms, 2),
            "raw": {j: round(float(v), 4) for j, v in raw.items()},
        }

        try:
            result = clamp.policy_clamp(raw, profile=clamp_profile)
        except ValueError as exc:
            # A garbage prediction (NaN/inf/unknown joint). Log and DROP it —
            # fail closed, never send. This is exactly the failure the spike
            # wants counted, not smoothed over.
            stats.invalid_steps += 1
            record["invalid_action"] = str(exc)
            sink(record)
            step += 1
            _pace(sleep, now, step_t, period)
            continue

        record["clamped"] = {j: round(float(v), 4) for j, v in result.clamped.items()}
        record["clamp_bit"] = result.bit
        record["clamp_bit_joints"] = list(result.bit_joints)

        if result.bit:
            stats.clamp_bit_steps += 1
            stats.per_joint_bit_counts.update(result.bit_joints)

        if live:
            sent = arm.send(result.clamped)
            record["sent"] = {j: round(float(v), 4) for j, v in sent.items()}
        else:
            record["sent"] = None

        sink(record)
        step += 1
        _pace(sleep, now, step_t, period)

    stats.n_steps = step
    stats.elapsed_s = now() - start
    return stats


def _pace(sleep: Callable[[float], None], now: Callable[[], float], step_t: float, period: float) -> None:
    """Sleep just enough to hold the target rate; never sleep negative."""
    if period <= 0:
        return
    remaining = period - (now() - step_t)
    if remaining > 0:
        sleep(remaining)


# --- trial CSV -----------------------------------------------------------
def append_trial_row(csv_path: Path, row: dict) -> None:
    """Append one scored trial, writing the header if the file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIALS_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in TRIALS_HEADER})


# --- execution mode ------------------------------------------------------
def resolve_execution(args: argparse.Namespace, config_dry_run: bool) -> tuple[bool, bool]:
    """Decide (dry, live) from the CLI and the config's DRY_RUN.

    CRITICAL safety coupling: ``ARMANI_DRY_RUN`` and ``--no-arm`` BOTH force a
    simulated arm AND observe-only. Without this, ``ARMANI_DRY_RUN=1 --live``
    would connect a REAL follower (dry_run keyed only off --no-arm) while
    ``safety.require_operator`` auto-approves under DRY_RUN — a fail-open on
    safety rule 1 (real motion with no operator confirmation). Tying both signals
    to one ``dry`` value makes that impossible: a dry environment never drives a
    real arm.
    """
    dry = bool(args.no_arm or config_dry_run)
    live = bool(args.live and not dry)
    return dry, live


# --- device --------------------------------------------------------------
def select_device(requested: str) -> str:
    """Resolve ``auto`` to mps if available, else cpu. Honour an explicit ask."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# --- CLI -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_zero_shot",
        description="Zero-shot SmolVLA baseline runner (Spike S2). Observe-only by default.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="instruction string fed to the policy")
    parser.add_argument("--live", action="store_true", help="SEND to the arm (operator + kill switch required)")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help=f"episode cap (<= {MAX_EPISODE_SECONDS:g})")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ, help="control rate")
    parser.add_argument("--episode-tag", default="ep", help="id used in the log/trial rows")
    parser.add_argument("--trial", action="store_true", help="after the episode, prompt for a 0-4 score and append to trials.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="torch device")
    parser.add_argument("--camera-index", type=int, default=None, help="C920 OpenCV index (default: ARMANI_CAMERA_INDEX)")
    parser.add_argument("--synthetic-frame", action="store_true", help="headless: feed a generated frame, no camera")
    parser.add_argument("--no-arm", action="store_true", help="headless: use a simulated arm (implies observe-only)")
    parser.add_argument("--policy-path", default=None,
                        help="LOCAL fine-tuned checkpoint dir to load instead of lerobot/smolvla_base "
                             "(Spike S3; default: ARMANI_SMOLVLA_CHECKPOINT, else the base model)")
    parser.add_argument("--clamp-profile", choices=["policy", "recorded"], default=None,
                        help="send-path clamp envelope. Default 'policy' (S2). 'recorded' (wider, "
                             "physical−2°) is RATIFIED for the FINE-TUNED S3 eval ONLY (requires "
                             "--policy-path); refused on the base model. Also ARMANI_EVAL_CLAMP_PROFILE.")
    return parser


def _banner(task: str, device: str, live: bool, clamp_profile: str = "policy") -> None:
    print("=" * 68)
    print("  SPIKE S2 — zero-shot SmolVLA baseline (untuned generalist VLA)")
    print(f"  task   : {task!r}")
    print(f"  device : {device}")
    print(f"  clamp  : {clamp.clamp_source(clamp_profile)}  (every action clamped before send)")
    if clamp_profile == "recorded":
        print("  *** RECORDED envelope (wider, physical−2°) — RATIFIED for the FINE-TUNED eval")
        print("      ONLY. Operator present, kill switch armed, hand on power. ***")
    print(f"  mode   : {'LIVE — the arm WILL move' if live else 'OBSERVE-ONLY — the control loop sends nothing'}")
    if live:
        print("  *** clear the table of all but the target. Hand on ESC. ***")
    else:
        print("  (a real arm still parks its goal at the current pose on connect — no motion)")
    print("=" * 68)


def _score_trial(stats: EpisodeStats) -> dict | None:
    """Prompt the operator for the 0-4 ladder score. Non-interactive -> skip."""
    if not sys.stdin.isatty():
        print("[trial] non-interactive; skipping scoring (run with a terminal to score).")
        return None
    print(
        "\nScore this trial (strict):\n"
        "  0 no purposeful motion   1 moved toward the object (~5 cm)\n"
        "  2 touched it   3 grasped it   4 lifted + completed"
    )
    try:
        raw = input("Score [0-4]: ").strip()
        score = int(raw)
        if not 0 <= score <= 4:
            raise ValueError
    except (ValueError, EOFError, KeyboardInterrupt):
        print("  not a 0-4 score; trial not recorded.")
        return None
    note = input("Note (optional): ").strip()
    return {"score": score, "note": note}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = select_device(args.device)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        print(f"ERROR: --seconds must be a positive finite number, got {args.seconds!r}.")
        return 2

    # Lazy imports: the heavy ML stack and armani.motion load only when actually
    # running, so --help and the unit tests stay fast and hardware-free.
    from experiments.s2_zero_shot import smolvla_io
    from armani import config, motion, safety

    # Resolve checkpoint + clamp profile UP FRONT so a bad config fails before we open
    # the camera or energise the arm. resolve_clamp_profile refuses `recorded` on the
    # base model (protects the closed S2 baseline) — that error must fire pre-hardware.
    checkpoint = smolvla_io.resolve_checkpoint(args.policy_path)
    is_base = checkpoint == smolvla_io.MODEL_ID
    clamp_profile = clamp.resolve_clamp_profile(args.clamp_profile, is_base)

    # DRY_RUN (env) or --no-arm both force a simulated arm AND observe-only.
    dry, live = resolve_execution(args, config.DRY_RUN)

    _banner(args.task, device, live, clamp_profile)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"episode_{args.episode_tag}_{stamp}.jsonl"
    log_handle = log_path.open("w")

    def sink(record: dict) -> None:
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()

    # Everything acquired below (camera, serial port, loaded policy) is released
    # in the finally, so a failure part-way through never leaks an open device.
    from experiments.s2_zero_shot import camera

    stream = None
    arm = None
    exit_code = 0
    try:
        # Camera source.
        if args.synthetic_frame:
            read_frame: FrameFn = camera.synthetic_frame
            print("[camera] synthetic frames (headless).")
        else:
            index = args.camera_index if args.camera_index is not None else config.CAMERA_INDEX
            if index is None:
                print("ERROR: no camera index. Pass --camera-index or set ARMANI_CAMERA_INDEX (see smoke_03).")
                return 2
            stream = camera.CameraStream(index)
            read_frame = lambda step: stream.read_bgr()  # noqa: E731 - step is ignored for a live camera
            print(f"[camera] C920 open at index {index} (must not move during the run).")

        # Arm (energised but holding position; motion is gated below by --live).
        arm = motion.connect(dry_run=dry, interactive=not dry)

        # Policy + inference closure (checkpoint/is_base resolved up front, above).
        print(f"[policy] loading {'lerobot/smolvla_base (zero-shot)' if is_base else checkpoint + ' (fine-tuned)'} "
              f"on {device} (first load is slow) ...")
        infer_fn, spec = smolvla_io.make_infer_fn(device=device, checkpoint=None if is_base else checkpoint)
        print(f"[policy] {spec.summary()}")

        if live:
            safety.install_kill_switch()
            if not safety.require_operator("run a LIVE zero-shot SmolVLA episode (untuned — expect erratic motion)"):
                print("Operator did not confirm — staying observe-only, nothing sent.")
                live = False

        run = lambda: run_episode(  # noqa: E731 - a tiny local alias, clearer than nesting
            arm=arm,
            read_frame=read_frame,
            infer_fn=infer_fn,
            task=args.task,
            live=live,
            hz=args.hz,
            seconds=args.seconds,
            sink=sink,
            clamp_profile=clamp_profile,
            stop=safety.stop_requested,
        )

        if live:
            entry_pose = arm.read_positions()
            # Any error returns the arm to the pose the episode began at (rule 4).
            with safety.SafeMotion(arm, "s2 zero-shot episode"):
                stats = run()
            # ESC / Ctrl-C sets the stop flag; the loop breaks and the arm holds.
            # Deliver the freeze menu the kill-switch message promised (return to
            # start / home / torque-off / leave) — safety rule 7.
            if safety.stop_requested():
                safety.handle_freeze(arm, entry_pose)
        else:
            stats = run()

        _report(stats, log_path, clamp_profile)
        sink(_summary_record(args, device, stats))

        if args.trial:
            scored = _score_trial(stats)
            if scored is not None:
                append_trial_row(
                    TRIALS_CSV,
                    {
                        "timestamp": stamp,
                        "episode_tag": args.episode_tag,
                        "task": args.task,
                        "device": device,
                        "live": live,
                        "n_steps": stats.n_steps,
                        "clamp_bit_rate": round(stats.clamp_bit_rate, 3),
                        "score": scored["score"],
                        "note": scored["note"],
                    },
                )
                print(f"[trial] recorded score {scored['score']} -> {TRIALS_CSV}")
    except KeyboardInterrupt:
        print("\n[interrupted]")
        exit_code = 130
    finally:
        log_handle.close()
        if stream is not None:
            stream.close()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass

    print(f"[log] {log_path}")
    return exit_code


def _summary_record(args: argparse.Namespace, device: str, stats: EpisodeStats) -> dict:
    return {
        "event": "episode_summary",
        "episode_tag": args.episode_tag,
        "task": args.task,
        "device": device,
        "live": stats.live,
        "n_steps": stats.n_steps,
        "elapsed_s": round(stats.elapsed_s, 3),
        "hz_target": args.hz,
        "clamp_bit_steps": stats.clamp_bit_steps,
        "clamp_bit_rate": round(stats.clamp_bit_rate, 4),
        "invalid_steps": stats.invalid_steps,
        "per_joint_bit_counts": dict(stats.per_joint_bit_counts),
        "clamp_source": stats.clamp_source,
    }


def _report(stats: EpisodeStats, log_path: Path, clamp_profile: str = "policy") -> None:
    print("-" * 68)
    print(f"  steps            : {stats.n_steps}  over {stats.elapsed_s:.1f}s")
    print(f"  clamp bit        : {stats.clamp_bit_steps}/{stats.n_steps} steps ({stats.clamp_bit_rate * 100:.0f}%)")
    if stats.per_joint_bit_counts:
        worst = ", ".join(f"{j}:{n}" for j, n in stats.per_joint_bit_counts.most_common())
        print(f"  clamp by joint   : {worst}")
    # Warn-only diagnostic: under the wider `recorded` profile, a clamp bite means the
    # policy commanded past even that envelope — i.e. into the outermost 2° margin that
    # keeps the arm off its hard stops (recorded = calibrated range −2°), not lost
    # workspace. Signal, never a block — the action was still clamped and sent bounded.
    if clamp_profile == "recorded" and stats.clamp_bit_steps:
        print(f"  [warn] policy strained PAST the recorded envelope on {stats.clamp_bit_steps}/"
              f"{stats.n_steps} steps — commanding into the outermost 2° off-hard-stop margin,")
        print("         not lost workspace. Diagnostic only; every action was still clamped + bounded.")
    if stats.invalid_steps:
        print(f"  invalid (dropped): {stats.invalid_steps} steps (NaN/unknown — never sent)")
    print("-" * 68)


if __name__ == "__main__":
    raise SystemExit(main())
