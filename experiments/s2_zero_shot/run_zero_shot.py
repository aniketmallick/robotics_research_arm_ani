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
never above ``MAX_EPISODE_SECONDS``). Nothing here runs unattended.

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
# Hard cap; still never unattended (operator present, kill switch armed, --live gated).
# Raised 30 -> 90 for the S3 fine-tuned eval: the demos were recorded at 30 fps, ~600
# waypoints per episode, and this loop executes ONE waypoint per step at the --hz pace,
# so the old 30 s cap ended the pick before the grasp at any rate below 20 Hz.
#
# 90 is HEADROOM, not the protocol. The ratified S3 trials run `--hz 30 --seconds 30`
# (demo speed: 600 waypoints = 20 s, so 30 s is ~1.5 demos). The cap has to clear the
# 10 Hz fallback too — ~60 s of playback — hence 90.
#
# Both rates are reachable because the loop is PACE-bound, not inference-bound. Measured
# (2026-08-14, mps): median step cost 9 ms, inference 18% of wall time, because the
# policy serves 50 waypoints per plan (n_action_steps=50) and only re-plans every 50th
# step, at ~400 ms. At the 10 Hz default that gave 813 waypoints in 90.1 s (9.0 Hz) —
# the shortfall from 10 Hz is those re-plan stalls alone.
#
# Why the 10 Hz fallback is still a valid trajectory: the policy config has
# `n_obs_steps: 1` — it conditions on a single frame and a single state, with no history
# and no velocity term. Executing the same waypoint sequence at 10 Hz instead of 30 Hz
# therefore does not change what the policy observes at any waypoint; only wall-clock
# differs. The trajectory is preserved, playback is ~3x slower. (It does stretch the
# OPEN-LOOP window between plans from ~1.67 s to ~5 s, which is why 30 Hz — training's
# own cadence — is the ratified default for scoring.)
MAX_EPISODE_SECONDS = 90.0
DEFAULT_HZ = 10.0

# One demo in armani_pick_red_v1 is ~600 waypoints (50 episodes x ~600 frames at 30 fps).
# An episode that produced fewer steps than that closed its window before the trained
# trajectory could finish — scoring it as a policy failure is exactly the false zero the
# cap was raised to prevent, so the report says so out loud. Fine-tuned runs only; the
# untuned base has no trajectory length to fall short of.
DEMO_WAYPOINTS = 600

_HERE = Path(__file__).resolve().parent
LOG_DIR = _HERE / "logs"
TRIALS_CSV = _HERE / "trials.csv"
TRIALS_HEADER = (
    "timestamp",
    "episode_tag",
    "task",
    # Which weights produced this score. A results file that cannot answer that is not
    # quotable; model_revision is the source commit when one exists, blank when it
    # genuinely does not (never a guess) — see smolvla_io.resolve_revision.
    "model_ref",
    "model_revision",
    "device",
    # Playback rate is an experimental variable now (10 Hz vs the ratified 30), so the
    # results file has to carry it: n_steps/elapsed_s gives the rate ACHIEVED, which is
    # what the "was it run at training speed?" question actually needs.
    "hz",
    "elapsed_s",
    "live",
    "clamp_profile",
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
    clamp_profile: str = ""
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
    # (min(nan, cap) is nan, and `elapsed >= nan` is always False -> unbounded,
    # unattended loop). The hard cap must be structural, so reject it outright.
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"seconds must be a positive finite number, got {seconds!r}")
    seconds = min(seconds, MAX_EPISODE_SECONDS)
    period = 1.0 / hz if hz > 0 else 0.0
    stats = EpisodeStats(
        live=live, clamp_profile=clamp_profile, clamp_source=clamp.clamp_source(clamp_profile)
    )

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
            # Per step, not once per file: a run killed mid-episode still leaves every
            # logged action self-describing about the envelope it was bounded by.
            "clamp_profile": clamp_profile,
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
    """Append one scored trial, writing the header if the file is new.

    A file whose header does not match ``TRIALS_HEADER`` is a hard error, not an
    append: DictWriter writes values in OUR column order under THEIR header, so a
    stale file (one written before ``clamp_profile`` existed) would silently shift
    every column of the results table. Migrate the file, or move it aside.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    if not new_file:
        with csv_path.open(newline="") as handle:
            existing = next(csv.reader(handle), None)
        if existing is not None and tuple(existing) != TRIALS_HEADER:
            raise ValueError(
                f"{csv_path} header {existing} does not match {list(TRIALS_HEADER)}; refusing to "
                "append misaligned rows. Migrate the file or move it aside."
            )
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


def _live_confirmation(model_ref: str, is_base: bool, clamp_profile: str, seconds: float, hz: float) -> str:
    """What the operator is actually confirming (completes "About to ...").

    Safety rule 1 is INFORMED consent: a prompt that describes the wrong experiment is
    not a confirmation. All four of these change what to watch for and how fast a hand
    has to move — which model, how wide the envelope, how long, and whether the arm is
    supposed to touch the table.
    """
    cap = min(float(seconds), MAX_EPISODE_SECONDS)
    envelope = f"{clamp_profile} clamp envelope" + (
        " (WIDER than policy — physical−2°)" if clamp_profile == "recorded" else ""
    )
    if is_base:
        return (
            "run a LIVE ZERO-SHOT SmolVLA episode (untuned base model — expect erratic motion, "
            f"snapping decisively to one clamped pose); up to {cap:g}s at {hz:g} Hz, {envelope}"
        )
    return (
        f"run a LIVE FINE-TUNED SmolVLA episode ({model_ref}) — a trained policy that REACHES "
        "FOR THE TABLE and closes the gripper on purpose, so table contact is intended, not a "
        f"fault; up to {cap:g}s at {hz:g} Hz, {envelope}"
    )


def _banner(task: str, device: str, live: bool, clamp_profile: str, model_ref: str, is_base: bool,
            revision: str = "") -> None:
    print("=" * 68)
    # Say which spike this run IS. A fine-tuned eval printing "zero-shot baseline" is
    # the same class of mislabelling as printing the wrong stats source.
    if is_base:
        print("  SPIKE S2 — zero-shot SmolVLA baseline (untuned generalist VLA)")
    else:
        print("  SPIKE S3 — FINE-TUNED SmolVLA eval")
    print(f"  model  : {model_ref}")
    print(f"  rev    : {revision or '(none recorded — not a Hub download)'}")
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
    # smolvla_io's module import is cheap (numpy only; torch/lerobot load inside load()).
    from experiments.s2_zero_shot import smolvla_io

    # Resolve checkpoint + clamp profile UP FRONT — before the armani import, before the
    # camera, before the arm is energised. resolve_clamp_profile refuses `recorded` on the
    # base model (protects the closed S2 baseline) AND when armani.safety is unavailable
    # (there is no fallback table for the recorded envelope, and we never guess one). That
    # second refusal is only reachable if it runs BEFORE `from armani import ...`, which
    # would otherwise raise a bare ImportError in exactly the env it is meant to catch.
    checkpoint = smolvla_io.resolve_checkpoint(args.policy_path)
    is_base = checkpoint == smolvla_io.MODEL_ID
    clamp_profile = clamp.resolve_clamp_profile(args.clamp_profile, is_base)
    revision = smolvla_io.resolve_revision(checkpoint)

    from armani import config, motion, safety

    # DRY_RUN (env) or --no-arm both force a simulated arm AND observe-only.
    dry, live = resolve_execution(args, config.DRY_RUN)

    _banner(args.task, device, live, clamp_profile, checkpoint, is_base, revision)

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
        # The numbers behind the stats label, so a wrong-scale load is visible here and
        # not only in the arm's behaviour (compare against check_dataset.py's ranges).
        print(f"[policy] {spec.action_unnorm}")

        if live:
            safety.install_kill_switch()
            if not safety.require_operator(
                _live_confirmation(checkpoint, is_base, clamp_profile, args.seconds, args.hz)
            ):
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

        # Only a fine-tuned run has a trajectory length to fall short of, and an episode
        # the operator stopped on purpose does not need advice about raising --seconds.
        expect = None if (is_base or safety.stop_requested()) else DEMO_WAYPOINTS
        _report(stats, log_path, clamp_profile, expect)
        sink(_summary_record(args, device, stats, checkpoint, revision))

        if args.trial:
            scored = _score_trial(stats)
            if scored is not None:
                row = {
                    "timestamp": stamp,
                    "episode_tag": args.episode_tag,
                    "task": args.task,
                    "model_ref": checkpoint,
                    "model_revision": revision,
                    "device": device,
                    "hz": args.hz,
                    "elapsed_s": round(stats.elapsed_s, 3),
                    "live": live,
                    "clamp_profile": stats.clamp_profile,
                    "n_steps": stats.n_steps,
                    "clamp_bit_rate": round(stats.clamp_bit_rate, 3),
                    "score": scored["score"],
                    "note": scored["note"],
                }
                try:
                    append_trial_row(TRIALS_CSV, row)
                    print(f"[trial] recorded score {scored['score']} -> {TRIALS_CSV}")
                except ValueError as exc:
                    # The header guard refusing is the right call, but it must not also
                    # destroy the score the operator just typed after a live trial. Print
                    # the row so it can be pasted in once the file is migrated.
                    print(f"[trial] NOT recorded — {exc}")
                    print(f"[trial] row: {row}")
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


def _summary_record(
    args: argparse.Namespace, device: str, stats: EpisodeStats, model_ref: str, model_revision: str
) -> dict:
    return {
        "event": "episode_summary",
        "episode_tag": args.episode_tag,
        "task": args.task,
        "model_ref": model_ref,
        "model_revision": model_revision,
        "device": device,
        "live": stats.live,
        "n_steps": stats.n_steps,
        "elapsed_s": round(stats.elapsed_s, 3),
        "hz_target": args.hz,
        "clamp_bit_steps": stats.clamp_bit_steps,
        "clamp_bit_rate": round(stats.clamp_bit_rate, 4),
        "invalid_steps": stats.invalid_steps,
        "per_joint_bit_counts": dict(stats.per_joint_bit_counts),
        "clamp_profile": stats.clamp_profile,
        "clamp_source": stats.clamp_source,
    }


def _report(stats: EpisodeStats, log_path: Path, clamp_profile: str = "policy",
            expect_waypoints: int | None = None) -> None:
    print("-" * 68)
    rate = stats.n_steps / stats.elapsed_s if stats.elapsed_s > 0 else 0.0
    print(f"  steps            : {stats.n_steps}  over {stats.elapsed_s:.1f}s ({rate:.1f} Hz achieved)")
    # A trained trajectory that never got to run is not a policy failure. Warn-only:
    # ESC and deliberately short runs land here too, and the operator judges which.
    if expect_waypoints is not None and stats.n_steps < expect_waypoints:
        print(f"  [warn] {stats.n_steps} steps < ~{expect_waypoints} waypoints in one training demo —")
        print("         the window may have closed BEFORE the trajectory finished. Raise --seconds")
        print("         (or --hz) and re-run before scoring this as a policy failure.")
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
