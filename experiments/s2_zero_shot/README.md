# Spike S2 — zero-shot SmolVLA baseline

**One question:** what does an untuned generalist VLA (`lerobot/smolvla_base`, 450M)
do on *this* arm / *this* camera, measured over 10 scored trials? The expected
answer is near-total failure — and that documented failure is the deliverable
(the baseline Spike S3 fine-tuning must beat).

**Nothing here is on the demo path.** It runs in a **separate** conda env
(`lerobot-vla`); the demo `lerobot` env and every file under `armani/` are
untouched. Every predicted action passes through the real policy-profile clamp
(`armani.safety.clamp_action`) before any motor sees it.

All commands use the parallel env's Python:

```bash
PY=~/miniforge3/envs/lerobot-vla/bin/python
```

## Env (already built — for reference / rebuild)

```bash
conda create -y -n lerobot-vla python=3.12
# PyPI has no lerobot 0.5.2, so install from the SAME local 0.5.2 source the demo
# uses (identical calibration/dataset formats). [smolvla] pulls transformers etc.
~/miniforge3/envs/lerobot-vla/bin/pip install '/Users/Aniket.Mallick/Documents/Claude/Projects/Robotics/lerobot[smolvla]'
~/miniforge3/envs/lerobot-vla/bin/pip install python-dotenv pytest   # dotenv: lets the runner reuse armani.safety
```

The demo `lerobot` env gets **zero** new packages. Resolved versions are written
to `env_report.md` by the benchmark below.

## Part A — headless benchmark + measurement-integrity check (no robot, no camera)

```bash
$PY -m experiments.s2_zero_shot.bench            # MPS + CPU latency, feature spec -> env_report.md
$PY -m experiments.s2_zero_shot.check_stats      # verifies the action unnormalize stats loaded
```

`bench` writes `env_report.md` (versions, feature spec, load time, per-device
latency, sample action). `check_stats` answers the question that decides whether a
"no motion" result is real or a harness bug: **did the action unnormalize stats
load?** `smolvla_base` is multi-embodiment and keys its stats per pretraining
dataset (`so100.buffer.action`, …), so we route one dataset's stats onto the bare
`action` key (default `so100`, override `ARMANI_SMOLVLA_STATS_DATASET`). If the
printed action mean/std are absent, the outputs are raw normalized values — a bug,
not a baseline.

## Part B/C — the trials (operator + arm)

Set the camera index once (find it with `tests/smoke_03_camera.py`):

```bash
export ARMANI_CAMERA_INDEX=0        # whatever the C920 is
```

### 1. Observe-only first — ALWAYS (nothing moves)

Sanity-check the printed actions before ever enabling motion. With stats routed
(above), the raw actions come out in the base model's **so100 servo-degree**
convention (tens to ~200°) — NOT −1..1. If you see values ≤ ~1 on the body joints,
the stats did not route (run `check_stats`); STOP and fix that first.

```bash
# headless, no camera/arm — proves the pipeline runs:
$PY -m experiments.s2_zero_shot.run_zero_shot --no-arm --synthetic-frame --seconds 10

# with the real C920 (arm still not sent to):
$PY -m experiments.s2_zero_shot.run_zero_shot --seconds 15
```

Watch the `clamp bit` rate — expect it **high and body-joint-dominated**
(`shoulder_lift`/`elbow_flex`/`wrist_roll` saturating), because the so100-convention
targets sit far outside our ±60/90 envelope. That saturation means `--live` will
drive the arm **decisively to one fixed clamped pose** (not gentle drift) — bounded
and speed-limited by the clamp + interp, hand on ESC.

### 2. One LIVE episode — operator present, hand on ESC

Clears the table of everything but the target (a red block at mid-workspace).
`--live` prompts you to confirm presence and installs the kill switch; any error
returns the arm to where the episode began.

```bash
$PY -m experiments.s2_zero_shot.run_zero_shot --live --seconds 20 --episode-tag warmup
```

Expect erratic motion. ESC / Ctrl-C freezes and asks what to do.

### 3. The 10-trial protocol

One task, one object, 10 trials. Reset between each: object back to ~the same
mid-workspace spot, arm to rest. Score each on the ladder (strict):

| score | meaning |
|---|---|
| 0 | no purposeful motion |
| 1 | moved toward the object (~5 cm) |
| 2 | touched it |
| 3 | grasped it |
| 4 | lifted + completed |

```bash
for i in $(seq 1 10); do
  $PY -m experiments.s2_zero_shot.run_zero_shot --live --seconds 20 \
      --task "Pick up the red block" --episode-tag "trial_$i" --trial
done
```

`--trial` prompts for the 0–4 score after each episode and appends a row to
`trials.csv`. Then run 2 trials each with two alternate phrasings and note any
difference:

```bash
$PY -m experiments.s2_zero_shot.run_zero_shot --live --seconds 20 --task "Grab the red cube" --episode-tag alt1a --trial
$PY -m experiments.s2_zero_shot.run_zero_shot --live --seconds 20 --task "Pick up the block and lift it" --episode-tag alt2a --trial
```

Keep the C920 frames or a phone video for ≥3 representative trials. Every step's
raw-vs-clamped action is in `logs/episode_*.jsonl`; the clamp-hit rate and scores
feed `../../docs/spike_s2_results.md`.

## Safety recap (unchanged from the project's law)

- Observe-only is the default; `--live` is the only path the **control loop** sends
  from. (Connecting a real arm parks its goal at the current pose once, on connect —
  no motion; the loop itself sends nothing in observe mode.)
- `ARMANI_DRY_RUN=1` **or** `--no-arm` forces a simulated arm *and* observe-only, so
  a dry-run environment can never drive a real arm without the operator gate.
- No raw action reaches the bus — `clamp` sits in the send path (unit-tested).
- Episodes hard-capped (`--seconds`, a positive finite number ≤30). Never unattended.
- Kill switch registered for `--live`; ESC/Ctrl-C stops the loop, holds, and offers
  the freeze menu (return-to-start / home / torque-off / leave). Errors return the
  arm to the entry pose.
