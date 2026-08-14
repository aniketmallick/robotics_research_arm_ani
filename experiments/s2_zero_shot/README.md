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
# Robot deps for the LIVE/observe path (lerobot lazily require_package()s these when
# motion.connect() builds the SO-101 bus + processor pipeline — the [smolvla] extra
# does NOT pull them). Needed for --live and real-camera observe; NOT for --no-arm.
~/miniforge3/envs/lerobot-vla/bin/pip install feetech-servo-sdk deepdiff pynput
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

> Routing applies to the **base model only**. A fine-tuned checkpoint (`--policy-path`)
> ships its own MEAN_STD stats over *our* dataset, already keyed by the bare feature;
> the runner uses exactly those, never routes a pretraining dataset over them, and
> **refuses to run** if they are absent. The `[policy]` banner names which is active —
> `stats=checkpoint:<dir>` vs `stats=pretrain:so100` — and echoes the action mean/std
> in the send path so a wrong-scale load is visible before the arm moves.
> (`check_stats` itself always loads the base; it cannot see your checkpoint.)

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
raw-vs-clamped action (and the `clamp_profile` it was bounded by) is in
`logs/episode_*.jsonl`; the clamp-hit rate and scores feed
`../../docs/spike_s2_results.md`.

## Part D — evaluating a FINE-TUNED checkpoint (Spike S3)

Same runner, three flags. Full protocol in
[`../s3_finetune/eval.md`](../s3_finetune/eval.md); the mechanics are:

```bash
CKPT=~/models/smolvla_pick_red_v1

# headless first — confirms the checkpoint, its OWN stats, and the clamp envelope:
$PY -m experiments.s2_zero_shot.run_zero_shot --no-arm --synthetic-frame \
    --hz 30 --seconds 45 --clamp-profile recorded --policy-path "$CKPT"
```

Read five lines of that output before trusting anything downstream:

| line | must say |
|---|---|
| header + `model  :` | `SPIKE S3 — FINE-TUNED SmolVLA eval` and your checkpoint dir |
| `rev    :` | the commit the checkpoint was downloaded from (blank if it was never a Hub download) |
| `clamp  :` | `armani.safety.clamp_action(recorded)` — the envelope in the send path |
| `[policy] …` | `stats=checkpoint:<your dir> (own MEAN_STD stats, no pretrain routing)` |
| `[policy] action unnormalize MEAN_STD:` | your dataset's mean/std (cross-check `check_dataset.py`) |

- **`--clamp-profile recorded`** — the wider physical−2° envelope, **architect-ratified
  for the fine-tuned eval ONLY** (the demos legitimately exceed policy ±60°, so a
  `policy` clamp strangles the grasp and scores a false zero). Default stays `policy`;
  `recorded` is **refused on the base model**, and refused outright if `armani.safety`
  is unimportable (there is no fallback table for it — an embedded guess would drift
  from the operator's real calibration). Operator present, kill switch armed.
- **`--hz 30 --seconds 45`** — RATIFIED for the scored trials: replay near the speed the
  demos were teleoperated at. Reachable because the loop is **pace-bound, not
  inference-bound** (median step cost 9 ms; inference 18% of wall time). Measured
  headless: **~22 Hz achieved, not 30** — the ~400 ms re-plan every 50th step
  (`n_action_steps: 50`) plus ~30 ms/step of non-inference work. Report it as ~0.73×
  training speed; per-waypoint behaviour is unchanged (`n_obs_steps: 1`, position-
  conditioned), only the open-loop window moves, 1.67 s → 2.3 s. The 45 s window is the
  point: ~990 waypoints against the ~600 a demo needs (**65% headroom**), and it survives
  the live rate dropping to 15 Hz. 30 s was only ~11% headroom before real-camera
  latency. See `eval.md` for the full rationale, the one unscored 10 Hz dry trial that
  precedes the set, and why you stop each episode by hand once the outcome is decided.
- **`MAX_EPISODE_SECONDS = 90`** — the hard cap, raised from 30 so a 10 Hz fallback run
  (~60 s of playback) still fits. It is headroom, not a target; do not run at the cap.
- **`--policy-path`** — the checkpoint dir. Empty/unset is a hard error, never a silent
  base-model run.

Then the live trials (operator + arm), per `eval.md`. Every trial row in `trials.csv`
carries its `clamp_profile` — so a `recorded` run can never be mistaken for a `policy`
one — plus `model_ref` and `model_revision`, so a score can always name the weights that
produced it. The live run's operator-confirmation prompt names the model, the envelope,
the cap and the rate, and says that table contact is intended.

## Safety recap (unchanged from the project's law)

- Observe-only is the default; `--live` is the only path the **control loop** sends
  from. (Connecting a real arm parks its goal at the current pose once, on connect —
  no motion; the loop itself sends nothing in observe mode.)
- `ARMANI_DRY_RUN=1` **or** `--no-arm` forces a simulated arm *and* observe-only, so
  a dry-run environment can never drive a real arm without the operator gate.
- No raw action reaches the bus — `clamp` sits in the send path (unit-tested).
- Episodes hard-capped (`--seconds`, a positive finite number ≤90 — raised from 30 for
  the S3 fine-tuned eval, see Part D). Never unattended.
- Kill switch registered for `--live`; ESC/Ctrl-C stops the loop, holds, and offers
  the freeze menu (return-to-start / home / torque-off / leave). Errors return the
  arm to the entry pose.
