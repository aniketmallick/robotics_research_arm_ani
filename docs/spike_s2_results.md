# Spike S2 — zero-shot SmolVLA baseline

**Question:** what does an untuned generalist VLA (`lerobot/smolvla_base`, 450M)
actually do on this table, this camera, this arm — over 10 scored trials?

**Hypothesis (to be confirmed):** near-total failure. That failure, documented
with numbers and video, IS the deliverable — the baseline Spike S3 (fine-tuning)
must beat, and evidence for interviews. We are measuring it, not fixing it.

## Eval protocol (fixed BEFORE trial 1)

- **Task:** "Pick up the red block". One red block at the arm's mid-workspace,
  same spot each trial (±~2 cm). Arm returned to rest between trials.
- **10 trials**, fresh reset each time.
- **Score ladder (strict, decided up front):**
  - `0` no purposeful motion
  - `1` moved toward the object (gripper ends within ~5 cm of it)
  - `2` touched it
  - `3` grasped it (jaws closed on it, held briefly)
  - `4` lifted + completed (off the table, still held)
- **Alternate phrasings:** 2 trials each of "Grab the red cube" and "Pick up the
  block and lift it" — note any behavior difference vs the primary phrasing.
- **Evidence:** keep C920 frames or a phone video for ≥3 representative trials.
- **Every step logged:** raw vs clamped action → `experiments/s2_zero_shot/logs/
  episode_*.jsonl`; scores → `experiments/s2_zero_shot/trials.csv`.

Runner details in `experiments/s2_zero_shot/README.md`. Observe-only is the
default; `--live` requires operator presence + kill switch; every action is
clamped to the policy envelope before any send (unit-tested).

## Environment (headless, verified by me)

Parallel conda env `lerobot-vla` (ratified exception). The demo `lerobot` env got
**zero** new packages (proven by a `pip freeze` diff against a pre-spike snapshot).
Full versions + the generated latency table: `experiments/s2_zero_shot/env_report.md`.

- lerobot **0.5.2** (installed from the SAME local source the demo uses — PyPI has
  no 0.5.2 — so calibration/dataset formats are identical), torch 2.11.0 (MPS),
  transformers 5.5.4, numpy 2.2.6, opencv-headless 4.13, accelerate 1.14.
- `python-dotenv` + `pytest` added to `lerobot-vla` only, so the runner reuses the
  real `armani.safety.clamp_action`.

## Checkpoint feature spec (discovered, not assumed)

`lerobot/smolvla_base` expects/produces:

- **state_dim 6** — matches our 6 SO-101 joints.
- **action_dim 6** — mapped **positionally** onto our JOINTS order (shoulder_pan,
  shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper). This is an OOD
  assumption: the base model's joint order/sign/units are its training arm's, not
  ours. The action is unnormalized with the checkpoint's OWN stats (see
  *Measurement integrity* below — those stats are keyed per pretraining dataset and
  had to be routed), so targets land in the base model's convention, not our
  calibration. The policy clamp is what keeps that safe.
- **three cameras** (`camera1/2/3`). The checkpoint *declares* a 3×256×256 input
  shape per camera, but SmolVLA resizes-and-pads every image to 512×512 before the
  encoder, so the declared size is not load-bearing. We have one C920, so the same
  frame is fed to all three keys (further OOD; documented).
- **chunk_size / n_action_steps 50** — one model inference yields 50 steps; the
  runner pops one per control tick.

## Part A — load + latency (M1 Max, synthetic observation) — DONE, headless

| quantity | MPS | CPU |
|---|---|---|
| model construct + load | **25.2 s** | not benched here (an earlier CPU-only load was *observed* at ~225 s) |
| chunk inference — one-shot | **565 ms** | 9 841 ms |
| chunk inference — steady mean | **503 ms** | 10 712 ms |
| amortized per control step (÷50) | **10.1 ms** | 214 ms |
| sample-action absmax (unnormalized) | ≈ so100 mean (~120–130) | ~120 |

Full table + versions: `experiments/s2_zero_shot/env_report.md`. (The sample-action
magnitudes are post-fix — see *Measurement integrity* below; pre-fix they were
~0.8, which was the un-normalization bug, not the model.)

**Read of the numbers:** MPS supports a ~10 Hz control loop comfortably (the 503 ms
chunk inference amortizes over 50 popped steps to ~10 ms/step, with a visible
~0.5 s hitch each time the chunk refills). CPU is ~21× slower — a ~10 s stall per
chunk — so **not** real-time viable; MPS is the only sane device here.

**Compute-tier verdict (banked):** 503 ms/chunk on MPS with 50-step chunks = a
SmolVLA-class System-1 policy runs **in real time on the M1 Max alone** — no
procurement needed for this tier, exactly as projected. **MPS is load-bearing:**
CPU-only at 10.7 s/chunk is dead, so the answer depends on Metal being available.

## Measurement integrity (Q1) — the un-normalization bug we caught and fixed

Before trusting anything as a "zero-shot result," we verified the harness actually
maps the policy's output back to joint space. **At first it did not — and the fix
changes the story.** (This check was the reviewer's blocker; it was the right call.)

`smolvla_base` is **multi-embodiment**: its normalize/unnormalize stats are keyed
per pretraining dataset (`so100.buffer.action`, `so100-red.buffer.action`,
`so100-blue.buffer.action`), never the bare `action` key our runner looks up.
lerobot's normalizer **silently returns the tensor unchanged** when the key is
missing (no error), so the first pass fed the model's **raw normalized output**
(|·| ≤ ~1) to the arm as if it were joint degrees. That "small, normalized-looking"
signal was OUR bug wearing a zero-shot costume — exactly what the review flagged.

**Fix (`smolvla_io.route_dataset_stats`):** select one pretraining dataset's stats
(`so100`, overridable via `ARMANI_SMOLVLA_STATS_DATASET`) and alias them onto the
bare feature keys, for BOTH the input state-normalizer and the output
action-unnormalizer. Verified by `check_stats.py`: action stats now load
(mean `[1.6, 120, 110, 57, -27, 12]`, std `[26, 52, 50, 37, 59, 19]`) and the
unnormalize is genuine. Covered by two headless unit tests.

(Note: this checkpoint's shared stats file carries **action stats only**, so input
**state normalization is also a no-op** — state is fed raw. It doesn't change the
result: the model ignores the scene regardless, and only the action path reaches the
motors. `check_stats.py` reports this explicitly.)

**What the fixed model actually does** (headless, synthetic AND the real
`logs/last_frame.jpg` table image): the **normalized** output sits near 0 on every
joint (|·| ≤ ~0.3, and ≤ 0.17 on the real image) — i.e. **the model regresses to
its pretraining mean pose and essentially ignores the scene.** Unnormalized, that
mean is in the **so100 servo-degree convention** (shoulder_lift ~120°, elbow ~110°,
wrist_roll ~−27°), a *different convention* from our SO-101 centred degrees
(±60/90). So the honest baseline is a **cross-convention, regress-to-mean** failure:
one fixed foreign-convention pose, scene-invariant.

**Live-safety consequence:** those ~110–200° targets sit far outside our envelope,
so the policy clamp **saturates** (shoulder_lift/elbow → 60°, wrist_roll → 150°).
The arm will drive *decisively* to one fixed clamped pose and hold — bounded and
speed-limited by the safety stack, but a real, large motion. Operator: expect a
snap-to-one-pose, not gentle drift, on `--live`.

## Part B/C — the trials (NEEDS THE OPERATOR + ARM)

Primary task "Pick up the red block", 10 trials:

| trial | score 0–4 | clamp-bit rate | notes (what it actually did) |
|---|---|---|---|
| 1 | **TODO** | TODO | |
| 2 | **TODO** | TODO | |
| 3 | **TODO** | TODO | |
| 4 | **TODO** | TODO | |
| 5 | **TODO** | TODO | |
| 6 | **TODO** | TODO | |
| 7 | **TODO** | TODO | |
| 8 | **TODO** | TODO | |
| 9 | **TODO** | TODO | |
| 10 | **TODO** | TODO | |
| **mean** | **TODO / 4** | **TODO %** | |

Alternate phrasings:

| phrasing | trial a | trial b | difference noted |
|---|---|---|---|
| "Grab the red cube" | TODO | TODO | |
| "Pick up the block and lift it" | TODO | TODO | |

- **Clamp-hit rate (live):** TODO % of steps had ≥1 joint clamped. Worst joints: TODO.
- **Headless preview (synthetic frame, observe-only, post-fix, 49 steps):** the clamp
  bit on **49/49 steps (100 %)** — `shoulder_lift` and `elbow_flex` on every step
  (raw ~110–205° → clamped to 60°), `wrist_roll` on every step (raw ~250° → 150°),
  plus `wrist_flex`/`gripper` on most. This is the real signal: the so100-convention
  targets sit far outside our envelope, so the clamp saturates the body joints — the
  arm would drive to one fixed clamped pose. (Pre-fix this was a misleading
  gripper-only 88 %; that was the un-normalization bug.) The live rate on the real
  C920 is the number that counts.
- **Evidence kept:** TODO (paths / video).

## Conclusion (3 sentences — fill after the arm run)

**Predicted from the headless measurement (confirm on hardware):** untuned
`smolvla_base` **regresses to its pretraining mean pose and ignores the object** —
its normalized output is ~0 regardless of the image, so it emits one fixed pose in
the foreign **so100 servo-degree** convention, which the clamp saturates against our
±60/90 envelope. Predicted trial outcome: **score ~0** (no task-directed motion) on
most/all trials; the arm snaps to a fixed clamped pose. That is the honest baseline
Spike S3 (fine-tuning on our teleop data, in our convention) must beat — **any**
task-directed behavior is a measurable improvement over regress-to-mean.

**TODO after the 10 live trials:** confirm the above, fill the mean score, note
whether alt phrasings change anything (unlikely, given scene-invariance), and record
the live clamp-bit **per-joint** breakdown (`per_joint_bit_counts` in the summary
record) — expect it dominated by `shoulder_lift`/`elbow_flex`/`wrist_roll`
saturation, NOT the gripper.
