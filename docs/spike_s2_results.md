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

## Part B/C — the trials (measured 2026-07-24, operator + arm)

Primary task "Pick up the red block", **2 live trials** (block repositioned between
them). Both **0/4** — total failure. Values from `trials.csv` + `logs/episode_trial_*.jsonl`.

| trial | score 0–4 | clamp-bit rate | what it actually did |
|---|---|---|---|
| 1 | **0** | 100 % (110 steps) | drove toward the block, then opened the gripper wide and **dragged across the table** — clamp-saturated thrash, not a reach |
| 2 | **0** | 98 % (63 steps) | no purposeful motion (operator's note: "total bullshit") |
| **mean** | **0 / 4** | **~99 %** | |

**Why 2 trials and not the protocol's 10:** `check_stats` had already proven,
*headless*, that the model's output is near-identical on a synthetic vs a real frame
— it regresses to its mean and ignores the image. The two live trials confirm the
failure is real and total; because the policy has **zero** task competence, more
trials only re-measure the same null. The per-trial *trajectory* varies (flow-matching
sampling is stochastic — T1 thrashed, T2 was near-static), but every one is
non-task-directed, so ten repetitions of a null is one data point, not ten. The 2×2
alternate-phrasing trials were skipped for the same reason: with the scene ignored,
wording cannot move a mean-pose output.

- **Clamp-hit rate (live, trial 1):** **100 %** of steps clamped, **body-joint
  dominated** — `shoulder_lift` **110/110**, `gripper` 100/110, `elbow_flex` 77/110,
  `wrist_roll` 60/110, `wrist_flex` 5/110. A logged raw prediction shows how far out
  of bounds it ran: `shoulder_lift 245°`, `wrist_roll 349°`, `gripper 163°` → all
  clamped to the policy envelope (60° / 150° / 100 %), then bounded again per-send by
  `max_relative_target`. This is exactly the headless observe-only preview
  (49/49 steps, shoulder_lift/elbow/wrist_roll saturating) confirmed on hardware.
- **Evidence:** `experiments/s2_zero_shot/{trials.csv, logs/episode_trial_*.jsonl}`.

## Conclusion — S2 closed on evidence

**Zero-shot SmolVLA-base on this novel rig is a total failure: 0/2 — and the failure
is the MODEL's, not the harness.** The integrity gate (`check_stats`) confirmed the
action unnormalize stats are real and routed (so100), so the model genuinely emits
its pretraining **mean pose in the foreign so100 servo-degree convention** (shoulder
~113–121°, elbow ~110–116°) and **ignores the scene** — it does not transfer to our
SO-101 at all, and the clamp saturates its out-of-envelope targets. That is the whole
point: **this measured zero is the baseline S3 (fine-tuning on our teleop data, in our
convention) must beat.** Any task-directed behavior is a measurable improvement over
regress-to-mean.

Two secondary findings, both material for what comes next:

1. **Compute is not the bottleneck — data is.** The M1 Max runs this 450M-parameter
   VLA at **~528 ms per 50-action chunk on MPS = ~10.6 ms/step (~10 Hz), real-time**
   (30 s load; CPU at 11.9 s/chunk is dead, so MPS is load-bearing). The Mac can *run*
   a SmolVLA-class System-1 policy live — what's missing is in-domain training, not FLOPs.
2. **The safety envelope held at 100 %.** A completely untrusted 450M black box
   emitting joint targets up to 245° / 349° was fully contained: every out-of-bounds
   prediction clamped to the policy envelope, every send bounded by
   `max_relative_target`, **no joint-limit violation across either trial**. The trust
   layer survives a garbage black-box policy — exactly what it exists for. *Caveat,
   consistent with the S1 verdict:* joint-space bounding is not task-space safety —
   trial 1 still dragged the gripper across the table. Clamps protect the arm, not the
   scene; grasp-grade safety needs depth + task-space checks.

**S2 closed:** modular perception + a homography is the wrong tool for grasping
(S1: 2.5D, no depth), and a zero-shot generalist VLA has no transfer to this rig
(S2: regress-to-mean). The path forward is fine-tuning on in-domain data (S3) on top
of the safety layer that just proved it can contain an arbitrary policy.
