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
  ours, and the action is unnormalized with the checkpoint's OWN stats — so
  targets land in the base model's space, not our calibration. The policy clamp is
  what keeps that safe.
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
| sample-action absmax | 0.80 | 0.37 |

Full table + versions: `experiments/s2_zero_shot/env_report.md`.

**Read of the numbers:** MPS supports a ~10 Hz control loop comfortably (the 503 ms
chunk inference amortizes over 50 popped steps to ~10 ms/step, with a visible
~0.5 s hitch each time the chunk refills). CPU is ~21× slower — a ~10 s stall per
chunk — so **not** real-time viable; MPS is the only sane device here.

**The OOD signal (important).** The postprocessed action for a synthetic frame +
zero state is small and varies run-to-run:

- MPS sample: `[0.08, 0.42, 0.22, 0.15, 0.80, -0.31]`
- CPU sample: `[0.375, 0.05, 0.33, -0.22, 0.35, -0.29]`

All components sit in ~[-0.6, 1.7] across steps (see the headless log). **By the
project's own tripwire this already trips "STOP":** the README says if the body
joints look normalized (−1..1) the unnormalization/stats path is suspect, and
absmax ≤ 0.8 with logged raw body-joint values ≤ ~1.7 is exactly that. If read as
**degrees** the arm barely moves; the 6th component maps to our **gripper
(0–100 %)** and comes out negative. It is not *proof* the stats path is broken —
the base model may simply emit near-mean actions on a wildly OOD input — but it is
the #1 thing the operator must eyeball in a **real-camera** observe-only run before
going live. Either way the values are small and the clamp bounds them, so `--live`
is safe.

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
- **Headless preview (synthetic frame, observe-only, 48 steps):** the clamp bit on
  **42/48 steps (88 %)**, *every one on the gripper* — the base model's 6th action
  component is negative (~−0.3 to −0.6), clamped up to 0. The five body-joint
  targets were small enough to stay inside the policy envelope and did not clamp.
  **Read this carefully:** that 88 % is a gripper **unit artifact** (a normalized
  value hitting our 0–100 % floor), NOT body joints leaving the degree envelope —
  do not cite it as evidence of wild degree-space motion. This is a preview only;
  the live rate on the real C920 is the number that counts.
- **Evidence kept:** TODO (paths / video).

## Conclusion (3 sentences — fill after the arm run)

**TODO after the 10 live trials.** Draft skeleton to complete with real numbers:
On this arm/camera the untuned SmolVLA base scored a mean of **TODO/4** across 10
trials (best trial: **TODO**). The dominant failure mode was **TODO** (e.g. no
purposeful motion / motion toward a wrong location / thrashing), which is what a
domain-specific fine-tune (Spike S3) must fix. This establishes the baseline:
**any** S3 result above mean **TODO/4** is a measurable improvement.

> When reporting the clamp-bit rate, split it: the headless preview shows the bite
> is **gripper-only** (a normalized-vs-percent unit artifact), so a high overall
> rate is NOT by itself evidence of large body-joint motion. State the per-joint
> breakdown (`per_joint_bit_counts` in the summary record), not just the headline %.
