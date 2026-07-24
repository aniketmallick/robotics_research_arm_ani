# S3 eval — score the fine-tuned policy on the arm

Reuses the S2 runner unchanged except the checkpoint swap (`--policy-path`). The
policy clamp, kill switch, episode caps, 0–4 scoring, and per-step JSONL are
identical to S2 — so the eval is directly comparable to the zero-shot baseline (0/2).

## Setup (`lerobot-vla` env, arm connected)

```bash
PY=~/miniforge3/envs/lerobot-vla/bin/python
export ARMANI_CAMERA_INDEX=0
export ARMANI_FOLLOWER_PORT=/dev/tty.usbmodem<follower>   # changes per plug-in
CKPT=/path/to/checkpoints/last/pretrained_model            # downloaded from Colab
```

**First, headless — confirm the checkpoint loads and speaks our convention** (no arm):

```bash
$PY -m experiments.s2_zero_shot.run_zero_shot --policy-path "$CKPT" --no-arm --seconds 10
```
The runner prints the loaded model_ref and the camera keys it fills — confirm it says
your checkpoint dir (not `lerobot/smolvla_base`) and `cameras filled
[observation.images.camera1]` (one key, matching the one-camera fine-tune). If `$CKPT`
is unset the runner now **aborts** (empty `--policy-path` is an error, not a silent
base-load), so a mistagged base run can't slip through.

> `check_stats` always loads the **base** model (it has no `--policy-path`), so it only
> sanity-checks the base's stats — it cannot see your fine-tuned checkpoint. The
> fine-tuned stats are verified solely by the `run_zero_shot --policy-path` line above.

Expect (fine-tuned): action targets in OUR degree convention (roughly the demo joint
ranges), NOT the so100 servo-degree convention (~120°) the base emitted.

**Read the clamp-bit rate as a diagnosis, not just a pass/fail.** Eval clamps with the
`policy` profile (shoulder_lift/elbow_flex/wrist_flex = ±60°), but the demos were
teleop-recorded in the wider `recorded` profile and may legitimately reach beyond ±60°
to touch the table (the S1 geometry finding: near-vertical table reach is marginal
within ±60°). A faithfully-cloned policy predicts those same targets, so:
- **Low clamp-bit rate** → the learned grasp lives inside the policy envelope. Good.
- **High rate, concentrated on shoulder_lift/elbow_flex/wrist_flex** → the demonstrated
  reach exceeds the ±60° envelope. That is an **envelope/geometry problem, NOT a bad
  checkpoint** — the arm physically can't complete the pick under the policy clamp no
  matter how well training converged. Reconcile before trusting the eval: keep demos
  inside the envelope (SOP), add a riser, or eval a wider profile as a deliberate,
  reviewer-approved exception. Do **not** misread it as "wrong checkpoint or stats."

## Scoring ladder (same as S2 — strict, decided up front)

`0` no purposeful motion · `1` moved toward the object (~5 cm) · `2` touched it ·
`3` grasped it · `4` lifted + completed. A trained policy is stochastic, so we want a
**success RATE**, not one shot.

## Trials — reset (object + arm to rest) between every one

| set | positions | n | what it measures |
|---|---|---|---|
| **A — trained** | inside the training region, near demo spots | ~10 | did it learn the task? |
| **B — interpolation** | INSIDE the trained 20×15 region, but BETWEEN demonstrated grid spots | ~5 | does it generalize in-distribution (interpolate between demos)? |
| **C — novel object** | a different object, trained region | ~3 | specificity — **expect failure** (it learned "red block", not "grasp") |

Set B is *interpolation* — positions inside the trained region that fall between the
grid spots you demonstrated. It is NOT the area outside the region (that is
out-of-distribution extrapolation, a different and harder claim; run it as a separate
labeled set if you want it, don't fold it into the in-distribution number).

```bash
# Set A (repeat for _1 .. _10, reposition the block in the trained region each time):
$PY -m experiments.s2_zero_shot.run_zero_shot --policy-path "$CKPT" \
   --live --seconds 20 --task "Pick up the red block" --episode-tag s3A_trial_1 --trial
# Set B: block INSIDE the trained region but between demo spots (interpolation), tag s3B_interp_1 ..
# Set C: swap in a different object (e.g. a marker), tag s3C_object_1 ..
```

`--trial` prompts a 0–4 score after each and appends to
`experiments/s2_zero_shot/trials.csv`; per-step raw-vs-clamped actions land in
`logs/episode_s3*.jsonl`. Report success **rate** per set (e.g. "A: 6/10 reached ≥3,
mean 2.4").

## Safety — this policy actually reaches and grasps

Unlike the S2 baseline (which snapped to one clamped pose), a *working* fine-tuned
policy makes real reach → descend → grasp motions, and **table contact is now
intended** for the grasp. It is still clamped to the policy envelope, speed-bounded
per send, and kill-switched — but it moves purposefully toward the table.

- Clear everything except the target from the workspace.
- Grant Input Monitoring so **ESC** works as the kill switch (or keep a hand on the
  DC power switch); Ctrl-C also freezes.
- Operator present, hand near the kill, every trial. Start with Set A's easiest
  (centre) position to confirm sane behavior before the spread.
- If it drives hard into the table or off the workspace, kill it — a bad checkpoint
  is still bounded by the clamp, but you don't need to watch it grind.
