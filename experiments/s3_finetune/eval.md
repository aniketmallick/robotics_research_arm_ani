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

**Clamp profile is decided from the data, before eval — measure first.** Eval clamps
with the `policy` profile (shoulder_lift/elbow_flex/wrist_flex = ±60°) **by default**,
but the demos were teleop-recorded in the wider `recorded` profile and a table-reaching
grasp very likely exceeds ±60° (the S1 geometry finding). A policy trained on
recorded-envelope demos and then clamped *tighter* than those demos gets strangled at
the grasp and produces a **false zero** — the clamp lying, not the policy failing.

So the profile is set from `check_dataset.py`'s per-joint action range, not guessed at
eval:
- **Demos within ±60°** (checker says "within the policy envelope") → keep the default
  `policy` clamp; a high clamp-bit rate then really would mean a bad checkpoint/stats.
- **Demos exceed ±60°** (checker flags shoulder_lift/elbow_flex/wrist_flex) → **send the
  ranges to the architect.** The recorded profile is already a ratified safe bound and a
  fine-tuned policy is far closer to "replaying teleop" (recorded) than to "untrusted
  LLM JSON" (policy). The architect ratifies the `recorded` profile for the **fine-tuned
  eval ONLY**, wired as `--clamp-profile recorded` (see the ratification below). A riser
  under the block stays as the fallback if the recorded envelope feels too loose live.

Do **not** misread a geometry-driven high clamp-bit rate as "wrong checkpoint or stats."

> **RATIFIED (architect), S3 fine-tuned-eval ONLY:** if `check_dataset` flags any body
> joint beyond policy ±60°, the operator is authorized to eval the fine-tuned checkpoint
> with `--clamp-profile recorded`. Mandatory: operator present, kill switch armed, hand
> on power. NEVER the demo pipeline, NEVER the base/S2 baseline, NEVER unattended.
> Rationale: clamping a policy tighter than its training data guarantees a false-negative
> grasp; recorded is an already-ratified safe bound. Authorization expires when S3 closes.

The runner enforces the scope structurally: `--clamp-profile recorded` is **refused on
the base model** (it requires a fine-tuned `--policy-path`), so the closed S2 baseline
can never be re-measured under a wider envelope. Default stays `policy` — omit the flag
for the normal case.

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

# If check_dataset flagged the demos beyond policy ±60° (see the RATIFIED block above),
# add --clamp-profile recorded to EVERY fine-tuned trial (operator + kill switch armed):
#   $PY -m ...run_zero_shot --policy-path "$CKPT" --clamp-profile recorded --live ...
```

`--trial` prompts a 0–4 score after each and appends to
`experiments/s2_zero_shot/trials.csv`; per-step raw-vs-clamped actions land in
`logs/episode_s3*.jsonl`. Report success **rate** per set (e.g. "A: 6/10 reached ≥3,
mean 2.4").

## Safety — this policy actually reaches and grasps

Unlike the S2 baseline (which snapped to one clamped pose), a *working* fine-tuned
policy makes real reach → descend → grasp motions, and **table contact is now
intended** for the grasp. It is still clamped to the ratified envelope (`policy` by
default, or `recorded` per the ratification above), speed-bounded per send, and
kill-switched — but it moves purposefully toward the table.

- Clear everything except the target from the workspace.
- Grant Input Monitoring so **ESC** works as the kill switch (or keep a hand on the
  DC power switch); Ctrl-C also freezes.
- Operator present, hand near the kill, every trial. Start with Set A's easiest
  (centre) position to confirm sane behavior before the spread.
- If it drives hard into the table or off the workspace, kill it — a bad checkpoint
  is still bounded by the clamp, but you don't need to watch it grind.
