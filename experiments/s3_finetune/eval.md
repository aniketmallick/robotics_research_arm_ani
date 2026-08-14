# S3 eval — score the fine-tuned policy on the arm

Reuses the S2 runner: same control loop, same clamp in the send path, same kill switch,
same 0–4 ladder, same per-step JSONL. Four things are deliberately **not** the same as
the S2 baseline (0/2), and each is recorded per trial so the comparison stays honest —
the checkpoint (`--policy-path`), the clamp envelope (`--clamp-profile recorded`), the
playback rate and window (`--hz 30 --seconds 45` vs 10 Hz / 20 s), and the ratified
best-state scoring rule. Treat it as a comparison with stated deltas, not a like-for-like
rerun.

## Setup (`lerobot-vla` env, arm connected)

```bash
PY=~/miniforge3/envs/lerobot-vla/bin/python
export ARMANI_CAMERA_INDEX=0
export ARMANI_FOLLOWER_PORT=/dev/tty.usbmodem<follower>   # changes per plug-in
CKPT=/path/to/checkpoints/last/pretrained_model            # downloaded from Colab
```

**First, headless — confirm the checkpoint loads, uses its OWN stats, and speaks our
convention** (no arm, no camera):

```bash
$PY -m experiments.s2_zero_shot.run_zero_shot --policy-path "$CKPT" \
    --no-arm --synthetic-frame --hz 30 --seconds 45 --clamp-profile recorded
```

Five things to read before going anywhere near the arm:

0. `model  :` / `rev    :` — your checkpoint dir and the commit it was downloaded from.
   That pair is what lands in `trials.csv`, so a score can always name its weights.
1. `[policy] loading <your ckpt dir> (fine-tuned)` — **not** `lerobot/smolvla_base`. If
   `$CKPT` is unset the runner **aborts** (empty `--policy-path` is an error, not a
   silent base-load), so a mistagged base run can't slip through.
2. `cameras filled [observation.images.camera1]` — one key, matching the one-camera
   fine-tune.
3. `stats=checkpoint:<your ckpt dir> (own MEAN_STD stats, no pretrain routing)` — the
   normalization statistics computed over YOUR dataset during training, shipped inside
   the checkpoint. If this said `pretrain:so100`, every action would be denormalized
   against a generic scale and the arm commanded to wrong positions with no error. The
   next line, `action unnormalize MEAN_STD: mean=[…] std=[…]`, prints the actual numbers
   — cross-check them against `check_dataset.py`'s per-joint ranges. A checkpoint that
   does not carry its own stats is a hard refusal, not a warning.
4. `clamp  : armani.safety.clamp_action(recorded)` — the ratified envelope (below).

> `check_stats` always loads the **base** model (it has no `--policy-path`), so it only
> sanity-checks the base's stats — it cannot see your fine-tuned checkpoint. The
> fine-tuned stats are verified solely by the `run_zero_shot --policy-path` lines above.

Expect (fine-tuned): action targets in OUR degree convention (roughly the demo joint
ranges), NOT the so100 servo-degree convention (~120°) the base emitted.

## Playback rate — RATIFIED: score at `--hz 30 --seconds 45`, stop each episode by hand

**Run the scored trials at demo speed.** The demos were recorded at 30 fps, ~600
waypoints each, and the runner executes one waypoint per step at the `--hz` pace. The
window is `--seconds 45`, and **you end the episode yourself once the outcome is
decided** — see "Stop it when it's decided" below. The hard cap stays 90 s as headroom;
do not run at the cap.

Why 30 and not the 10 Hz default:

- **It removes a confound you cannot rule out later.** If the policy fails at 10 Hz,
  "we ran it at a third of training speed" stays a live explanation forever.
- **It restores the observation cadence training assumed.** The policy serves 50
  waypoints per plan (`n_action_steps: 50`), so it looks at the world once per chunk and
  drives open-loop in between: **~1.67 s at 30 Hz** (what training saw) versus ~5 s at
  10 Hz. Worth remembering for Set C, where the scene is less predictable — the block is
  static, so it will not move your Set A/B scores.
- **It is not a safety escalation.** 30 Hz is the exact speed the demos were teleoperated
  at, fifty times. You already know what it looks like.

This is reachable because the loop is **pace-bound, not inference-bound**: median step
cost is 9 ms and inference is only 18% of wall time.

**Measured (2026-08-14, mps, headless, `--hz 30`): ~22 Hz achieved, not 30.** Two costs
eat the difference: the ~400 ms re-plan every 50th step and ~30 ms/step of non-inference
work (frame build, per-step JSONL flush). This is accepted and **must be stated in the
results as ~0.73× training speed** — not "demo speed".

Why 0.73× is scientifically fine: the policy is position-conditioned with
`n_obs_steps: 1`, so per-waypoint behaviour is identical at any playback rate. The only
real difference is the open-loop window between plans — **2.3 s at 22 Hz versus 1.67 s at
30 Hz** — which is marginal against a static block. (At the 10 Hz default it was ~5 s,
which is why 10 Hz is the fallback and not the protocol.)

**The real risk is margin, not speed, and the fix is the flag — 45 s, not 30.** At 22 Hz
a 45 s window is ~990 waypoints against the ~600 a demo needs: **65% headroom**, still
well under the 90 s cap, and it survives the live rate dropping as low as 15 Hz
(15 × 45 = 675 > 600). A 30 s window was only ~11% headroom on synthetic frames, and live
camera reads eat into that — `stream.read_bgr()` on the C920 can add up to a frame period
per step. Do not let the truncation `[warn]` discover this on trial one; a truncated
first trial costs a reset and adds noise for nothing.

The report prints the achieved Hz every run, and a fine-tuned episode that ends under
~600 waypoints warns that the window may have closed before the trajectory did. If you
ever see that warn on an episode you did **not** stop yourself, raise `--seconds` and
re-run — do not score a truncated episode as a policy failure.

### Stop it when it's decided

**End the episode by hand as soon as the outcome is settled.** If it grasps and lifts at
22 s, stop there; otherwise you hand a policy that already succeeded another 23 seconds
to wander off and drop the block. Same for a clear failure — no value in watching it
flail to the cap.

ESC (or Ctrl-C) **freezes**: the loop stops commanding, the arm holds where it is —
still holding the block, if it has one — and you get the freeze menu (return to start /
home / torque-off / leave). Nothing auto-drives. The scoring prompt comes after the menu.
Operator-stopped episodes are already exempt from the truncation warn, so a deliberate
early stop reads as a deliberate early stop, not as a truncated run.

```bash
# ONE unscored dry trial at the 10 Hz default first — purely to calibrate your hand on
# ESC at the slower speed. Do not score it, and stop it with ESC well before the window
# closes (that is the drill), so it never reaches the truncation warn.
$PY -m experiments.s2_zero_shot.run_zero_shot --policy-path "$CKPT" \
   --live --hz 10 --seconds 30 --episode-tag s3_dryrun --clamp-profile recorded
```

Then switch to 30 Hz for every scored trial — **and do not mix rates within a set.**
`trials.csv` records `hz` and `elapsed_s` per row, so a mixed set is at least detectable
afterwards (and `n_steps / elapsed_s` gives the rate actually achieved) — but a set with
two rates answers a muddier question than the one you set out to ask.

> Fallback if 30 Hz looks wrong live (jerky, or the re-plan stalls read as stutter):
> drop back to `--hz 10 --seconds 90` and say so in the results. That is still a valid
> trajectory — the policy has `n_obs_steps: 1` (a single frame + single state, no
> history, no velocity term), so replaying slower preserves the waypoint sequence
> exactly; only wall-clock changes. But it reopens the speed confound, so it is a
> fallback, not a preference.

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

> **RATIFIED scoring rule — score the BEST state the arm reached during the episode, not
> the state it ended in.** Decided before the first trial and held for every scored trial
> in all three sets. A policy that grasps and lifts at 22 s and then wanders off and drops
> the block scored a 4; post-success wandering does not erase a success. This is also why
> you stop the episode once the outcome is decided — the two rules work together, and
> without the first one the second would quietly cost you successes.

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
   --live --hz 30 --seconds 45 --clamp-profile recorded \
   --task "Pick up the red block" --episode-tag s3A_trial_1 --trial
# Set B: block INSIDE the trained region but between demo spots (interpolation), tag s3B_interp_1 ..
# Set C: swap in a different object (e.g. a marker), tag s3C_object_1 ..
```

`--clamp-profile recorded` belongs on EVERY fine-tuned trial here because `check_dataset`
flagged the demos beyond policy ±60° (see the RATIFIED block above) — operator present,
kill switch armed. Drop it only if a future dataset stays inside ±60°.

`--trial` prompts a 0–4 score after each and appends to
`experiments/s2_zero_shot/trials.csv`, which is now self-describing: every row carries
`clamp_profile` (a `recorded` trial can never be read back as a `policy` one) and
`model_ref` + `model_revision` (which weights produced the score). The two closed S2 rows
are backfilled `policy` / `lerobot/smolvla_base`; their revision is deliberately blank,
because the cache can say what `main` points at today, not what ran on 2026-07-24.
Per-step raw-vs-clamped actions land in `logs/episode_s3*.jsonl`, each step tagged with
its `clamp_profile`, and the episode summary repeats the model identity. Report success
**rate** per set (e.g. "A: 6/10 reached ≥3, mean 2.4").

**Keep the hand on ESC for the whole run.** Unlike S2 — which snapped to one clamped pose
— this policy reaches, descends, and closes on the block at teleop speed.

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
