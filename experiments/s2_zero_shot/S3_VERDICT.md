# S3 VERDICT — fine-tuned SmolVLA vs zero-shot baseline
**Spike S3 · closed 2026-08-16 · architect-reviewed**

## Question

Can ~50 clean teleoperated demonstrations of one task teach a 450M-parameter VLA (`lerobot/smolvla_base`) to perform that task on this SO-101, where the untuned model scored 0?

## Answer

**Yes.** Baseline 0/2 with 98–100% clamp saturation → fine-tuned **5/18 full completions (28%), 8/18 grasp-or-better (44%), 14/18 contact-or-better (78%), purposeful approach 18/18. Mean 2.50 on the 0–4 ladder.**

## Protocol

- Model: `smolvla_pick_red_v1` (fine-tune of smolvla_base; 20k steps, batch 64, seed 1000, frozen vision encoder; revision `85eb875e…` in every trial row).
- Data: `armani_pick_red_v1` — 50 clean teleop episodes, 30 fps, one task string ("Pick up the red block"), block placed across a marked 5×5 grid.
- Eval: `--live --hz 30 --seconds 45 --clamp-profile recorded` (ratified: recorded profile for fine-tuned eval only; operator present, kill switch armed). Achieved rate 21–22 Hz (~0.73× training rate) on all rows.
- Scoring: pre-registered 0–4 ladder, best state reached during the episode. Strict throughout.
- Full provenance per row in `trials.csv`: model ref + revision, clamp profile, hz, elapsed, n_steps, clamp bite rate, note with placement.

## Results by set

| Set | Condition | n | Score 4 | ≥3 (grasp) | ≥2 (contact) | Mean |
|---|---|---|---|---|---|---|
| A | Trained (on-dot) positions | 10 | 3 | 5 | 9 | 2.70 |
| B | Novel (between-dot) positions | 5 | 2 | 2 | 3 | 2.40 |
| C | Novel objects | 3 | 0 | 1 | 3 | 2.00 |
| **A+B+C** | | **18** | **5** | **8** | **14** | **2.50** |

**Completion quality split:** of the 5 fours — 2 clean first-approach completions (A3, B3), 3 completions after autonomous recovery from a deflected object (A10, A11, B4).

**Supplementary (excluded from official sets — pre-protocol tag reuse, all tagged ft_A1):** one off-dot clean 4 (additional novel-position evidence), plus 0 / 2 / 3 at other placements. Directionally consistent with the official sets.

## Set D (live baseline re-run): aborted for safety — not fabricated

The base-model live re-run was **aborted by the operator to protect the hardware** after violent commanded excursions; no scores were recorded and none are invented here. The baseline stands on: (1) the closed July live baseline — 0/2, zero grasps, clamp bite 100% / 98.4%; (2) the headless Set D characterization under this exact protocol — 100% of 936 steps clamped, base action mean [1.6, 119.9, 109.8, 56.7, −27.4, 12.0] (so100 servo convention) vs fine-tune [14.1, −40.7, 50.2, 65.4, −3.2, 11.1] (this rig's convention). The base model speaks a different coordinate language; a wider envelope would let it drive further wrong, not closer.

**Standing rule from the abort:** base-model live runs are permanently banned on this rig. The policy clamp bounds positions, not trajectories — an untuned model saturating the clamp can still command violent motion inside it.

## Findings

1. **Coverage predicts capability.** Failures cluster in the far band and corners of the workspace — where demonstrations were thinnest and teleop was least comfortable (shoulder_pan demo range is asymmetric: [−38.9°, +67.7°]). Interior novel positions succeed (B3: clean first-try 4 at a never-trained position); edge positions under-reach by 2–8 cm (A9, B1, B5). The deployed policy's failure map is predictable from the training corpus's coverage map — measured here at row level on one rig.
2. **Emergent recovery via replanning.** The policy replans from a fresh observation every ~50 steps (~2.3 s at achieved rate). After the (round) object deflects on contact, the stale chunk "grasps air"; the next replan re-aims at the object's new position. Every fumble was discarded from training, so recovery was never demonstrated — it emerges from vision-in-the-loop replanning. Three of five completions arrived this way.
3. **Episode windows truncate recovery.** Four Set A trials scored 2 were operator-stopped at 17–23 s, while A11 converged to a 4 using its full 45 s from a worse state. "Stop when decided" drifted toward "stop when it looks stuck." The 28% completion rate is therefore a floor. Future protocol: fixed window; stop only on success or safety; record stop reason as a field.
4. **Object-affordance transfer, language-blindness.** With the task string unchanged, the policy grasped a never-seen tyre (3), touched-and-toppled a thin red scale (2), and hallucinated at red electrical tape (1). The reddest object performed worst: the policy learned "block-shaped graspable mass at a position," not color, and ignores the (constant-in-training) instruction string.
5. **Rate matters; the earlier architect ruling was wrong.** 10 Hz warm-ups: 0/3 grasps with a consistent near-miss signature. At ~22 Hz grasps appear immediately. Position-conditioning preserves the trajectory but not the correction cadence — the final centimeters of a grasp need fresh observations. Closed-loop cadence is part of a policy's competence.

## Limitations

- Single task, single rig, single camera; n=18 official trials; scores by one (strict, consistent) operator.
- Eval at ~0.73× training rate (pace-bound loop). Judged immaterial for outcome given per-replan re-aiming; not proven.
- Placements recorded in free-text notes, not a structured field; four pre-protocol rows carry a reused tag (excluded above).
- Round object inflates "deflection" relative to a cuboid; object physics and policy precision are partially confounded in the 2-scores.
- Checkpoints 010000/015000 not evaluated (kept; comparison is parked follow-up).

## Implications

- **For the ego-video experiment:** the teleop control condition now exists with real numbers, at a difficulty inside the measurable band (28–44% primary metrics — headroom in both directions). Minted-episode conditions get compared against exactly this protocol, this rig, this ladder.
- **For the episode standard / verification harness (v0.1 requirements harvested from this eval's own gaps):** placement as a structured field · fixed episode windows with stop-reason enum · outcome + path-quality split (clean vs recovery) · per-row model ref + revision + control-rate provenance · coverage map of the training corpus as a predictor of deployment failure zones.

## Ledger line (site)

`2026-08-16 — First result: 50 self-recorded demos → 28% full completion, 44% grasp rate, 78% contact on a $200 arm — vs 0 baseline. Three of five completions via emergent recovery. Full protocol + limitations in the repo.`
