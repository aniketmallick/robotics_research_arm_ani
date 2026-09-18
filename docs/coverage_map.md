# Coverage map — does demonstration coverage predict where the S3 policy failed?
**Measured 2026-09-18 · script: `experiments/s3_finetune/coverage_map.py` · inputs: the public S3 dataset (`anikmall/armani_pick_red_v1`), the 18 official eval logs (`experiments/s2_zero_shot/logs/episode_ft_*.jsonl`), `trials.csv`**

![coverage map](coverage_map.png)

## What the figure is

Grey ×: the arm's pose (shoulder_pan, shoulder_lift) at the moment the gripper closed in each of the 50 demonstrations — 45 after dropping five transient closes near the home pose. Coloured markers: the pose at the fine-tuned policy's first grasp attempt in each of the 18 official eval trials, coloured by the 0–4 score. Faint numbers: demonstrations per 5×5 cell.

Two joints stand in for table position (pan ≈ left/right, lift ≈ near/far). It is a per-rig proxy, not a calibrated workspace map.

## What the numbers say

```
median nearest-demo distance:  score 4 → 4.9°  |  score 2–3 → 5.2°  |  score ≤1 → 4.9°
demos in the attempt's cell:   score 4 → [0,1,1,2,3] (median 1.0)
                               score 2–3 → [1,2,2,2,2,2,3,3,4] (median 2.0)
                               score ≤1 → [1,1,2,2] (median 1.5)
```

**At n = 18, with this proxy, coverage does not separate successes from failures.** Two of the four non-grasps (A9, B5) sit in the far band where demonstrations are thinnest — consistent with the operator's note — but so does a clean success (A3), and one success (B4) sits in a cell with zero demonstrations.

## What this changes

`S3_VERDICT.md` finding 1 — *"coverage predicts capability … the deployed policy's failure map is predictable from the training corpus's coverage map"* — was an observation from operator notes, not a measurement. Measured, it does not hold up on this data. Two further corrections to the verdict's supporting sentence:

- The shoulder_pan asymmetry it cites ([−38.9°, +67.7°]) is the *whole-episode* action range and includes the place spot at ≈ +62°. Grasp poses span [−36°, +40°] — close to symmetric. The asymmetry was the place, not the coverage.
- The proxy has a structural bias in the finding's favour that it still failed to show: for under-reaching failures (A9 "2–3 cm short", B1 "7–8 cm short") the attempt pose is where the arm stopped, not where the block was — which pulls failures *toward* covered space. A clean test needs the block's placement as a structured field. It was free text. That is now a requirement of the certification protocol (PR-001).

**What remains true and measured:** the corpus's gaps were visible before the robot moved — gripper commanded to at most 63.0% of its declared 0–100 range ([dataset_report.md](../experiments/s3_finetune/dataset_report.md)); three joints outside the ±60° policy envelope; the far band the thinnest region of the workspace. Whether those gaps *predicted* the failures is, on this data, untested — not confirmed.

The finding is downgraded from "measured" to "observed, untested". Public statements made before 2026-09-18 used the stronger wording.

## Reproduce

```bash
python experiments/s3_finetune/coverage_map.py \
  --root <path to anikmall/armani_pick_red_v1> \
  --logs experiments/s2_zero_shot/logs \
  --trials experiments/s2_zero_shot/trials.csv \
  --out docs/coverage_map.png
```
