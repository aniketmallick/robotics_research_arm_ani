# S3 dataset report — `anikmall/armani_pick_red_v1`
**Generated 2026-09-18 by the repo's own check: `python experiments/s3_finetune/check_dataset.py --root <dataset>`. Verbatim output.**

This is the derivation behind "the gripper never exceeded 63% of its declared range": the declared gripper range is 0–100 %; the demonstrations command it to at most **63.0 %**. Three body joints exceed the ±60° `policy` clamp envelope, which is why the fine-tuned eval was ratified on the wider `recorded` profile (see `S3_VERDICT.md`).

```
dataset: anikmall/armani_pick_red_v1 (local copy)
  codebase_version : v3.0   (match the Colab lerobot to this)
  episodes / frames: 50 / 29991   fps: 30
  cameras          : ['observation.images.camera1']
  action / state   : [6] / [6]
  episode lengths  : min 598 / median 600 / max 600 frames

  action range / joint (degrees; gripper is %):
    shoulder_pan   [   -38.9,     67.7]
    shoulder_lift  [  -111.2,     71.3]
    elbow_flex     [   -73.1,     97.2]
    wrist_flex     [    40.1,     96.6]
    wrist_roll     [   -16.2,      8.3]
    gripper        [     0.4,     63.0]
  ⚠ demos EXCEED the policy envelope (±60° on lift/elbow/wrist_flex) on:
      shoulder_lift: demos [-111.2, 71.3] vs policy [-60, 60]
      elbow_flex: demos [-73.1, 97.2] vs policy [-60, 60]
      wrist_flex: demos [40.1, 96.6] vs policy [-60, 60]
  → SEND these ranges to the architect. Eval clamps `policy` by default; if the
    demos need more, the architect ratifies `recorded` for the fine-tuned eval
    ONLY (operator-present + kill-switch). Do NOT distort the grasp to fit policy —
    record the natural pick; the clamp profile is decided from these numbers.

READY: camera present, 6-D state/action, episodes consistent.
  ALWAYS visualize before training (push to the Hub, then):
  https://huggingface.co/spaces/lerobot/visualize_dataset
```
