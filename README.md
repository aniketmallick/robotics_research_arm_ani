# ARM-ANI — measurement-first robotics on an SO-101

**One desk arm, taken seriously:** teleop demonstrations recorded by hand → SmolVLA fine-tuned on them → a hardened, pre-registered live evaluation → a public verdict — and every failure along the way turned into an automated check. This repo is where the [robot-data audit practice](https://buildsbyaniket.com) was born: each audit check we run commercially was earned from a measured failure here first.

## Headline result — Spike S3 (Aug 2026)

Fine-tuned SmolVLA on **50 hand-recorded pick-and-place episodes**, evaluated over **21 pre-registered live trials** against a frozen placement protocol:

| | base model | fine-tuned |
|---|---|---|
| full completions | 0/2 (baseline aborted for safety on remaining sets) | **5/18 (28%)** |
| grasp or better | 0 | **8/18 (44%)** |
| contact or better | 0 | **14/18 (78%)** |

**The finding that mattered:** the policy's failure zones were predictable from the training corpus's joint-coverage map *before the robot ever moved* — coverage predicts capability. Full write-up, deviations, and honest limitations: [`experiments/s2_zero_shot/S3_VERDICT.md`](experiments/s2_zero_shot/S3_VERDICT.md) · raw per-trial provenance: [`trials.csv`](experiments/s2_zero_shot/trials.csv) · model + dataset: [huggingface.co/anikmall](https://huggingface.co/anikmall)

Other findings: emergent recovery via ~2s replanning; object-affordance transfer (shape, not color); control rate is causal (10 Hz: 0/3 · 30 Hz: grasps); declared-vs-observed action envelopes diverge (gripper never exceeded 63% of its declared range).

## Repo map

- `experiments/s2_zero_shot/` — the eval harness (`run_zero_shot.py`): clamp profiles, 45s windows, best-state scoring, per-trial provenance rows, observed-envelope guard. Plus `S3_VERDICT.md` and `trials.csv`.
- `experiments/s3_finetune/` — the fine-tune recipe: recording SOP, dataset checks, Colab notebook, `s3_config.py`.
- `armani/` — the earlier voice-interactive agent (stages 1–3): LLM task planning with **trust gates in Python, not prompts** — every risky action routes through code-level safety gates; the model never touches the motor path.
- `docs/` — spike results S1–S3, runbooks, environment report.
- `prompts/` — the staged build prompts (this project was built spec-first, in reviewable stages).

## Corrections (append-only, dated)

**2026-09-04 — `freeze_vision_encoder`:** earlier versions of `experiments/s3_finetune/` (notebook + README) said `--policy.freeze_vision_encoder=false`. The shipped S3 checkpoint's own `train_config.json` records **`true`** — the encoder was frozen. Corrected in place. Kept here deliberately: a committed training file contradicting its shipped artifact is exactly the defect class our audit now checks for — it happened to us first, and we found it by diffing code against checkpoint.

## Design principles

Trust lives in code, not in the model. Nothing moves unless the operator is present. Kill-switch freezes and asks — it never auto-drives. Bars are pre-registered before results exist; failures publish with the same prominence as passes.

## Current work

The next layer is in progress: a **certified deployment pipeline** — audited RL training environment, frozen pre-registered evals, measured sim-to-real gap, runtime monitors, and a signed certification report, on this same arm. Follow along: [buildsbyaniket.com](https://buildsbyaniket.com) · [x.com/buildsbyaniket](https://x.com/buildsbyaniket)