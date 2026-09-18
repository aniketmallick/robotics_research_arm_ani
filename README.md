# ARM-ANI — measurement-first robotics on an SO-101

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**One desk arm, taken seriously:** teleop demonstrations recorded by hand → SmolVLA fine-tuned on them → a hardened live evaluation with the protocol fixed before trial 1 → a public verdict — and every failure along the way turned into an automated check. This repo is where the robot-data audit practice was born: each check we now run on other people's data was earned from a measured failure here first (table below).

## Headline result — Spike S3 (Aug 2026)

Fine-tuned SmolVLA on **50 hand-recorded pick-and-place episodes**, evaluated over **18 official live trials** (10 trained positions, 5 novel positions, 3 novel objects) against a placement protocol fixed before trial 1. `trials.csv` holds 24 rows: these 18, the 2 baseline trials, and 4 supplementary rows excluded from the official sets for a pre-protocol tag reuse.

| | base model | fine-tuned |
|---|---|---|
| full completions | 0/2 (baseline aborted for safety on remaining sets) | **5/18 (28%)** |
| grasp or better | 0 | **8/18 (44%)** |
| contact or better | 0 | **14/18 (78%)** |

Full write-up, deviations, and limitations: [`experiments/s2_zero_shot/S3_VERDICT.md`](experiments/s2_zero_shot/S3_VERDICT.md) · raw per-trial provenance: [`trials.csv`](experiments/s2_zero_shot/trials.csv) (the `note` column is the operator's verbatim contemporaneous note at trial time, unedited) · model: [`anikmall/smolvla_pick_red_v1`](https://huggingface.co/anikmall/smolvla_pick_red_v1) · dataset: [`anikmall/armani_pick_red_v1`](https://huggingface.co/anikmall/armani_pick_red_v1) · dataset measurements: [`dataset_report.md`](experiments/s3_finetune/dataset_report.md)

**Coverage map** — [`docs/coverage_map.md`](docs/coverage_map.md): the 50 demonstrations' grasp poses against the 18 eval attempts, by score. The verdict's first finding ("coverage predicts capability") was an operator-note observation; measured on 2026-09-18 it does **not** separate successes from failures at n=18 with the available proxy. Downgraded to *observed, untested* — see Corrections. What *is* measured: the corpus's gaps were visible before the robot moved (gripper commanded to at most 63.0% of its declared range; three joints outside the ±60° policy envelope; far band thinnest).

Other findings (from the verdict): emergent recovery via ~2 s replanning (3 of 5 completions); object-affordance transfer (shape, not colour — the policy ignores the constant instruction string); control rate is causal (10 Hz: 0/3 grasps · ~22 Hz: grasps); declared-vs-observed action envelopes diverge.

## Repo map

- `experiments/s2_zero_shot/` — the eval harness (`run_zero_shot.py`): clamp profiles, 45 s windows, best-state scoring, per-trial provenance rows, observed-envelope guard. Plus `S3_VERDICT.md` and `trials.csv`.
- `experiments/s3_finetune/` — the fine-tune recipe: recording SOP, dataset checks (`check_dataset.py` → `dataset_report.md`), `coverage_map.py`, Colab notebook, `s3_config.py`.
- `armani/` — the earlier voice-interactive agent (July hackathon, stages 1–3): LLM task planning with **trust gates in Python, not prompts** — every risky action routes through code-level safety gates; the model never touches the motor path. `armani/safety.py` (clamp profiles, envelope) is still the send-path guard for every experiment here.
- `docs/` — spike results S1–S3, runbooks, environment report, coverage map. `docs/spike_s3_results.md` is the unfilled pre-eval skeleton, kept as-is.
- `prompts/` — the staged build prompts (this project was built spec-first, in reviewable stages).

## From a failure here to a check in the audit

| What went wrong on this arm | The check it became |
|---|---|
| The training notebook said `freeze_vision_encoder=false`; the shipped checkpoint's `train_config.json` said `true` | Code-vs-artifact diff: every committed training config is compared against the checkpoint's own recorded config |
| `s3_config.py` said the recording was "pick only — no place"; all 50 episodes contained a place | Declared-vs-measured task structure: gripper-event detection over every episode, checked for invariance to the detector's parameters |
| Gripper commanded to at most 63% of its declared 0–100 range; three joints outside the policy envelope | Declared-vs-observed envelope per joint, reported before training and used to set the eval clamp |
| Eval placements recorded as free-text notes → the coverage finding could not be tested cleanly | Placement as a structured, required field per trial (now in the certification protocol) |
| "Stop when decided" drifted to "stop when it looks stuck" (four trials cut at 17–23 s) | Fixed episode windows with a stop-reason enum |
| A verdict's headline finding rested on an observation, not a measurement | Every finding carries its derivation or is labelled *observed, untested* |
| Base model saturated the clamp 100% and still commanded violent motion | Envelope guard on *commanded* targets + operator interlock; base-model live runs banned on this rig |

## Corrections (append-only, dated)

**2026-08-24 — the dataset is pick-AND-place.** `s3_config.py` and the SOP described the recording as pick-only. Measured over all 50 episodes of `armani_pick_red_v1`: every episode contains grasp → sustained closed hold (median 157 frames) → lateral transport (median 59.4° shoulder_pan swing) → release at a consistent spot (shoulder_pan 62.73° ± 1.44). 50/50, invariant to the detector's sustain parameter (5/10/20 frames). The task string "Pick up the red block" is left unchanged because it is baked into the trained checkpoint. Original wording kept struck-through in `s3_config.py`. Documents lie; episodes don't.

**2026-09-04 — `freeze_vision_encoder`.** Earlier versions of `experiments/s3_finetune/` (notebook + README) said `--policy.freeze_vision_encoder=false`. The shipped S3 checkpoint's own `train_config.json` records **`true`** — the encoder was frozen. Corrected in place; the notebook keeps the wrong lines as an erratum. A committed training file contradicting its shipped artifact is exactly the defect class the audit now checks for — it happened to us first, and we found it by diffing code against checkpoint.

**2026-09-18 — "coverage predicts capability" is under-supported.** `S3_VERDICT.md` finding 1 stated that the policy's failures were predictable from the training corpus's coverage map. That was an observation from operator notes. Measured (`coverage_map.py`, [`docs/coverage_map.md`](docs/coverage_map.md)): nearest-demo distance is 4.9° for score-4 trials and 4.9° for score-≤1 trials; demos-in-cell median 1.0 vs 1.5. At n=18 the proxy does not separate outcomes. The verdict's cited shoulder_pan asymmetry included the place spot; grasp poses are near-symmetric. A clean test needs structured placement fields, which S3 did not record. Finding downgraded to *observed, untested*. Public statements before this date used the stronger wording; this README, the verdict (erratum line added), and the site are corrected today.

## Design principles

Trust lives in code, not in the model. Nothing moves unless the operator is present. Kill-switch freezes and asks — it never auto-drives. Bars are set before results exist; failures publish with the same prominence as passes — including failures of our own findings.

## Current work

The next layer is in progress: a **certified deployment pipeline** on this same arm — a publicly pre-registered task and pass/fail bars, training provenance, a frozen evaluation, runtime monitors (envelope guard, out-of-distribution, drift, kill-switch log), and a *Certificate of Conformance to Pre-Registration* that states what is not covered first. Results publish on the pre-registered date, pass or fail. Follow along: [buildsbyaniket.com](https://buildsbyaniket.com) · [x.com/buildsbyaniket](https://x.com/buildsbyaniket)

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
