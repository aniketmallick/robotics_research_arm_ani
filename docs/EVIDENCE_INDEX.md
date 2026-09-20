# Evidence index — every public claim, where it lives, how to check it

For the reader who doesn't take a README's word for it. Each row: the claim as we state it publicly, the artifact that carries it, and a command or a line to look at. If a claim isn't in this table, we haven't earned the right to make it.

| # | Claim | Artifact | How to verify |
|---|---|---|---|
| 1 | Fine-tuned SmolVLA: **5/18 completions (28%), 8/18 grasp-or-better (44%), 14/18 contact-or-better (78%)** | [`experiments/s2_zero_shot/trials.csv`](../experiments/s2_zero_shot/trials.csv) | `python -c "import csv;r=[x for x in csv.DictReader(open('experiments/s2_zero_shot/trials.csv')) if x['episode_tag'] in [f'ft_A{i}' for i in range(2,12)]+[f'ft_B{i}' for i in range(1,6)]+[f'ft_C{i}' for i in range(1,4)]];s=[int(x['score']) for x in r];print(len(s),s.count(4),sum(v>=3 for v in s),sum(v>=2 for v in s))"` → `18 5 8 14` |
| 2 | **Zero-shot baseline 0/2**, clamp bite 100% / 98.4%; the live re-run was **aborted for safety and not scored** | `trials.csv` rows `trial_1`, `trial_2` · [`S3_VERDICT.md`](../experiments/s2_zero_shot/S3_VERDICT.md) §"Set D" | Read the two rows' `clamp_bit_rate` and `note`; the verdict's Set D section states what was and wasn't recorded |
| 3 | Every trial row names its **model revision, clamp profile, control rate, step count** | `trials.csv` header | `head -1 experiments/s2_zero_shot/trials.csv` |
| 4 | The **protocol was fixed before trial 1** (sets A/B/C, n, 0–4 ladder) | [`docs/spike_s3_results.md`](spike_s3_results.md) (pre-eval skeleton, TODO tables unfilled) · [`experiments/s3_finetune/eval.md`](../experiments/s3_finetune/eval.md) | `git log --format='%h %ad %s' --date=short -- docs/spike_s3_results.md` → skeleton committed **2026-07-24** (`49465c6`, `08d154b`, `eee1456`); the eval rows and verdict landed **2026-08-16** (`14fc9be`). The tables were never filled in that file on purpose |
| 5 | **Gripper commanded to at most 63.0% of its declared 0–100 range**; shoulder_lift / elbow_flex / wrist_flex exceed the ±60° policy envelope | [`experiments/s3_finetune/dataset_report.md`](../experiments/s3_finetune/dataset_report.md) | `python experiments/s3_finetune/check_dataset.py --root <local copy of anikmall/armani_pick_red_v1>` — same output, verbatim |
| 6 | The dataset is **pick-AND-place**, not pick-only: **50/50** episodes contain grasp → hold → transport → release; invariant to the detector's sustain parameter (5/10/20 frames) | [`experiments/s3_finetune/s3_config.py`](../experiments/s3_finetune/s3_config.py) lines ~30–45 (measurement + original wording struck through) | Load the parquet, per episode find gripper open→close→open; the release lands at shoulder_pan 62.73° ± 1.44 |
| 7 | The committed notebook said `freeze_vision_encoder=false`; the shipped checkpoint's `train_config.json` says **`true`** | README Corrections 2026-09-04 · [`finetune_smolvla.ipynb`](../experiments/s3_finetune/finetune_smolvla.ipynb) (erratum lines kept) · checkpoint config on the Hub | `grep -o 'freeze_vision_encoder=[a-z]*' experiments/s3_finetune/finetune_smolvla.ipynb` shows both values; the `true` is what the checkpoint carries |
| 8 | **Emergent recovery** — 3 of 5 completions after a deflected object (A10, A11, B4) | `trials.csv` notes for those rows · verdict finding 2 | Read the three notes; per-step logs (`experiments/s2_zero_shot/logs/episode_ft_*.jsonl`, local — not committed, available on request) show the re-aim after replan |
| 9 | **Control rate is causal**: 10 Hz warm-ups 0/3, ~22 Hz grasps | verdict finding 5 | Observed at n=3; graded *observed* in the README, not measured |
| 10 | **"Coverage predicts capability" is not supported** by the S3 data at n=18 (nearest-demo 4.9° vs 4.9°; demos-in-cell median 1.0 vs 1.5) | [`docs/coverage_map.md`](coverage_map.md) · [`coverage_map.py`](../experiments/s3_finetune/coverage_map.py) · [`coverage_map.png`](coverage_map.png) | `python experiments/s3_finetune/coverage_map.py --root <dataset> --logs <logs dir> --trials experiments/s2_zero_shot/trials.csv` reproduces the numbers and the figure |
| 11 | **Trust gates live in Python, not prompts**; every risky action routes through code-level gates; the model never touches the motor path | [`armani/gates.py`](../armani/gates.py), [`armani/pick.py`](../armani/pick.py), [`armani/safety.py`](../armani/safety.py) | `pytest tests/test_gates.py tests/test_safety.py` |
| 12 | Every commanded joint target passes a **clamp**; the `recorded` profile is **refused on the base model**; a checkpoint without its own normalization stats is a **hard refusal** | [`experiments/s2_zero_shot/run_zero_shot.py`](../experiments/s2_zero_shot/run_zero_shot.py) · [`tests/test_s2_zero_shot.py`](../tests/test_s2_zero_shot.py) | `pytest tests/test_s2_zero_shot.py -k "clamp or recorded or stats"` |
| 13 | Corrections are **append-only and dated**; wrong lines are kept struck-through, not deleted | README §Corrections · `s3_config.py` · `S3_VERDICT.md` erratum line | `git log -p -- README.md` — no correction has ever been removed |
| 14 | Model and dataset are **public** | [anikmall/smolvla_pick_red_v1](https://huggingface.co/anikmall/smolvla_pick_red_v1) · [anikmall/armani_pick_red_v1](https://huggingface.co/anikmall/armani_pick_red_v1) | Open the links without logging in |

## What is *not* in this repo, and why

- The per-step eval logs (`experiments/s2_zero_shot/logs/*.jsonl`) — 94 files, gitignored as runtime artefacts; available on request, and the coverage map was computed from them.
- Rows and harness changes from experiments after S3 (a second camera channel; arm-A trials from a separate research programme) — held until that programme closes; they are not part of any claim above.
- Any client data or client findings from the commercial audit practice — confidential until the client consents. The defect classes are demonstrated on our own data (rows 5–7) instead.

## Reading order for a 10-minute deep dive

1. README (3 min) — the loop, the numbers, the evidence grades.
2. [`S3_VERDICT.md`](../experiments/s2_zero_shot/S3_VERDICT.md) (4 min) — protocol, results by set, the aborted baseline, limitations, the erratum.
3. [`docs/coverage_map.md`](coverage_map.md) (2 min) — the headline finding that didn't survive measurement, and what replaced it.
4. `trials.csv` (1 min) — read the notes column. It is the operator, unedited.
