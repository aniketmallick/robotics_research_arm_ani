# Spike S3 — fine-tune SmolVLA on one pick (results)

**Question:** does fine-tuning `smolvla_base` on ~50 in-domain teleop demos of ONE
task ("Pick up the red block", pick only) turn the S2 zero into real, repeatable
success on THIS rig?

**Baseline to beat (S2, measured):** zero-shot SmolVLA-base = **0/2**, scene-blind,
regresses to its pretraining mean pose in the so100 servo-degree convention, 100 %
clamp-bite (body joints saturated). No transfer. **Any** task-directed behavior here
is an improvement; the bar for "S3 worked" is a real, repeatable pick success rate.

## Method (built in `experiments/s3_finetune/`)

- **Data:** ~50 clean teleop demos, ONE red block, fixed approach + wrist, varied
  only (x, y). Recorded with the C920 (`record_picks.py`) so the dataset has images
  SmolVLA needs. Rules: `SOP.md`. Sanity: `check_dataset.py`.
- **Fine-tune:** `smolvla_base` via `lerobot-train` (`finetune_smolvla.ipynb`, Colab).
  Vision encoder unfrozen. Base model `lerobot/smolvla_base`.
- **Eval:** the S2 runner with `--policy-path <checkpoint>`; clamp + kill switch +
  caps + 0–4 scoring unchanged, so it's directly comparable to the S2 zero. Protocol:
  `eval.md`.

## Dataset (fill after recording)

| quantity | value |
|---|---|
| episodes (clean) | **TODO** (~50) |
| codebase_version | **TODO** (expect v3.0) — Colab lerobot must match |
| camera key(s) | **TODO** (expect `observation.images.camera1` — named to match smolvla_base) |
| length min/median/max (frames) | **TODO** |
| per-joint action range | **TODO** (`check_dataset.py`; sent to the architect for the clamp-profile call) |
| eval clamp profile (ratified) | **TODO** (`policy` ±60° default, or `recorded` if the demos exceeded it) |
| Hub repo | **TODO** (`anikmall/armani_pick_red_v1`, private) |

## Training config (fill after the run — REQUIRED for reproducibility)

| quantity | value |
|---|---|
| lerobot commit + version | **TODO** (target `58ccc015…`, 0.5.2 — must read v3.0 + save 0.5.2-loadable) |
| device / GPU | **TODO** (Colab A100 / T4 / Mac-MPS) |
| batch_size / steps / scheduler_decay_steps | **TODO** |
| freeze_vision_encoder | **TODO** (expect false) |
| wall-clock | **TODO** |
| final train loss / plateau | **TODO** |

## Results — eval on the arm (fill after eval)

Headless check first: fine-tuned targets in OUR degree convention, clamp-bit rate
**TODO %**.

| set | positions | n | reached ≥3 (grasp) | mean 0–4 | notes |
|---|---|---|---|---|---|
| A — trained | demo region | ~10 | **TODO / 10** | **TODO** | |
| B — interpolation | inside trained region, between demo spots | ~5 | **TODO / 5** | **TODO** | in-distribution generalization |
| C — novel object | different object | ~3 | **TODO / 3** | **TODO** | specificity — expect failure |

- **Clamp-bit rate (live):** **TODO %** (S2 was 100 %). Read it as a diagnosis: a low
  rate means the learned grasp lives inside the ±60° `policy` envelope; a high rate
  concentrated on shoulder_lift/elbow_flex/wrist_flex means the demos exceeded ±60°
  (envelope/geometry problem, per the S1 verdict — not a bad checkpoint). Eval clamps
  the `policy` profile while demos were recorded in the wider `recorded` profile; note
  which diagnosis applies here.
- **Evidence:** `experiments/s2_zero_shot/{trials.csv, logs/episode_s3*.jsonl}`.

### Ratified exception — eval clamp profile

> **RATIFIED (architect), S3 fine-tuned-eval ONLY:** if `check_dataset` flags any body
> joint beyond policy ±60°, the operator is authorized to eval the fine-tuned checkpoint
> with `--clamp-profile recorded`. Mandatory: operator present, kill switch armed, hand
> on power. NEVER the demo pipeline, NEVER the base/S2 baseline, NEVER unattended.
> Rationale: clamping a policy tighter than its training data guarantees a false-negative
> grasp; recorded is an already-ratified safe bound. Authorization expires when S3 closes.

Enforced structurally: `--clamp-profile recorded` is refused on the base model (S2
baseline stays untouchable). Which profile the eval actually used is in each episode's
`clamp_source` (e.g. `armani.safety.clamp_action(recorded)`).

## Verdict (fill after eval)

**TODO.** The pivotal question — did learned manipulation clear the bar? Skeleton to
complete with numbers: fine-tuned SmolVLA scored **TODO** on trained positions vs the
**0/2** zero-shot baseline, so fine-tuning on ~50 in-domain demos **[did / did not]**
turn the zero into repeatable success. Generalization to novel positions: **TODO**.
This **[validates / does not validate]** learned manipulation as viable for this rig,
and points S-next at **TODO** (more data / more tasks / depth for tall objects, per
the S1 verdict). The safety layer contained the policy throughout (clamped, bounded,
kill-switched) — as it did the S2 black box.
