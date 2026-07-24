# Spike S3 — fine-tune SmolVLA on ONE pick (beat the zero)

**Question:** does fine-tuning `smolvla_base` on ~50 in-domain teleop demos of ONE
task turn the S2 zero (0/2, scene-blind) into real, repeatable success on THIS rig?

S2 proved zero-shot = total failure (regress-to-mean, no transfer). S3 teaches the
model our rig with our data. This is the pivotal spike.

## Pipeline + envs

| step | env | who |
|---|---|---|
| 1. Record ~50 demos | demo **`lerobot`** (calibrated teleop) | operator + arm |
| 2. Push dataset to HF Hub | demo `lerobot` | operator |
| 3. Fine-tune `smolvla_base` | **Colab GPU** (primary) / `lerobot-vla` on Mac (fallback, slow) | operator + GPU |
| 4. Eval on the arm | **`lerobot-vla`** (the S2 runner, `--policy-path`) | operator + arm |

Every hardware step is operator-present, kill-switched, episode-capped — same law
as S2. `armani.safety.clamp_action` stays in the send path at eval.

> **Set your namespace once.** The dataset repo id defaults to
> `anikmall/armani_pick_red_v1`. **If your Hugging Face username is not `anikmall`,**
> export `ARMANI_S3_REPO_ID=<your-hf-username>/armani_pick_red_v1` **before recording** —
> record, push, and train all read it, so they stay in agreement. Otherwise you push to
> a namespace you don't own (fails) or train points at a dataset that isn't there.

---

## 1. Record the dataset (demo `lerobot` env)

**Read [`SOP.md`](SOP.md) first — consistency is the whole spike.** Then:

```bash
export ARMANI_FOLLOWER_PORT=/dev/tty.usbmodem<follower>   # changes per USB plug-in
export ARMANI_LEADER_PORT=/dev/tty.usbmodem<leader>
export ARMANI_CAMERA_INDEX=0

python experiments/s3_finetune/record_picks.py           # DRY RUN — print the command
python experiments/s3_finetune/record_picks.py --go      # record (→ next, ← redo, ESC finish)
python experiments/s3_finetune/check_dataset.py          # camera? 6-D state/action? outliers?
#   Also prints the per-joint ACTION range. If it flags demos beyond the policy
#   envelope (±60° on lift/elbow/wrist_flex — likely for a table grasp), send those
#   numbers to the architect: they set the eval clamp profile (see eval.md).
```

`record_picks.py` wraps the proven `lerobot-record` teleop flow **plus the C920**
(`--robot.cameras`) — SmolVLA needs images, and the taught-zone datasets had none.
The camera is named **`camera1`** on purpose (dataset key `observation.images.camera1`):
`smolvla_base` declares cameras `camera1/2/3` and fine-tuning keeps that declaration, so
our one camera must be a *subset* of it — `camera1` passes, a friendly name like `front`
is rejected at fine-tune. Dataset id + task are config (`ARMANI_S3_REPO_ID`,
`ARMANI_S3_TASK`=`"Pick up the red block"`), so the harness is reusable for other
objects later.

## 2. Push the dataset to the Hub (private)

```bash
hf auth login          # paste a WRITE token from https://huggingface.co/settings/tokens
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
repo = os.getenv("ARMANI_S3_REPO_ID", "anikmall/armani_pick_red_v1")
root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo}")
LeRobotDataset(repo, root=root).push_to_hub(private=True)   # push_videos=True by default
print("pushed", repo)
PY
```

Then **always visualize** before training:
<https://huggingface.co/spaces/lerobot/visualize_dataset> → paste your repo id.

## 3. Fine-tune (Colab primary)

Open [`finetune_smolvla.ipynb`](finetune_smolvla.ipynb) in Colab (Runtime → GPU;
prefer **A100**, T4 works but slower). It is the official SmolVLA notebook adapted
with our dataset + the version pin below.

**⚠ Version pin (dataset-format compatibility).** Our dataset is LeRobotDataset
**`codebase_version: v3.0`**, built by lerobot **0.5.2** at commit
`58ccc0150867a027e4b3b4ce18dd589113d6ea09`. Pin Colab's lerobot to the SAME commit so
(a) the v3.0 dataset loads and (b) the output checkpoint loads in our `lerobot-vla`
eval env (also 0.5.2):

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && git checkout 58ccc0150867a027e4b3b4ce18dd589113d6ea09
conda install -y ffmpeg=7.1.1 -c conda-forge      # condacolab; needed for video decode
pip install -e ".[smolvla]"
```

**The fine-tune command** (`lerobot-train`; equivalently `python lerobot/scripts/train.py`):

```bash
DATASET_REPO=anikmall/armani_pick_red_v1   # ← the SAME repo id you pushed in step 2
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${DATASET_REPO} \
  --output_dir=/content/drive/MyDrive/s3/smolvla_pick_red \
  --job_name=smolvla_pick_red \
  --batch_size=64 \
  --steps=20000 \
  --policy.scheduler_decay_steps=20000 \
  --save_freq=5000 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.device=cuda \
  --wandb.enable=false        # flip to true ONLY after `wandb login` (else it stalls the run)
```

| arg | why |
|---|---|
| `--policy.path=lerobot/smolvla_base` | fine-tune FROM the pretrained base (not from scratch) |
| `--batch_size=64` | official default (A100). **T4 16 GB: drop to 16** (or 32) if it OOMs |
| `--steps=20000` | official default; SmolVLA converges fast on a small set — stop earlier if train loss plateaus |
| `--policy.scheduler_decay_steps=20000` | **match `--steps`** or the LR never decays (AGENT_GUIDE §7.5) |
| `--save_freq=5000` | checkpoint every 5k for eval + resume |
| `--policy.freeze_vision_encoder=false` | unfreeze the vision encoder — substantially better on a specialized task |
| `--output_dir=/content/drive/MyDrive/...` | **write to mounted Google Drive** so checkpoints survive a Colab disconnect |

**Expected wall-clock:** A100 ≈ 3–5 h at 20k/batch 64; T4 much slower — halve steps
(10k, `scheduler_decay_steps=10000`) and batch 16, still ~6–10 h.

**Don't fly the train blind.** W&B is off by default so an unattended Colab run can't
stall on a login prompt — but for *your* run, **turn it on**: `wandb login` (free), then
`--wandb.enable=true`. The loss curve is how you catch a broken or overfitting train at
minute 10 instead of after a wasted multi-hour run + eval. No W&B account? At least
watch the console/TensorBoard loss — just don't run it unwatched with no loss signal.

**Resume after a Colab disconnect** (works only if `--output_dir` was on Drive):

```bash
lerobot-train --config_path=/content/drive/MyDrive/s3/smolvla_pick_red/checkpoints/last/pretrained_model/train_config.json --resume=true
```

**Get the checkpoint** for eval: the last checkpoint is at
`<output_dir>/checkpoints/last/pretrained_model/` — download that whole folder (it
has `config.json`, `model.safetensors`, the processor JSONs + stats). That folder IS
the `--policy-path` you pass to eval.

**Mac-MPS fallback (slow — flag it honestly).** If no Colab GPU: run the same
`lerobot-train` in `lerobot-vla` with `--policy.device=mps --batch_size=4 --steps=10000
--policy.scheduler_decay_steps=10000`. SmolVLA on MPS is **many hours to a day+** —
Colab is strongly preferred. Whichever you use, **record the exact lerobot commit +
device in `docs/spike_s3_results.md`.**

## 4. Eval on the arm (`lerobot-vla`, the S2 runner)

The S2 runner loads a fine-tuned checkpoint via `--policy-path`; clamp, kill switch,
caps, scoring, JSONL are all unchanged. See [`eval.md`](eval.md) for the protocol —
including the **architect-ratified `--clamp-profile recorded`** exception (fine-tuned
eval only, if `check_dataset` flagged the demos beyond policy ±60°; refused on the base).

```bash
PY=~/miniforge3/envs/lerobot-vla/bin/python
$PY -m experiments.s2_zero_shot.run_zero_shot \
  --policy-path /path/to/checkpoints/last/pretrained_model \
  --live --seconds 20 --task "Pick up the red block" --episode-tag s3_trial_1 --trial
```
