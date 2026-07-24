# S3 data-collection SOP — read this fully before recording

**This document decides whether S3 works.** The fine-tune is easy; the data is
everything. You are recording **~50 demonstrations of ONE pick, done the SAME way
every time**, varying only *where* the block is. Consistency is not neatness — it
is the requirement.

## Why consistency matters (the one line that matters)

A first behavior-cloning pass learns to **copy** what it sees. On a small dataset it
**cannot** learn multi-modal behavior: if you approach the block from the left
sometimes and the right other times, the model averages the two into a motion that
does neither — it drives *between* them. That averaging is the most likely cause of
the earlier ACT attempt failing. **One way, many positions.** Pick a single clean
strategy and repeat it.

## FIXED — identical in every single demo

| thing | keep it… |
|---|---|
| **object** | ONE red block. Prefer a **roughly symmetric** one (cube/cylinder) so its orientation doesn't matter. If it isn't symmetric, always place it in the **same orientation**. |
| **instruction** | exactly `"Pick up the red block"` (the `ARMANI_S3_TASK` default) — one command, byte-identical, every demo. |
| **camera** | the C920 **locked** in the same pose as calibration. Do NOT move it during or between demos. |
| **lighting** | same lamp, same brightness. No sun changing through a window. |
| **start pose** | begin every demo from the **same rest pose**. |
| **approach** | same **direction** in every demo (e.g. always straight down from above, or always from the front). Pick one. |
| **wrist orientation** | the gripper's **roll fixed** the same every time — do not rotate the wrist to "line up" with the block. Same grip angle always. |
| **grasp height** | close the gripper at the **same height** above the table each time. |
| **lift height** | lift to the **same height**, then stop. |

> **Record the natural pick — do NOT contort it to fit the clamp.** You record in
> teleop (the wider `recorded` profile). A comfortable table-reaching grasp very likely
> needs shoulder_lift / elbow_flex / wrist_flex beyond ±60° (the S1 geometry finding —
> near-vertical reach is marginal there). That is fine and expected: demonstrate a
> **clean, consistent** grasp, don't distort it to stay inside ±60°. After recording,
> `check_dataset.py` reports the **per-joint action range**; send those numbers to the
> architect, who sets the eval clamp profile from them (default `policy` ±60°; if the
> demos exceed it, the architect ratifies the `recorded` profile for the fine-tuned
> eval only). A **riser** under the block is a fallback if a tighter live envelope is
> wanted. The one hard rule stays: keep the grasp *identical* across all demos.

## VARIED — the ONLY thing that changes

**Only the block's (x, y) position on the table.** Nothing else.

### Coverage plan (~50 positions)

- Mark a **target region** on the table that is BOTH **inside the arm's reach** AND
  **fully in the camera view** — roughly hand-sized (about 20 × 15 cm works).
- Lay a rough **grid** over it — e.g. **7 rows × 7 columns ≈ 49 spots**, evenly
  spread across the whole region (corners and middle, not clustered).
- At each spot, nudge the block a **small random jitter** (±1–2 cm) so positions
  aren't perfectly on a lattice.
- Spread coverage **evenly** — the model only learns to pick where it has seen the
  block. Sparse corners → misses in the corners.

## One demo = one clean success

Each episode: **rest → approach → grasp → lift → stop.** Keep every demo about the
same length (~15–20 s; `ARMANI_S3_EPISODE_TIME_S`). Smooth, deliberate, the same
motion each time.

**Discard every fumbled or failed take.** This is counter-intuitive but essential:
a first BC pass trains on **clean successes only** — it is learning to *copy*
success, not to *recover* from failure. If the grasp slips, the block rolls, the
timing is off, or you had to correct mid-motion → **redo it** (press **←**), don't
keep it. A dataset of 50 clean picks beats 80 with 30 messy ones.

## Recording — keys and flow

```bash
# 1. Ports change per USB plug-in on macOS — set them for the session:
export ARMANI_FOLLOWER_PORT=/dev/tty.usbmodem<follower>
export ARMANI_LEADER_PORT=/dev/tty.usbmodem<leader>
export ARMANI_CAMERA_INDEX=0            # the C920

# 2. Dry-run first — read the command it will run:
python experiments/s3_finetune/record_picks.py

# 3. Record (operator + arm). Keys during recording:
python experiments/s3_finetune/record_picks.py --go
#   →  finish this episode, move to the next (reposition the block first)
#   ←  redo the current episode (use this for any fumble)
#   ESC finish the whole session
#   Add more later:  record_picks.py --go --resume
```

## After recording — verify BEFORE training

```bash
python experiments/s3_finetune/check_dataset.py     # camera present? 6-D state/action? length outliers?
```

Then **always visualize** (the single best way to catch bad data): push to the Hub
(see `README.md`), open <https://huggingface.co/spaces/lerobot/visualize_dataset>,
paste your repo id, and scrub through episodes — look for camera blur, the block out
of frame, inconsistent approaches, or a demo where the pick failed. Fix the data
now; a bad demo you keep is a bad habit the model learns.
