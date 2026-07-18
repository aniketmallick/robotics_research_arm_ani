# Recording the pick macros — operator runbook

You teleoperate one **working pick** at each marked spot. `armani/pick.py` replays
the macro for whichever spot Gemini says the object is on. About 15 minutes.

This is the demo pick path (CLAUDE.md, Grasp). The reason it exists: you
demonstrating a grasp that works is strictly more reliable than the arm computing
one. Nothing here needs coordinates, homography or IK.

**Do these first:**

```bash
conda activate lerobot
python scripts/capture_home.py      # if you have not already
python scripts/define_zones.py      # click each marked spot once
```

`define_zones.py` prints the episode order at the end. **That order is the
contract** — zone 1's macro is episode 0, zone 2's is episode 1, and so on. Record
them in exactly that order.

Entry point re-verified on this machine (2026-07-19): `lerobot-record` exists in
the `lerobot` env and every flag below appears in `lerobot-record --help`.

## 1. Set up the table

- Put a **physical marker** at each spot (tape cross, sticker, drawn circle) — the
  same marks you clicked in `define_zones.py`.
- Put a **demo object physically on the spot** you are recording. You are
  recording a real grasp, not a mime: the gripper must actually close on
  something at the right height, or the replay will close on air.
- Do not move the camera. The zone pixels were measured where it is now.

## 2. The command

Five episodes, one per zone, **in the order `define_zones.py` printed**:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$ARMANI_FOLLOWER_PORT" \
  --robot.id=follower_arm \
  --robot.use_degrees=true \
  --teleop.type=so101_leader \
  --teleop.port="$ARMANI_LEADER_PORT" \
  --teleop.id=leader_arm \
  --dataset.repo_id=anikmall/armani_picks \
  --dataset.root="$HOME/.cache/huggingface/lerobot/anikmall/armani_picks" \
  --dataset.single_task="pick the object from the marked spot" \
  --dataset.num_episodes=5 \
  --dataset.fps=30 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.video=false \
  --dataset.push_to_hub=false \
  --display_data=false
```

Differences from the gesture recording, and why:

- **`--dataset.repo_id` / `--root` point at `armani_picks`**, a separate dataset.
  Mixing picks into the gesture dataset would shift every gesture's episode index
  and silently break stage 2 and 3.
- **`--dataset.episode_time_s=30`** (gestures used 10). A pick is approach,
  descend, close, lift, retreat — it needs the room. Overrun is the most common
  reason a take ends with the object still on the table.
- **`--dataset.reset_time_s=10`** so you have time to put the object back on the
  mark between takes.
- **`--dataset.root` is still not optional.** `lerobot-record` calls
  `stamp_repo_id()` at creation and appends `_YYYYMMDD_HHMMSS` to the repo id, so
  without an explicit root the data lands in `anikmall/armani_picks_20260719_...`
  and `smoke_11` reports no macros forever. (`config._resolve_dataset_root` falls
  back to the newest timestamped sibling, but pin it anyway.)

## 3. What each macro must do

**Every macro starts AND ends at the captured home pose.** Same rule as the
gestures, same reason: it makes replays chainable and keeps pre-positioning to a
short, safe move.

The shape of one episode:

1. start at home
2. move over the spot
3. descend to the object
4. **close the gripper on it**
5. lift clear of the table
6. return toward home **still holding it**

Point 6 is the one people get wrong. The replay does **not** auto-home afterwards
— `motion.home()` commands every joint including the gripper, so homing after a
grasp would open the jaws and drop the object. `pick.play_pick` therefore passes
`return_home=False`, and the arm ends exactly where your recording ends. So end
your recording somewhere sensible, holding.

Other rules, inherited from the gesture runbook because the replay engine is the
same one:

- **Do not jerk.** Frame-to-frame jumps above 8° are refused at load time —
  lerobot would clip them at send time and play something you never recorded.
- **Stay off the mechanical stops.** The `recorded` clamp profile trims 2°, so a
  take pressed into a stop will not match on replay.
- **Re-record a bad take immediately** (left-arrow redoes the current episode).
  Fixing order afterwards is painful: the episode index *is* the zone.
- Leave about a second of stillness at each end.

## 4. Verify

```bash
python tests/smoke_11_pick.py --dry-run      # decision path, no hardware
python tests/smoke_11_pick.py --live --object banana
```

Then the number that actually matters — can Gemini tell the objects apart on the
spots?

```bash
python tests/smoke_11_pick.py --identity 10
```

That writes an `identity_accuracy` entry to `logs/decisions.jsonl`, with the
confusions broken out. Identity is vision's *only* job on this path, so that
figure — not a millimetre figure — is the demo's competence bar.

## Troubleshooting

| symptom | cause |
|---|---|
| `no pick macros recorded` | `--dataset.root` not pinned, or repo id does not match `config.PICK_DATASET_REPO_ID` |
| `no pick macro for 'x': needs episode N but only M are recorded` | recorded fewer episodes than zones, or out of order |
| arm picks the wrong spot | episodes recorded in a different order than `define_zones.py` printed — the index *is* the zone |
| macro runs but the object is not lifted | the object was not on the mark when you recorded, or the grasp height is off; re-record that episode |
| object is lifted then dropped | the take ended with the gripper opening — re-record ending while still holding |
| `jumps N deg between frames` | jerky take; re-record that episode more slowly |
| every object reads as ambiguous | two spots are closer than `ARMANI_ASSIGNMENT_MARGIN_PX`; move them apart |
| object reads as "not on a marked spot" | it is more than `ARMANI_ZONE_MAX_DISTANCE_PX` from every zone, or the camera moved — re-run `define_zones.py` |
