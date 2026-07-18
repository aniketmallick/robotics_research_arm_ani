# Recording the gesture dataset — operator runbook

You teleoperate the arm through eight gestures; `armani/gestures.py` replays them
by name. About 20 minutes. **Do `capture_home` first** — every gesture must start
and end at that pose.

Entry point verified on this machine: `lerobot-record` (lerobot 0.5.2, conda env
`lerobot`). Every flag below was checked against `lerobot-record --help`.

## 0. Before you start

```bash
conda activate lerobot
python scripts/capture_home.py     # if you have not already
```

Have both arms plugged in and know which port is which:

```bash
ls /dev/tty.usbmodem*
```

Put the follower port in `.env` as `ARMANI_FOLLOWER_PORT` and the leader as
`ARMANI_LEADER_PORT`, then use them below.

## 1. The command

Eight episodes, one per gesture, **in this exact order** — `armani/config.py`
maps names to episode indices by position, so the order is the contract:

| episode | gesture | what it should read as |
|---|---|---|
| 0 | `bow` | a deliberate forward bow, hold, return |
| 1 | `wave` | side-to-side wave from a raised position |
| 2 | `dance` | rhythmic side-to-side with some wrist |
| 3 | `nod_yes` | two clear nods |
| 4 | `shake_no` | two clear side-to-side shakes |
| 5 | `look_around` | slow scan left, pause, right, pause, centre |
| 6 | `celebrate` | fast rise, gripper snaps open, small shimmy |
| 7 | `sad_droop` | slow collapse forward, hold low, slow recover |

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$ARMANI_FOLLOWER_PORT" \
  --robot.id=follower_arm \
  --robot.use_degrees=true \
  --teleop.type=so101_leader \
  --teleop.port="$ARMANI_LEADER_PORT" \
  --teleop.id=leader_arm \
  --dataset.repo_id=anikmall/armani_gestures \
  --dataset.single_task="expressive gesture" \
  --dataset.num_episodes=8 \
  --dataset.fps=30 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=5 \
  --dataset.video=false \
  --dataset.push_to_hub=false \
  --display_data=false
```

Notes on the choices, so you can change them safely:

- **`--robot.use_degrees=true`** must match `config.USE_DEGREES`. Record in one
  unit and replay in another and every joint is wrong by a factor of ~2.7.
- **No `--robot.cameras`.** Gestures do not need vision, and leaving cameras out
  avoids the `libavdevice` duplicate-symbol clash between `cv2` and `av` in this
  env (it prints a warning about "mysterious crashes" whenever both load).
- **`--dataset.video=false`** — no cameras, so nothing to encode. Loading is
  faster and `gestures.py` never touches video.
- **`--dataset.push_to_hub=false`** — local only. The dataset lands in
  `~/.cache/huggingface/lerobot/anikmall/armani_gestures`.
- **`--display_data=false`** — the viewer competes for the same USB bandwidth.

If you must restart partway through, add `--resume=true` and set
`--dataset.num_episodes` to the number still missing.

## 2. Recording rules

**Every gesture starts and ends at the captured home pose.** This is the single
most important rule: it is what makes replays chainable, keeps pre-positioning to
a short move, and lets stage 3 run gestures back to back without a lurch between
them. `smoke_07` measures the spread across all eight and tells you if they drift.

- **Slow and exaggerated beats fast and subtle.** The replay is literal. A move
  that reads clearly to a person watching from two metres away is right; anything
  subtle disappears on camera.
- **Stay off the mechanical stops.** Pressing into a stop is trimmed by 2° on
  replay (the `recorded` clamp profile's standoff), so the take will not match
  what you did. `smoke_07` reports how much each episode gets trimmed.
- **Do not jerk.** Frame-to-frame jumps above 8° are refused at load time,
  because lerobot would clip them at send time and silently play something you
  never recorded. Measured on existing teleop, normal motion peaks around 6°.
- **Re-record a bad take immediately** — press the left-arrow key to redo the
  current episode rather than accepting it. Fixing order afterwards is painful:
  the episode index *is* the gesture name.
- Leave roughly a second of stillness at each end. It makes the start and end
  poses unambiguous.

## 3. Verify

```bash
python tests/smoke_07_gestures.py --dry-run
```

Expect all eight to load, with frame counts, durations, and a small clamp
deviation. Then watch one for real:

```bash
python tests/smoke_07_gestures.py --gesture bow
```

## Troubleshooting

| symptom | cause |
|---|---|
| `SKIP: gesture dataset not recorded yet` | dataset root missing — check `--dataset.repo_id` matched `config.GESTURE_DATASET_REPO_ID` |
| a gesture `jumps N deg between frames` | jerky take; re-record that episode slowly |
| `clamp would alter frames by up to N` | the take pressed into a mechanical stop; re-record with more margin |
| replayed gesture drifts from where it started | that episode did not end at home; re-record it |
| wrong gesture plays | episodes recorded out of order — the index is the identity |
