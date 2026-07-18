# Environment Report — ARM-ANI

Discovered 2026-07-18 during stage 1. Everything here was read off this machine,
not assumed. Nothing under `~/.cache/huggingface/lerobot/` was modified.

## The environment to use

**Use this interpreter for everything:**

```
/Users/Aniket.Mallick/miniforge3/envs/lerobot/bin/python
```

| | |
|---|---|
| Conda env | `lerobot` (miniforge3) |
| Python | 3.12.13 |
| OS | macOS 26.3.1, arm64 |
| lerobot | 0.5.2, **editable install** |
| lerobot source | `/Users/Aniket.Mallick/Documents/Claude/Projects/Robotics/lerobot` |
| lerobot git rev | `58ccc01` |

### Why this env and not the other one

There are two environments with lerobot 0.5.2, both editable-installed from the
same checkout:

| Env | Python | Talks to motors? |
|---|---|---|
| `miniforge3/envs/lerobot` | 3.12.13 | **Yes** — has `feetech-servo-sdk` 1.0.0 |
| `Robotics/lerobot/.venv` | 3.13.13 | No — no feetech SDK |

The `.venv` cannot drive an SO-101 at all. The conda `lerobot` env is the one
that runs teleop. **Do not create a third env and do not upgrade lerobot.**

## Calibration (already exists — never recreate)

| Role | Path | id |
|---|---|---|
| Follower | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json` | `follower_arm` |
| Leader | `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader_arm.json` | `leader_arm` |

Both files were **recalibrated on 2026-07-18 (13:33 and 13:38)** — they are fresh.

The directory is `so_follower`, not `so101_follower`, because **lerobot 0.5.2 merged
SO-100 and SO-101 into a single `SOFollower` class** whose `name` attribute is
`"so_follower"`. `SO101Follower` and `SO101FollowerConfig` still exist as aliases,
so CLAUDE.md's stack decision holds — but there is no distinct SO-101 class any more.
lerobot resolves calibration to `calibration/robots/<class name>/<id>.json`, which is
why the id must stay `follower_arm`.

### Units — the thing most likely to cause a bug

`SOFollowerConfig.use_degrees` defaults to `True`, so:

* the five body joints are commanded in **degrees** (`MotorNormMode.DEGREES`);
* the **gripper is always `RANGE_0_100`** (a 0–100 percentage) regardless of `use_degrees`.

CLAUDE.md says "normalized -100..100 joint positions". That is the `use_degrees=False`
behaviour and does **not** match this version's default. `armani/config.py` pins
`USE_DEGREES = True` explicitly so a library default change cannot silently reinterpret
every number in the config. **Flagged for the reviewer.**

Also worth knowing: in DEGREES mode `MotorsBus._unnormalize()` does **not** clamp to
`range_min`/`range_max`. An out-of-range degree value is converted and sent to the motor.
`safety.clamp_action()` is therefore the only thing preventing an over-travel command.

### Joint ranges derived from the follower calibration

`deg = (ticks - (range_min + range_max)/2) * 360 / 4095`

| Joint | ticks min–max | physical range | policy limit (CLAUDE.md) |
|---|---|---|---|
| shoulder_pan | 715–3329 | ±114.9° | **±90°** |
| shoulder_lift | 764–3289 | ±111.0° | **±60°** |
| elbow_flex | 1281–3503 | ±97.7° | **±60°** |
| wrist_flex | 651–2841 | ±96.3° | **±60°** |
| wrist_roll | 0–4095 | ±180.0° | **±60°** |
| gripper | 2033–3511 | 0–100 % | **0–100 %** |

Every policy limit is strictly inside its physical limit, so policy binds first.
`config._assert_limits_within_physical()` enforces this at import time.

### Measured rest pose (2026-07-18) — and why one envelope was not enough

The parked follower reported:

| joint | at rest | inside policy? |
|---|---|---|
| shoulder_pan | 3.78° | yes |
| shoulder_lift | **−111.69°** | no (±60) |
| elbow_flex | **+96.79°** | no (±60) |
| wrist_flex | **+68.92°** | no (±60) |
| wrist_roll | 1.36° | yes |
| gripper | 31.12 % | yes |

Note `shoulder_lift` reads **past** its calibrated −111.0: lerobot's DEGREES branch in
`_normalize()` uses the raw tick value, not the range-bounded one, so reads at a mechanical
stop legitimately overshoot. Hence `PHYSICAL_TOLERANCE = 2.0`.

This pose broke the original single-envelope design: `home()` targets all six joints, three
were "outside limits", so it refused — taking the kill switch and every error-recovery path
with it. Fixed by splitting into the `policy` / `recorded` / `physical` profiles described in
the README, locked in by `tests/test_safety.py`.

## lerobot 0.5.2 dataset & replay API (stage 2 findings)

Stage 3 builds on this layer, so here is what actually holds.

**Loading is easy and offline.** Verified against an existing local dataset:

```python
ds = LeRobotDataset(repo_id, root=<local path>, episodes=[i])   # 0.32 s, HF_HUB_OFFLINE=1
actions = ds.hf_dataset.select_columns("action")["action"]      # no video decode
names   = ds.meta.features["action"]["names"]  # ['shoulder_pan.pos', ... 'gripper.pos']
```

`episodes=[i]` really does filter — `num_episodes` becomes 1 and `num_frames` is that
episode's length. `select_columns` never touches the video files, so replay loading stays
fast even on a dataset recorded with cameras.

**Surprises worth carrying into stage 3:**

1. **Datasets are v3.0 format**; `meta/` holds `episodes/` (a directory), `info.json`,
   `stats.json`, `tasks.parquet`. `total_episodes` lives in `info.json`.
2. **An out-of-range episode raises `ValueError: Instruction "train" corresponds to no data!`** —
   useless to an operator who recorded 3 of 8 gestures. `gestures.load_gesture` checks
   `total_episodes` itself first and says which gestures are missing.
3. **Nothing in the metadata records `use_degrees`.** A dataset recorded with
   `use_degrees=false` stores normalised −100..100 and replays as silently wrong angles.
   The only available signal is the value range, so `gestures._check_units` warns when no
   joint exceeds 100. The runbook pins `--robot.use_degrees=true`.
4. **Real teleop violates the interpolation speed budget.** Measured frame-to-frame deltas on
   an existing episode: up to **6.16** (gripper), 4.04 (shoulder_lift), 3.08 (elbow_flex) at
   30 fps, against a `MAX_JOINT_SPEED/CONTROL_HZ` budget of 1.50. A `max_relative_target`
   derived from that budget would have silently clipped every replay. `MAX_FRAME_DELTA = 8.0`
   is sized off recorded reality instead, and interp_move keeps enforcing 1.8°/step for
   interpolated motion.
5. **Real teleop also violates the policy envelope** — episodes start at
   `shoulder_lift ≈ −107.9`, which `policy` would clip by ~48°. This is the empirical
   justification for the `recorded` profile in safety rule 2.
6. **`enable_torque` does not set `Goal_Position`.** It writes only `Torque_Enable` and `Lock`,
   so a servo drives to whatever goal was last written — possibly from a previous session.
   Both `capture_home` (before releasing its `torque_disabled` block) and `connect()`
   (immediately after connecting) now park the goal at the measured pose. A twitch *during*
   `connect()` cannot be prevented: `configure()` re-enables torque before we get control.
7. **`cv2` and `av` ship duplicate `libavdevice` dylibs.** Every lerobot CLI prints an
   objc warning about "spurious casting failures and mysterious crashes". Harmless so far;
   the gesture recording avoids cameras entirely, which sidesteps it.
8. **CLI robot type is still `so101_follower`** even though the class is `SOFollower`, and
   `so101_leader` for the teleoperator. Both verified present in `lerobot-record --help`.

## Serial ports

`ls /dev/tty.usbmodem*` → **no matches at the time of writing. Both arms are unplugged.**

Leader vs follower could therefore **not** be identified this stage. A previous session on
this machine used `/dev/tty.usbmodem5B3D0412751` for the follower with
`--robot.id=follower_arm`; treat that as a hint, **not** as verified — macOS port names can
change between boots and USB ports.

**Operator action:** plug in the follower only, run `python tests/smoke_01_ports.py`, note the
port, then plug in the leader and note the second port. Put the follower one in `.env` as
`ARMANI_FOLLOWER_PORT`. `motion.resolve_follower_port()` prompts rather than guessing when
more than one port is present.

## Cameras

Four cameras respond to OpenCV/AVFoundation. macOS reports these devices:

* OBS Virtual Camera
* FaceTime HD Camera
* **HD Pro Webcam C920** ← the one we want
* Aniket's iPhone Camera (Continuity)

| OpenCV index | negotiated |
|---|---|
| 0 | 640x480 @ 30 |
| 1 | 640x480 @ 30 |
| 2 | 1920x1080 @ 60 |
| 3 | 640x480 @ 30 |

**The C920's index is NOT yet confirmed.** OpenCV cannot report device names, and the
AVFoundation index order does not reliably match the `system_profiler` order. A frame
captured from index 0 showed a person at a desk, i.e. index 0 is *not* the table view.

`pyobjc-framework-AVFoundation` would give real device names but is not in the approved
install list for this stage, so it was not installed. Instead `smoke_03_camera.py` writes
`tests/out/probe_<index>.jpg` for every working index and asks the operator to pick.

**Operator action:** run `python tests/smoke_03_camera.py`, look at the probe images, choose
the C920's table view, then set `ARMANI_CAMERA_INDEX` in `.env`.

## Audio

* Default input: `MacBook Pro Microphone` — verified working (2 s take, peak 0.0406).
* Default output: `MacBook Pro Speakers` — playback verified.
* **The demo mic must be the wired headset**, not this and not the C920 mic. Switch the macOS
  input device before the demo and re-run `smoke_04_mic.py`.

## macOS permissions

| Permission | State | Needed for |
|---|---|---|
| Camera | **Granted** (was denied; the first probe triggered the prompt) | C920 capture |
| Microphone | **Granted** (real audio captured, not silence) | voice input |
| Input Monitoring / Accessibility | **Not verified** — needs a human to press space | global push-to-talk |

Permissions attach to the **terminal application**, not to Python. After granting one,
the terminal app must be restarted.

## Packages installed for this project

Installed into the conda `lerobot` env. Verified beforehand with `pip install --dry-run`:
**zero already-installed packages changed version**, so the working teleop env is untouched.

| Package | Version |
|---|---|
| openai-agents | 0.18.3 |
| openai | 2.46.0 |
| google-genai | 2.12.1 |
| anthropic | 0.117.0 |
| python-dotenv | 1.2.2 |
| sounddevice | 0.5.5 |
| soundfile | 0.14.0 |
| pynput | 1.8.2 (was already present) |

### `opencv-python` was deliberately NOT installed

lerobot depends on `opencv-python-headless`; the two packages both provide the `cv2`
module and conflict. It turned out that `opencv-python` **4.13.0.92 is already installed**
(pulled in by `gym-pusht`) at exactly the same version as the headless build, so `import cv2`
works and no install was needed. Installing it again was avoided as a needless risk to a
working env.

Other pre-existing versions of note: numpy 2.2.6, torch 2.11.0.

### OWLv2 / transformers

Not attempted. CLAUDE.md marks it optional and stage 1 does not need it.

## API keys

Checked by `tests/smoke_05_keys.py`, which prints presence only and never a value.

| Key | State | Live check |
|---|---|---|
| `GOOGLE_API_KEY` | set | **PASS** — `gemini-robotics-er-1.6-preview` (the *primary* model) answered a vision query on a real camera frame |
| `OPENAI_API_KEY` | **missing** | not run |
| `ANTHROPIC_API_KEY` | **missing** | not run |

Gemini answering on the first-choice model means no fallback is needed for stage 4.

**Operator action:** create `.env` from `.env.example` and add the OpenAI and Anthropic keys.
