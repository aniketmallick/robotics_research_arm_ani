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

---

# Stage 4 findings — eyes, homography and the IK ladder

## IK ladder: Plan A works (placo + lerobot `RobotKinematics`)

`lerobot.model.kinematics.RobotKinematics` exists in 0.5.2 and drives **placo**.
placo was NOT installed, and installing it is the one dependency risk in this stage,
because pip's first solution wanted to pull `numpy` from 2.2.6 up to 2.3.5 — i.e. to
move a pin underneath the working teleop environment.

**What was done instead** (numpy held at the installed version with a constraint file):

```bash
python -m pip install -c constraints.txt placo   # constraints.txt: numpy==2.2.6
python -m pip install -c constraints.txt "cmeel-urdfdom==4.0.1" "cmeel-tinyxml2==10.0.0"
```

The two extra pins are required: holding numpy back makes pip select `placo 0.9.16`,
whose binary links `liburdfdom_sensor.4.0` and `libtinyxml2.10`, while the unconstrained
solve would have paired it with urdfdom 6.0.0 / tinyxml2 11.0.0. Mismatched cmeel wheels
fail at *import* time with a `dlopen` error, not at install time.

Resulting versions: `placo 0.9.16`, `pin 3.4.0`, `eigenpy 3.10.3`, `cmeel-urdfdom 4.0.1`,
`cmeel-tinyxml2 10.0.0`.

**Verified unchanged after the install:** numpy 2.2.6, cv2 4.13.0, torch 2.11.0,
lerobot 0.5.2, and `from lerobot.robots.so_follower import SO101Follower` still imports.
Nothing about teleop moved.

## URDF provenance

`armani/data/so101_kin.urdf` is `so101_new_calib.urdf` from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(`Simulation/SO101/so101_new_calib.urdf`, Apache-2.0), with **all 34 `<visual>` and
`<collision>` elements removed**. placo refuses to load a URDF whose referenced STL
meshes are absent, and inverse kinematics needs only joints, origins and limits — so
stripping the geometry avoids vendoring 13 binary meshes for nothing.

Regenerate:

```bash
curl -sSLO https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf
python -c "
import xml.etree.ElementTree as ET
t=ET.parse('so101_new_calib.urdf'); r=t.getroot()
for link in r.iter('link'):
    for tag in ('visual','collision'):
        for el in list(link.findall(tag)): link.remove(el)
t.write('armani/data/so101_kin.urdf', encoding='utf-8', xml_declaration=True)"
```

Joint names match ours exactly (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper`) and the frame `gripper_frame_link` exists — it is the IK target.

### URDF zero == calibrated zero (assumed, corroborated, still needs the arm)

lerobot's own `robot_kinematic_processor.py` feeds raw calibrated `.pos` degrees straight
into `forward_kinematics`, so the library itself assumes the two conventions coincide.
The joint ranges corroborate it — URDF limits vs the ranges derived from this robot's
calibration file line up closely:

| joint | URDF | our `PHYSICAL_LIMITS` |
|---|---|---|
| shoulder_pan | ±110.0 | ±114.9 |
| shoulder_lift | ±100.0 | ±111.0 |
| elbow_flex | ±96.8 | ±97.7 |
| wrist_flex | ±95.0 | ±96.3 |
| wrist_roll | -157.2 .. +162.8 | ±180.0 |

**This has not been confirmed on hardware.** `tests/smoke_10_hover.py` is the test that
confirms it: if FK disagrees with reality, the arm hovers somewhere other than the object
and the operator sees it immediately.

### `inverse_kinematics` has no success signal

It is a *single* soft-QP step (`solver.solve(True)`), not an iterate-to-convergence solve,
and it returns joint angles whether or not they achieve anything. lerobot gets away with
calling it once per tick because teleop deltas are tiny. `armani/kinematics.py` therefore
iterates it to rest **and verifies the answer with forward kinematics**, reporting failure
rather than returning a plausible-looking pose. It also hands placo our *policy* limits
(`set_joint_limits` + `enable_joint_limits`) so solutions are policy-legal by construction
instead of being clamped afterwards behind the verification.

## Top-down reach inside the policy envelope — the constraint that shapes stage 5

Brute-force sweep of the policy envelope (±60° on shoulder_lift / elbow_flex / wrist_flex),
`shoulder_pan = 0`, reporting the reachable radius *r* in metres:

| approach lean | z=+0.15 | z=+0.12 | z=+0.10 | z=+0.08 | z=+0.05 | z=0.00 | z=-0.05 |
|---|---|---|---|---|---|---|---|
| ≤5° (vertical) | — | — | — | — | — | 0.14–0.26 | 0.15–0.30 |
| ≤15° | — | — | — | — | — | 0.13–0.32 | 0.12–0.34 |
| ≤25° | — | — | — | — | 0.19–0.33 | 0.13–0.37 | 0.09–0.36 |
| ≤35° | — | — | 0.24–0.33 | 0.21–0.36 | 0.19–0.38 | 0.13–0.39 | 0.06–0.38 |
| ≤45° | 0.30 | 0.23–0.38 | 0.21–0.40 | 0.20–0.40 | 0.19–0.41 | 0.13–0.42 | 0.03–0.40 |

Read it:

* A **near-vertical approach only exists at z ≤ 0 m** inside the policy envelope.
* At a 10 cm hover the best achievable lean is about **35°**. This is why
  `HOVER_MAX_TILT_DEG` is 40 and not 5 — hover constrains *position* and reports the lean.
* It is not purely a policy artefact: even at full URDF limits, vertical at z=0.10 m is a
  razor-thin band (r 0.189–0.219) and z=0.12 m is impossible. The SO-101 simply cannot
  fold far enough over itself to point straight down 10 cm up.

**Consequence for stage 5 (architect's call, not mine):** a vertical grasp at z≈0 is
available for r 0.14–0.26, but the overlap with a policy-legal 10 cm hover over the *same*
(x, y) is only **r 0.24–0.26** — very tight. Options: relax the pitch-joint policy limits
(a safety rule 2 change), lower `HOVER_HEIGHT_M`, or mount the arm on a riser so
`TABLE_HEIGHT_M` goes negative and the whole table moves into the easy part of the
envelope. All of the above assume `TABLE_HEIGHT_M = 0`; the operator measures it.

## Camera calibration

OpenCV 4.13.0 has full `cv2.aruco` including `CharucoBoard` / `CharucoDetector`, and
highgui works (needed for click-to-pick in the gripper-tip method). Both
`opencv-python` and `opencv-python-headless` are installed; `namedWindow` succeeds.

The ChArUco path is verified end-to-end on a synthetic render in `tests/test_calibrate.py`
(24 corners detected, homography fit to 0.009 px), so only the physical board and the
operator's ruler are untested.

**Note on method choice:** ChArUco is the specified primary and is fast, but it can only
give *pixel* accuracy — the *robot* coordinates come from the operator measuring where the
board sits, so a 5 mm ruler slip shifts the entire map 5 mm. The gripper-tip method needs
no ruler at all: forward kinematics supplies exact robot coordinates and the robot
effectively measures itself. If the first hover is visibly off, try `--method tip`.

---

# Stage 6 findings — trust gates

## BLOCKER FOR DEMO DAY: the Gemini free tier is 20 requests/day/model

Hit while testing G5 on 2026-07-19. Every model in `config.GEMINI_MODELS` returned
`429 RESOURCE_EXHAUSTED`, and it was **still** exhausted 75 seconds later — this is
the **per-day** quota (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
`quotaValue: 20`), not a per-minute burst limit.

What one gated pick costs:

| call | when | requests |
|---|---|---|
| `eyes.locate` | G1, every pick | 2 (`EYES_SAMPLES`, two prompt variants) |
| `eyes.list_visible` | G2, only when ambiguous | 1 |
| `eyes.confirm_held` | G5, every completed pick | 1 |

So **3 requests for a clean pick, 4 for an ambiguous one** — roughly **five to six
picks per day** on the free tier, across rehearsal AND the demo itself.

**Operator action before the venue: enable billing on the Google API key.** This
is not optional. There is no code change that fixes it; the gates degrade
gracefully (G5 falls back to the gripper reading and says "vision would not
commit") but G1 failing means no pick can start at all.

Also fixed while finding this: `eyes._ask` used to retry every model twice (once
with `ThinkingConfig`, once without), so a rate-limited call burned **six**
requests instead of three. It now only re-asks the same model when the error
actually looks like a rejected config, and stops immediately on a quota error.

## Zone geometry vs the ambiguity margin

The operator's five zones are spaced 104–215 px apart, the closest pair being
z2–z3 at 104 px. For two spots `D` apart, an object `d` from the nearer one has
margin `D − 2d`, so staying above `ASSIGNMENT_MARGIN_PX` needs:

    d <= (D - margin) / 2

| margin | clear radius at D=104 px |
|---|---|
| 60 (default) | 22 px |
| 50 | 27 px |
| 40 | 32 px |

So at the default, Gemini's point must land within **22 px** of the clicked
centre for the pick to read unambiguous. That is tight — the click is the centre
of the *mark*, while Gemini points at the centre of the *object*, and those differ
by more than 22 px for anything larger than a marker pen.

**This is a tuning dial, not a bug**: firing G2 too often makes the robot ask
"which one?" when it did not need to, which is annoying but safe. If the demo
asks too often, lower `ARMANI_ASSIGNMENT_MARGIN_PX` to 40. Measure with
`smoke_11_pick.py --identity` before changing it.

## Confidence in practice

`confidence = vision × (0.5 + 0.5 × clamp(margin/120, 0, 1))`. Consequences worth
knowing before tuning `CONF_APPROVAL = 0.60`:

- A dead-centre object on a well-spaced spot keeps its full vision confidence.
- An object exactly on the 60 px ambiguity threshold is multiplied by 0.75.
- The *maximum* achievable confidence equals the vision confidence, which for a
  clear object has measured around 0.83–0.95 in stage 4/5 runs. A vision score
  below ~0.60 can therefore never clear G4 no matter how clean the assignment —
  which is the intended fail-closed direction.
