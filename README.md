# ARM-ANI

A voice-interactive SO-101 robot arm with trust gates: it talks back, performs
gestures, picks named objects it sees — and routes every risky action through
safety gates that live in Python, not in a prompt.

See `CLAUDE.md` for the project constitution and `docs/env_report.md` for what is
actually installed on this machine.

**Status: stage 1 (foundation + smoke tests) complete.** No feature code yet.

## Setup

**Activate the conda env first — every command below assumes it.** Do not create a new
env and do not reinstall lerobot:

```bash
conda activate lerobot
```

Being in `(base)` is *not* enough: on this machine bare `python` resolves to
`~/.platformio/penv/bin/python`, which has none of the dependencies. Check with:

```bash
python -c "import sys, lerobot; print(sys.executable)"
# expect: /Users/Aniket.Mallick/miniforge3/envs/lerobot/bin/python
```

Or skip activation and use the interpreter directly:

```
/Users/Aniket.Mallick/miniforge3/envs/lerobot/bin/python scripts/doctor.py
```

lerobot is an **editable install**: the source lives in
`~/Documents/Claude/Projects/Robotics/lerobot`, but it is importable from this env. That is
expected — this project does not vendor or reinstall it.

Every smoke test and `doctor.py` aborts immediately with the correct interpreter path if run
under the wrong Python, rather than reporting six confusing import failures.

Then create your `.env`:

```bash
cp .env.example .env
# add OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY
# add ARMANI_FOLLOWER_PORT and ARMANI_CAMERA_INDEX once smoke 01 and 03 reveal them
```

`.env` is gitignored. It is never committed and never printed.

## How to run — stage 1

### Everything at once

```bash
python scripts/doctor.py --dry-run     # safe: touches no hardware, no network
python scripts/doctor.py               # LIVE: smoke 02 moves the arm after you confirm
python scripts/doctor.py --skip-motion # live, but never moves the arm
python scripts/doctor.py --only 3      # just one check
```

Exit codes: `0` all passed, `1` at least one FAIL. Each individual test exits
`0 = PASS`, `1 = FAIL`, `2 = SKIP`.

### Individual smoke tests

Every test is standalone and every one accepts `--dry-run`.

| Test | What it proves | Needs |
|---|---|---|
| `tests/test_safety.py` | clamp/interpolate/envelope logic is correct | nothing (pure logic) |
| `tests/smoke_01_ports.py` | follower connects, reports all 6 joints (commands no motion) | arm plugged in, **operator confirms** |
| `tests/smoke_02_wiggle.py` | one joint moves ±5° and returns (**MOTION**) | **operator watching the arm** |
| `tests/smoke_03_camera.py` | a 640x480 frame from the C920 | C920 on its tripod |
| `tests/smoke_04_mic.py` | 2 s record + playback, rejects silence | headset selected as input |
| `tests/smoke_05_keys.py` | all three APIs answer | network, `.env` filled in |
| `tests/smoke_06_ptt.py` | global spacebar hold/release | Input Monitoring granted |

```bash
python tests/smoke_01_ports.py --dry-run
python tests/smoke_02_wiggle.py --joint wrist_roll   # default joint; rotates in place
```

### Before touching the motors

1. Plug in the follower. Confirm a port appears: `ls /dev/tty.usbmodem*`
2. `python tests/smoke_01_ports.py` — connects and streams positions, commands no motion.
3. Only then `python tests/smoke_02_wiggle.py`, with a hand near the arm.

Both tests ask *"Operator present and watching? [y/N]"* and will not proceed without a `y`,
and both refuse if stdin is not a terminal.

Smoke 01 asks too, because **connecting is not passive**: lerobot's `connect()` calls
`configure()`, which re-enables torque. The arm goes stiff the moment it connects.

### Joint limits: three profiles, not one

A single envelope cannot answer both "where may the arm *be*" and "where may it be
*commanded to go*". The parked arm rests at `shoulder_lift = -111.7°`, far outside the
conservative envelope — and treating that as illegal made `home()` refuse, which silently
broke the kill switch. So targets are clamped against a profile chosen by their origin:

| profile | applies to | envelope |
|---|---|---|
| `policy` | LLM- and IK-derived targets | ±90° base, ±60° others |
| `recorded` | gesture replay, return-to-entry recovery, `home()` | physical − 2° |
| `physical` | the send boundary only — hard backstop | full calibrated range |

`interp_move()` starts from the **measured** pose and lerps to an already-clamped target, so
every step lies between them: the arm only ever moves *toward* legality, never further out.

`OutsideEnvelopeError` now means only one thing — a joint read **beyond its physical range**
by more than 2°, i.e. an encoder or calibration fault. A parked pose outside the policy
envelope is normal and is not blocked.

`tests/test_safety.py` locks this in using the arm's real measured rest pose. Run it alone
with `python tests/test_safety.py` — no hardware, no operator.

**Kill switch:** Ctrl-C during a move stops the arm and walks it back under control.
A second Ctrl-C aborts immediately. ESC works too if Input Monitoring is granted.

### Dry run everywhere

```bash
export ARMANI_DRY_RUN=1     # or pass --dry-run per test
```

In dry-run nothing opens a serial port; a simulated arm prints the actions that
would have been sent, through the exact same clamp-and-interpolate code path.

`doctor.py` **refuses to start a live run while `ARMANI_DRY_RUN` is set**, so a leftover
flag can never produce a green summary that touched no hardware.

## Layout

```
armani/
  config.py    limits, thresholds, ids, models — one place for every constant
  safety.py    clamp_action, interp_move, SafeMotion, kill switch, require_operator
  motion.py    connect / read_positions / goto / home   (the only lerobot boundary)
  logutil.py   JSONL decision log -> logs/decisions.jsonl
tests/         smoke_01..06, each standalone, each --dry-run capable
scripts/doctor.py
docs/env_report.md
```

## Safety notes that are easy to get wrong

* The five body joints are in **degrees**; the **gripper is 0–100 percent**, not degrees.
* lerobot does **not** clamp degree targets to the calibrated range — `safety.clamp_action()`
  is the only guard against an over-travel command.
* `config.HOME_POSE` is still an **unverified placeholder**. Smoke 02 deliberately returns to
  the pose the arm started in rather than driving to it. The kill-switch and error paths *do*
  home, because safety rule 7 requires it — so verify `HOME_POSE` on hardware early in stage 2.
* Clamping happens in `Arm.send()`, the last line before the motors, so no caller can bypass it.
* A **second Ctrl-C** is a hard abort: the arm is left where it is and is *not* homed.
* Calibration already exists and is never recreated. `motion.connect()` passes
  `calibrate=False` so lerobot's interactive recalibration can never be triggered by accident.
