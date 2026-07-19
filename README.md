# ARM-ANI

A voice-interactive SO-101 robot arm with trust gates: it talks back, performs
gestures, picks named objects it sees — and routes every risky action through
safety gates that live in Python, not in a prompt.

See `CLAUDE.md` for the project constitution and `docs/env_report.md` for what is
actually installed on this machine.

**Status: stage 3 (realtime voice agent).** ARM-ANI talks, roasts, and moves on
command via push-to-talk. No vision or trust gates yet (stages 4+).


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

## How to run — stage 2

Order matters: home must be verified before anything will drive to it.

```bash
# 1. Pure-logic tests. No hardware, no network.
pytest tests/ -q

# 2. Capture the home pose by hand.  OPERATOR REQUIRED — torque is released.
python scripts/capture_home.py --dry-run
python scripts/capture_home.py

# 3. Record the 8 gestures. See docs/recording_gestures.md for the full runbook.
#    Then verify them:
python tests/smoke_07_gestures.py --dry-run     # loads and inspects, no motion
python tests/smoke_07_gestures.py --gesture bow # OPERATOR REQUIRED — replays it

# 4. Claude-choreographed moves.
python scripts/improvise_cli.py "do a slow clap" --dry-run
python scripts/improvise_cli.py "take a proud bow"    # OPERATOR REQUIRED
```

## How to run — stage 3 (voice agent)

ARM-ANI is a realtime voice agent (OpenAI Agents SDK, model `gpt-realtime-2.1`)
that talks back and drives the arm through six tools. Motion runs on a dedicated
worker thread, so ARM-ANI keeps talking while the arm moves.

```bash
# Build + one text round trip through the model. No audio, no motion.
python tests/smoke_08_agent.py --dry-run

# The live voice session. OPERATOR REQUIRED — one presence gate at startup.
python scripts/run_agent.py            # push-to-talk: HOLD SPACE to speak, release to send
python scripts/run_agent.py --text     # keyboard only, when audio is impossible
python scripts/run_agent.py --no-motion  # personality only; motion tools refuse
```

**Push-to-talk:** turn detection is off — hold the spacebar to stream the mic,
release to commit and get a reply. Tapping space while ARM-ANI is speaking is a
**barge-in** (it interrupts and listens). Needs Input Monitoring for the global
spacebar; use `--text` if that is unavailable.

**Tools** (each validates its args, logs to the decision log, returns compact JSON):
`list_gestures`, `play_gesture`, `improvise_move`, `go_home`, `get_status`,
`stop_motion`. `stop_motion` aborts the current move and holds — it does *not* open
the freeze menu (that belongs to the human kill switch only). Declining the startup
operator gate drops into NO-MOTION mode: the personality still demos, motion tools
return `refused`.

**Kill switch during a session:** ESC / Ctrl-C freezes the arm and hands *you* the
operator menu (return-to-start / home / torque-off / leave); the agent pauses its
own input while you decide, then resumes. The background motion worker never prompts
on stdin itself.


## How to run — stage 4 (eyes + calibration + hover)

Stage 4 gives ARM-ANI eyes and a pixel→robot map, and proves them by hovering
**10 cm above** a named object. **It stops at hover** — no descent, no gripper, no
pick. That is stage 5, and `smoke_10` fails if anything commands a descent.

Vision is deliberately **not** wired into the voice agent yet; the trust gates in
stage 6 are what get to drive it.

```bash
# 0. Pure-logic tests (IK, homography, point parsing, workspace check). No hardware.
python -m pytest tests/ -q

# 1. Does Gemini point at the right thing?  Camera + network, NO MOTION.
python tests/smoke_09_vision.py --object "red block"
#    -> writes tests/out/detect.jpg. OPEN IT. Is the marker on the object?

# 2. Map the camera to the table.  OPERATOR REQUIRED.
python scripts/calibrate_camera.py --print-board      # print at 100% scale, then MEASURE a square
python scripts/calibrate_camera.py --method charuco   # board flat on the table, one capture
python scripts/calibrate_camera.py --method tip       # fallback: no ruler needed, uses the arm

# 3. The stage deliverable.  OPERATOR MUST WATCH THE ARM.
python tests/smoke_10_hover.py --dry-run
python tests/smoke_10_hover.py --object "red block"
```

**Measure the table height first.** `ARMANI_TABLE_HEIGHT_M` is the height of the
table *surface* in robot-base coordinates: `0.0` when the arm's base sits on the same
surface as the objects, **negative** when the arm is on a riser. It decides whether the
hover is reachable at all — see the reach table in `docs/env_report.md`.

**Calibration is a physical promise.** The homography is valid only for the tripod
position, table position and frame size it was measured at. Bump any of them and every
coordinate the arm computes from vision is wrong — and it will reach confidently to the
wrong place. Re-run the calibration (~5 min). `armani/data/homography.json` records the
frame size and reprojection error so this is auditable after the fact.

**A bad map is worse than no map.** `calibrate.save()` reprojects every calibration
point and **refuses to write** above `ARMANI_CALIB_MAX_REPROJ_PX` (default 15 px mean).
An uncalibrated system has an empty table polygon, and the workspace check
(safety rule 3) then refuses every reach — it fails closed, not open.

**The confidence number is honest.** Gemini's pointing API returns no calibrated
confidence, so `eyes.locate()` builds one from what can actually be observed: how many
independently-worded queries saw the object, how closely they agreed on *where* it is,
and the model's own self-report (capped at half the score). `grasp.combined_confidence()`
then folds in how comfortably the arm can reach it. Nothing here is a calibrated
probability, and it is not presented as one.

**Approach lean.** Inside the policy envelope the SO-101 cannot point straight down
10 cm above the table — the best achievable lean is about 35°, so hover constrains
*position* and reports the lean (`HOVER_MAX_TILT_DEG`). The measured reach map is in
`docs/env_report.md`; it is the constraint stage 5's grasp has to be designed around.

## How to run — stage 5 (taught-zone pick) ⭐ the demo pick path

The architect ratified **taught zones** after stage 4's calibration proved too
fragile. Instead of computing where an object is and solving IK to reach it, the
operator teleop-records a **working pick at each marked spot**, and the arm
replays it. Vision's job shrinks to *identity* — "which spot holds the red block?" —
a coarse call that tolerates being tens of pixels wrong.

That erases the coordinate-precision, IK-verticality, riser and camera-bump
problems in one move. Every trust gate still applies, because gates are about
judgment, not millimetres.

Stage 4's `eyes.py` / `calibrate.py` / `kinematics.py` / `grasp.py` hover path
stays in the repo as a **stretch goal** and is not built on here.

```bash
# 0. Decision-path checks. No camera, no network, no motion.
python tests/smoke_11_pick.py --dry-run

# 1. Define the zones: click each marked spot once, label it. ~2 minutes.
python scripts/define_zones.py

# 2. Record one pick macro per zone, IN THE ORDER define_zones.py printed.
#    See docs/recording_picks.md for the full runbook.

# 3. Identity without risk: real Gemini call, simulated arm, asserts no motion.
python tests/smoke_11_pick.py --frame tests/out/frame.jpg --object "red block"

# 4. The real pick. OPERATOR MUST WATCH THE ARM.
python tests/smoke_11_pick.py --live --object "red block"

# 5. The competence bar: does Gemini put each object on the right spot?
python tests/smoke_11_pick.py --identity 10
```

**Zone labels name the SPOT, not the object** ("front-left", not "red block").
Which object is on which spot is decided live on every frame, so objects can be
swapped between spots at demo time and nothing needs redoing.

**Zone order is the contract.** Zone 1's pick macro is episode 0, zone 2's is
episode 1. Record them in the order `define_zones.py` prints, or the arm picks
from the wrong spot with complete confidence.

**Picks never auto-home.** `motion.home()` commands *every* joint including the
gripper, so homing after a successful grasp would open the jaws and drop the
object. `pick.play_pick` passes `return_home=False`; the macro is recorded to end
near home while still holding.

**Refusals are the feature.** `pick_object()` returns a falsy `PickResult`
**without moving** when the object is unseen, sits between two spots, is not on
any spot, or that zone has no recorded macro. Those four fields are exactly what
stage 6's trust gates read — which is why `PickResult` carries them as fields
rather than one error string.

**Identity accuracy is the number that matters** on this path, not a millimetre
figure — the grasp itself is a human recording. `--identity` writes the tally and
the confusions to `logs/decisions.jsonl`.

## How to run — stage 6 (trust gates) ⭐ the product

Five gates, in order, enforced in Python around every pick. The voice model
speaks the questions and relays the answers; **it never decides whether a gate
passes.**

| gate | asks | fails closed by |
|---|---|---|
| **G1 seen** | is the object there at all? | saying it can't see it |
| **G2 ambiguous** | two of them, or between two spots? | asking WHICH, then resolving the answer in Python |
| **G3 reachable** | is there a taught macro for that spot? | saying nobody has shown it that spot |
| **G4 confidence** | how sure am I? | stating the number; below 60% it needs spoken approval within 10s or **stands down** |
| **G5 verify** | did I actually get it? | admitting it didn't |

```bash
# Six scripted scenarios: clean, ambiguous, approved, TIMED OUT, unseen, missed grasp.
# No camera, no network, no arm.
python tests/smoke_12_gates.py --dry-run

# Real eyes on your real table, real gates, stubbed macro — nothing moves.
python tests/smoke_12_gates.py --object "red block"

# The three demo acts for real. OPERATOR MUST WATCH THE ARM.
python tests/smoke_12_gates.py --live

# And in the voice session — the actual pitch:
python scripts/run_agent.py
```

**The invariant.** There is no code path where the model's output alone moves the
arm past a gate. `gates.run_gated_pick` is ordinary Python; the model's only
inputs are the *text* of a clarification and a *yes/no* on approval, and both are
re-checked here — the answer is matched to a zone in Python (`_match_zone_by_words`,
which refuses anything it doesn't understand rather than guessing), and the
approval deadline is `gates.py`'s own clock. A prompt jailbreak can make ARM-ANI
say anything; it cannot make it pick anything.

**The 10-second stand-down is Python's.** `gates._ask_with_deadline` runs the
injected `approve()` callable on its own thread and abandons it at the deadline.
A voice handler that hangs, a model that never calls back, and a human who says
nothing all produce the same result: no motion. A *late* yes cannot resurrect a
discarded pick — there is a test for exactly that.

**The confidence number** is `vision × (floor + (1−floor) × assignment_clarity)`,
defined once in `gates.confidence_for` with the weights and rationale in
`config.py`. Vision says "that's a red block"; the assignment margin says how sure
we are *which spot* it's on. The persona may joke about a low number but it comes
from Python and is never rounded up.

**Every run writes one gate-by-gate record** to `logs/decisions.jsonl` as a
`gated_pick` event — which gate stopped it, the confidence, the approval, the
verification. That is the judges' audit trail, and stage 7 renders exactly it.

## How to run — stage 7 (dashboard + demo day)

**The full demo script is `docs/demo_runbook.md`.** Read that on the day; this is
the summary.

```bash
# 1. Go / no-go. ~90 seconds, one table, one verdict. Run it from the SAME
#    terminal you will run the demo from (permissions belong to the app).
python scripts/preflight.py

# 2. The projector screen. Read-only — it never touches the arm.
python scripts/run_dashboard.py            # -> http://localhost:8770

# 3. The demo.
python scripts/run_agent.py

# Insurance: tell the whole gate story from a log recorded earlier.
python scripts/run_dashboard.py --replay

# Before travelling, and after any re-recording:
python scripts/backup_datasets.py
```

**The dashboard reads, it does not run.** Everything on screen is derived from
`logs/decisions.jsonl` and the last frame perception looked at — so it can only
ever show what actually happened. It deliberately does **not** open the camera:
the agent needs it, and two processes fighting over one C920 mid-demo is a risk
with no upside. Built on the standard library's `http.server`, because the
morning of a demo is the wrong time to pip-install a web framework.

**Preflight is where the demo-day failures live.** The two that have actually
bitten during this build:

- **Gemini quota** — the free tier is 20 requests/day/model and one gated pick
  spends 3–4. Preflight makes a real call and goes RED on 429, and WARNs if the
  primary model is spent and you are running on the fallback.
- **Kill-switch permission** — on macOS an untrusted process starts a key
  listener happily and then never receives an event, so the ESC kill switch and
  push-to-talk are both silently dead. Preflight checks `IS_TRUSTED`, not just
  "did the listener start".

**Back up the datasets.** `armani_gestures` and the pick macros live in the
HuggingFace cache *outside the repo*; nothing in git protects them and
re-recording needs the arm, the leader and half an hour you will not have at a
venue. `scripts/backup_datasets.py` copies both into
`armani/data/dataset_backup/` (gitignored — put it on a USB stick).

## How to run — stage 8 (the face)

Two screens ship, and they do different jobs. Run both.

| screen | job | command |
|---|---|---|
| **dashboard** | the **proof** — gates, confidence, audit trail | `python scripts/run_dashboard.py` |
| **avatar** | the **delight** — the mascot, mirroring what it's doing | `python scripts/run_avatar.py` |

```bash
python scripts/run_avatar.py                  # -> http://localhost:8771
python scripts/run_avatar.py --host 0.0.0.0   # + prints a LAN URL for a phone
python scripts/run_avatar.py --port 9001
```

The face shows five states — **idle · listening · thinking · talking · doing** —
polled 8×/second from `/state`. The agent publishes each transition at moments
that already existed: PTT key-down, key-up, the first audio chunk of a reply,
and the motion worker starting and stopping a job. Text mode and `--no-motion`
publish the same states, so the screen still animates without a mic or an arm.

**Preview it with no backend:** `http://localhost:8771/?demo=1` runs the canned
timeline, for showing the face when nothing else is running.

**It is a mirror, not a controller.** `armani/uistate.py` writes
`logs/ui_state.json` atomically (temp file + `os.replace`), never raises, and
dedupes — so publishing `talking` on every audio chunk costs one write per real
transition. If the file is missing, corrupt, or **stale** the reader says
`idle`; a session that dies mid-sentence must not leave the mascot talking to an
empty room.

**The on-screen button is a visual affordance only.** The avatar server is a
separate process from the voice session with no handle on the realtime socket,
so it cannot open the mic — the **spacebar** does, through the agent's own
global listener. Wiring the button would mean new IPC into `run_agent.py`,
alongside the push-to-talk path the whole demo depends on. Not a trade worth
making in a presentation stage.

### Gestures

Eight macros replayed from one local teleop dataset, one episode each:
`bow, wave, dance, nod_yes, shake_no, look_around, celebrate, sad_droop`.

Frames are streamed at the recorded fps rather than re-interpolated — a recording
is a performance a human already gave safely, and re-timing it destroys the gesture.
They are still clamped, with the `recorded` profile. Verified frame-accurate: 280
frames replay in 9.3 s against 9.3 s recorded, +0.01 s drift.

A recording whose frame-to-frame jump exceeds `MAX_FRAME_DELTA` (8°) is **refused at
load time**, because lerobot would clip it at send time and play something nobody
recorded.

### Improvise

Claude returns JSON keyframes; nothing about them is trusted. They are parsed
defensively (fences and prose stripped), validated hard (≤8 keyframes, 0.3–5 s each,
known joints only, no unexpected keys, numbers only), clamped to the **`policy`**
profile — the conservative envelope, because this is LLM-originated motion — and then
run as ordinary interpolated `goto`s inside `SafeMotion`. A rejected plan earns one
retry with the error attached, then gives up cleanly.

### Individual smoke tests

Every test is standalone and every one accepts `--dry-run`.

| Test | What it proves | Needs |
|---|---|---|
| `pytest tests/` | clamp/interpolate/envelope + improvise validator | nothing (pure logic) |
| `tests/smoke_01_ports.py` | follower connects, reports all 6 joints (commands no motion) | arm plugged in, **operator confirms** |
| `tests/smoke_02_wiggle.py` | one joint moves ±5° and returns (**MOTION**) | **operator watching the arm** |
| `tests/smoke_03_camera.py` | a 640x480 frame from the C920 | C920 on its tripod |
| `tests/smoke_04_mic.py` | 2 s record + playback, rejects silence | headset selected as input |
| `tests/smoke_05_keys.py` | all three APIs answer | network, `.env` filled in |
| `tests/smoke_06_ptt.py` | global spacebar hold/release | Input Monitoring granted |
| `tests/smoke_07_gestures.py` | all 8 gestures load and are playable | recorded dataset |
| `tests/smoke_08_agent.py` | agent builds, 6 tools register, one text turn answers | `OPENAI_API_KEY` (dry-run) |

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
  config.py      limits, thresholds, ids, models — one place for every constant
  safety.py      clamp_action, interp_move, SafeMotion, kill switch, require_operator
  motion.py      connect / read_positions / goto / home   (the only lerobot boundary)
  gestures.py    recorded macro replay
  improvise.py   Claude-authored keyframes, validated and clamped
  agent.py       realtime voice agent + tool dispatch
  eyes.py        Gemini pointing + camera capture  (NEVER moves the arm)
  zones.py       taught-zone registry, pixel space only  (demo pick path)
  pick.py        pick_object: identity -> zone -> replay the recorded macro
  kinematics.py  FK/IK via placo + the SO-101 URDF   [stage-4 stretch]
  calibrate.py   pixel->robot homography, table polygon  [stage-4 stretch]
  grasp.py       hover_over, no descent                  [stage-4 stretch]
  logutil.py     JSONL decision log -> logs/decisions.jsonl
  data/          home_pose.json, zones.json, homography.json, so101_kin.urdf
tests/           smoke_01..11, each standalone, each --dry-run capable
scripts/         doctor.py, capture_home.py, define_zones.py,
                 calibrate_camera.py, run_agent.py
docs/            env_report.md, recording_gestures.md, recording_picks.md
```

**Demo path vs stretch.** The taught-zone path (`zones.py` + `pick.py` +
recorded macros) is what the demo runs on. The stage-4 homography/IK/hover path
is kept, still tested, and not built on — see CLAUDE.md, Grasp.

**Dependency direction is deliberate:** `eyes` never imports `motion`, `kinematics` or
`grasp`; `kinematics` never imports `motion`. `grasp` is the only place where seeing and
moving meet, which is where stage 6 will put the trust gates.

## Safety notes that are easy to get wrong

* The five body joints are in **degrees**; the **gripper is 0–100 percent**, not degrees.
* lerobot does **not** clamp degree targets to the calibrated range — `safety.clamp_action()`
  is the only guard against an over-travel command.
* **Home must be verified before anything drives to it.** `motion.home()` raises until
  `scripts/capture_home.py` has recorded a pose the operator physically set (safety rule 4).
* **The kill switch freezes, it never drives.** First Ctrl-C (or ESC) stops commanding and holds
  position, then asks: return to this motion's start, home (only if verified), torque off, or
  leave it. A **second Ctrl-C** is a hard abort — the arm is left exactly where it is.
* **Errors return to where the motion began**, not to home, and only the joints that actually
  moved. If nothing moved, recovery is a no-op.
* Clamping happens in `Arm.send()`, the last line before the motors, so no caller can bypass it.
* **Torque coming back on does not reset the servo's goal.** lerobot's `enable_torque` writes only
  `Torque_Enable`/`Lock`, so a servo will drive to whatever target was last written — possibly from
  a previous session. `capture_home` parks the goal at the captured pose before releasing the block,
  and `connect()` parks it immediately after connecting. A twitch *during* `connect()` itself cannot
  be prevented (lerobot re-enables torque inside `configure()` before we get control), so keep a
  hand near the arm when connecting.
* Calibration already exists and is never recreated. `motion.connect()` passes
  `calibrate=False` so lerobot's interactive recalibration can never be triggered by accident.
