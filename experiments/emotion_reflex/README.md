# Emotion → Gesture Reflex (System-1 prototype)

**Experimental spike, not part of the demo pipeline.** Proves the fast reflex
loop: *laptop webcam sees the operator's face → a small emotion model classifies
it → the arm fires a matching pre-recorded gesture* — locally, in real time, with
every safety rule intact. If it feels good, the recorded macro gets swapped for a
learned policy next. Until then: small, isolated, safe.

Nothing here imports or is imported by the demo path. It moves the arm **only**
by replaying an already-recorded, already-clamped gesture through
`gestures.play_gesture`; it generates no motion of its own.

```
detector.py     Part A — the eyes.  Own venv (cv2 + onnxruntime + numpy).
reflex.py       Part B — the hands. lerobot conda env (imports armani).
spikeconfig.py  shared, dependency-free config (paths, cam index, map, tunables)
statefile.py    the atomic emotion hand-off file (like uistate.py)
emotion_smooth.py  rolling-window majority-vote debounce (pure)
reflex_rules.py    the fire/suppress decision (pure)
tests/          hardware-free unit tests
```

The two halves are **separate processes** that talk only through
`logs/emotion_state.json`, written atomically (temp file + `os.replace`) exactly
like `armani/uistate.py`. That keeps the vision deps out of the lerobot conda env
(numpy is pinned there — never fight pip into it).

---

## One-time setup

### Part A — the detector's own venv

```bash
cd experiments/emotion_reflex
python3 -m venv .venv                     # a CLEAN system python3, NOT conda
./.venv/bin/python -m pip install -r requirements-detector.txt
./.venv/bin/python detector.py --download # fetches FER+ (~34 MB) and YuNet (~0.2 MB)
```

Verified on macOS with Python 3.9.6 → cv2 5.0.0, numpy 2.0.2, onnxruntime 1.19.2
(CoreML + CPU). The `.venv/` and `models/*.onnx` are gitignored (heavy, local).

Models:
- **FER+** `emotion-ferplus-8.onnx` (ONNX model zoo) — 8 classes: neutral,
  happiness, surprise, sadness, anger, disgust, fear, contempt. `spikeconfig`
  folds happiness→happy, sadness→sad, anger→angry, contempt→angry.
- **YuNet** `face_detection_yunet_2023mar.onnx` (opencv_zoo) — face detection.
  Optional: if absent, the detector falls back to the **Haar cascade** that ships
  with OpenCV (zero download). If YuNet misbehaves on your OpenCV build, just
  delete its `.onnx` and Haar takes over.
- *Not implemented in this spike:* the MediaPipe-blendshape fallback the brief
  mentions. FER+ is the classifier path; if it will not load, the detector prints
  the `--download` command and exits rather than silently degrading.

### Part B — the reflex uses the existing lerobot env

```bash
conda activate lerobot        # already has armani, motion, gestures, safety
```

No new deps. It reads the state file and replays gestures; it never opens a
camera.

---

## Config

All tunables live in `spikeconfig.py`, overridable by `ARMANI_REFLEX_*` env vars
(see `.env.example`). The current **emotion → gesture map**:

| emotion  | gesture       | why |
|----------|---------------|-----|
| happy    | `nod_yes`     | acknowledge the good mood |
| sad      | `celebrate`   | the "cheer up" gesture |
| surprise | `look_around` | mirror the surprise |
| angry    | `bow`         | placate / apologise |

`neutral`, `disgust`, `fear` are intentionally **unmapped** — a reflex should
stay quiet rather than fire something that does not fit. Edit with
`ARMANI_REFLEX_MAP="happy:wave,sad:bow"`; any gesture named must be one of the
recorded ones (`bow, wave, dance, nod_yes, shake_no, look_around, celebrate,
sad_droop`) — `reflex.py` drops and warns about anything else.

Key safety tunables: `ARMANI_REFLEX_COOLDOWN_S` (default 15s between gestures),
`ARMANI_REFLEX_ALLOW_REPEAT` (default off — a persistently held emotion fires
once, then waits for it to change).

---

## Bring-up order (do these in sequence)

Each step adds one moving part. Do not skip ahead — step 3 is the first time the
arm moves.

**1. Eyes only, no arm** — validate detection quality before anything can move:
```bash
cd experiments/emotion_reflex
./.venv/bin/python detector.py --watch
```
Shows the webcam with the face box, current emotion, score and FPS. Writes
**nothing**. Confirm it tracks your face, reads happy/sad/surprise sensibly, and
runs at a usable frame rate. If the wrong camera opens, note the responding
indices it printed and set `ARMANI_REFLEX_CAM_INDEX`.

**2. Full loop, no motion** — real emotions, dry-run arm:
```bash
# terminal A (detector venv):
./.venv/bin/python detector.py
# terminal B (lerobot env):
python reflex.py            # --dry-run is the DEFAULT
```
`reflex.py` prints `WOULD play <gesture> for <emotion> (score X)` for a real
detected emotion, and `suppressed (<reason>)` otherwise. No arm involved.

**3. One real gesture, operator present** — the safest first live test:
```bash
conda activate lerobot
python reflex.py --sim-emotion sad --live --once
```
`--sim-emotion` injects one emotion with no camera, so this tests only the arm
path. It prompts **`Operator present and watching? [y/N]`** and refuses on
anything but yes, arms the freeze kill switch, fires **one** gesture, and exits.

**4. Full live** — detector + reflex, the whole reflex:
```bash
# terminal A: ./.venv/bin/python detector.py
# terminal B: conda activate lerobot && python reflex.py --live
```

---

## Safety (this moves the arm — re-read CLAUDE.md)

- **Laptop webcam only, never the C920.** The C920 is locked to the arm's
  workspace homography; opening or moving it breaks calibration. `detector.py`
  uses a separate `cv2.VideoCapture` index (`ARMANI_REFLEX_CAM_INDEX`).
- **Operator-present gate.** `--live` prompts and refuses on anything but an
  explicit `y` (CLAUDE.md rule 1), via `safety.require_operator`.
- **Freeze kill switch** (`safety.install_kill_switch`) is armed whenever the arm
  is connected. Ctrl-C mid-gesture freezes and hands the operator the menu;
  between gestures it stops the loop cleanly.
- **`--dry-run` is the default.** You must ask for `--live` explicitly.
- **One gesture at a time, no backlog.** The loop is single-threaded and blocking,
  so the arm is never commanded while it is already moving, and a stale emotion is
  never queued — when the arm is free it reacts to the *current* emotion.
- Every decision (detected / fired / suppressed+reason / outcome) is appended to
  `logs/decisions.jsonl` (`reflex_decision`, `reflex_gesture_start/end`) in the
  existing record style, so the dashboard can render it.

## Tests

```bash
conda activate lerobot
pytest experiments/emotion_reflex/tests/ -q
```
Pure logic only — atomic write + missing/stale read, the debounce smoother, the
cooldown/repeat/unmapped decision, and config-map parsing. No camera, no ONNX,
no arm. `pytest`'s `testpaths=tests` keeps these out of the main suite.
