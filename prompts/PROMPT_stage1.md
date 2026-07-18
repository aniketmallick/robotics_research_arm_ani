# STAGE 1 — Foundation & Hardware Smoke Tests
Act as a distinguished engineer and Read `CLAUDE.md` in the repo root fully before anything else. It is your constitution: safety rules, locked stack, definition of done. This stage builds the skeleton and PROVES every hardware and API dependency works, before any feature code exists.

## Goal

By the end of this stage, `python scripts/doctor.py` walks the operator through every dependency check, and we know with certainty: ports work, the arm responds, the camera and mic work, all three API keys are live, and macOS permissions are granted. Nothing else.

## Tasks (in order)

1. **Repo init**: `git init`, `.gitignore` (`.env`, `__pycache__/`, `logs/`, `tests/out/`, `.venv` if created), `README.md` skeleton, directory layout per CLAUDE.md, empty `logs/`.

2. **Environment discovery** → write findings to `docs/env_report.md`:
   - Which Python env currently runs teleop successfully; its Python version; `lerobot` version and how it was installed. USE THIS ENV for the project — do not create a parallel one, do not upgrade lerobot.
   - Existing robot ids + calibration files under `~/.cache/huggingface/lerobot/calibration/` (list them; touch nothing).
   - Serial ports present (`ls /dev/tty.usbmodem*`) — with the operator's help, identify which is leader, which is follower (unplug/replug method).
   - Available camera indices via OpenCV (AVFoundation backend) and which one is the C920.
   - Then install ONLY: `openai-agents`, `google-genai`, `anthropic`, `python-dotenv`, `opencv-python`, `sounddevice`, `soundfile`, `pynput` — into that same env. If pip reports a dependency conflict with lerobot, STOP that install and record it in the report; do not force.

3. **`armani/config.py`**: follower/leader ports, robot ids, camera index + 640x480@30, joint clamp table (base ±90°, others ±60° — refine later from calibration ranges), placeholder home pose, placeholder table polygon, thresholds (`CONF_APPROVAL = 0.60`, `APPROVAL_TIMEOUT_S = 10`), `DRY_RUN` flag read from env. `.env.example` with the three key names.

4. **`armani/safety.py`**: `clamp_action(action) -> action`, speed-limited interpolation helper (`interp_move(current, target, duration, hz=25)` yielding intermediate actions), `SafeMotion` context manager (any exception → slow home + log), kill-switch registration (ESC/Ctrl-C → stop + slow home), `require_operator()` (prompts "Operator present and watching? [y/N]" before any motion; auto-yes only when `DRY_RUN`).

5. **`armani/motion.py`** (minimal for this stage): `connect() -> SO101Follower` using existing calibration id, `read_positions()`, `goto(action, duration)` = clamp + interpolate + send_action loop, `home(slow=True)`. Everything honors `DRY_RUN` (prints instead of sending).

6. **Smoke tests** — each standalone, each supports `--dry-run`, motion ones call `require_operator()`:
   - `tests/smoke_01_ports.py` — connect follower, print live joint positions.
   - `tests/smoke_02_wiggle.py` — OPERATOR: one joint ±5° and back, then slow home. Nothing bigger.
   - `tests/smoke_03_camera.py` — grab frame, save `tests/out/frame.jpg`, print actual resolution/fps.
   - `tests/smoke_04_mic.py` — record 2s from default input, save wav, play it back.
   - `tests/smoke_05_keys.py` — minimal live checks: OpenAI (open+close a realtime session or list models), Gemini (send `tests/out/frame.jpg`, ask "list the objects you see" — print reply; try `gemini-robotics-er-1.6-preview`, fall back per CLAUDE.md and RECORD which model actually worked), Anthropic (1-token ping). Print cost-free/cheap confirmations, never print the keys.
   - `tests/smoke_06_ptt.py` — pynput global spacebar listener: hold/release detection; on failure print the exact macOS Accessibility/Input Monitoring toggle to flip.
   - `scripts/doctor.py` — runs 01→06 in order, interactive, clear PASS/FAIL summary table at the end.

## Hard constraints

- NO feature code (no gestures, no voice loop, no vision pipeline, no IK). That's stages 2–6.
- NO motion beyond smoke_02's ±5° wiggle and slow home.
- Do not modify anything under `~/.cache/huggingface/lerobot/`.
- If the C920 index or ports can't be determined without the human, write the test to prompt them interactively rather than guessing.

## Definition of done

Per CLAUDE.md (all five steps), plus stage-specific: doctor.py completes with the operator present and all six checks PASS (or documented-with-reason), `docs/env_report.md` filled in, commit `stage 1: foundation + smoke tests`. Then produce the four-part report (WHAT I BUILT / HOW I TESTED / KNOWN LIMITATIONS / QUESTIONS FOR REVIEWER) — the human will paste that report plus your code to the external architect for review before Stage 2 is issued.
