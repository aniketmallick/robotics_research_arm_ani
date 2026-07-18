# ARM-ANI — Voice-Interactive SO-101 with Trust Gates

Project context for Claude Code. **Read this file fully before doing any work. Re-read it after any context compaction.**

## What we are building (hackathon demo, ~18h)

A Gen-Z-personality robot arm ("ARM-ANI"): you talk to it (push-to-talk), it talks back (deadpan, roasts you when you slip), performs expressive gesture macros and improvised moves, picks named objects it sees through a fixed C920 camera, and — the core differentiator — runs every risky action through **trust gates**: clarifies when ambiguous, states a confidence number, requires spoken approval below threshold, stands down on 10s silence (fail-closed), and verifies its own grasp afterward. A live dashboard shows the robot's "mind" (camera + detections + confidence + which gate fired).

Demo = 3 acts: (1) banter + gestures + one improvised move, (2) clean pick with confidence stated, (3) ambiguous request → asks "which one?"; low-confidence request → asks approval; no approval → stands down.

## Non-negotiable safety rules

1. **No motor motion unless the operator has explicitly confirmed they are present and watching.** Motion test scripts must prompt and wait. Never auto-run motion in a test suite.
2. Every joint target passes through `safety.clamp_action()`. Two clamp profiles: **`policy`** — base ±90°, other joints ±60°, wrist_roll ±150° (pure rotation, collision-free; calibrated range is ±180°), gripper 0–100% — used for every LLM/IK-originated target; **`recorded`** — the calibrated physical range minus a 2° margin — used only when replaying human teleop recordings, which are ground truth and may legitimately exceed policy limits. Both profiles share the max-speed limit. All motion is interpolated (20–30 Hz), never a raw jump to target.
3. Workspace check before any IK-driven move: target (x, y) must be inside the calibrated table polygon.
4. Any exception mid-motion → controlled return to the pose where that motion began (captured on entry; always known-reachable — the arm was just there, and it retraces the corridor it already traversed safely). Log it. Auto-driving to `HOME_POSE` is forbidden until it is hardware-verified via `capture_home`; after verification, home becomes the standard recovery target.
5. Every hardware-touching feature has a `--dry-run` mode that prints what it *would* do.
6. **Trust gates live in our Python code inside `pick()` — never in the LLM prompt.** The LLM cannot bypass a gate no matter what it decides; prompts are style, gates are law.
7. Kill switch: first ESC / Ctrl-C → **freeze** — stop commanding new targets, hold position, then prompt the operator (return-to-start / home-if-verified / torque-off / leave). Never auto-drive anywhere from the kill switch: the operator is present (rule 1) and chooses. Second Ctrl-C → hard abort, arm left exactly where it is. Always registered when motors are connected.
8. LLM-generated motion (improvise_move) is never trusted raw: strict JSON schema, validated, clamped, max 8 keyframes, max 5s per move.

## Hardware & machine facts

- **MacBook Pro (macOS).** Serial ports are `/dev/tty.usbmodem*` (NOT ttyACM, no chmod needed).
- **SO-101 leader + follower, ALREADY CALIBRATED — teleop works today.** Do NOT recreate calibration. Discover existing robot ids and calibration under `~/.cache/huggingface/lerobot/calibration/`. **Do NOT upgrade or reinstall lerobot** — use the exact env/version that already runs teleop; discover it and record it in `docs/env_report.md`.
- Logitech C920 on a locked tripod. Run at **640x480@30fps** (1080p chokes bandwidth for zero benefit). OpenCV with the AVFoundation backend. **The camera must never move after homography calibration** — if bumped, recalibration is required (~5 min).
- macOS permissions the terminal app needs: Camera, Microphone, Accessibility + Input Monitoring (for the pynput global spacebar listener). Smoke tests must detect missing permissions and tell the operator exactly which System Settings toggle to flip.
- Demo mic = wired headset, not the C920 mic.

## Locked stack decisions (do not relitigate)

- **Voice + brain:** OpenAI Agents SDK (`openai-agents`) Realtime over WebSocket, model `gpt-realtime-2.1`. Push-to-talk spacebar (`turn_detection` off). Persona = system prompt constant `PERSONA` in `agent.py`. Tools return `{"status": "started"}` immediately for long actions so the model keeps talking while the arm moves.
- **Eyes:** `google-genai` SDK, model `gemini-robotics-er-1.6-preview` (fallback order: `gemini-robotics-er-1.5-preview`, then a current Gemini flash model with bounding boxes). Points come back normalized 0–1000 as [y, x].
- **Optional second detector:** OWLv2 via `transformers` ONLY if it installs cleanly into the existing lerobot env (dependency conflict risk). If it conflicts: skip it — confidence then = Gemini dual-query agreement + IK reachability margin + MCQA logprob check (OpenAI). Do not fight pip for more than 20 minutes.
- **improvise_move:** `anthropic` SDK (Claude Sonnet), strict JSON keyframes → validated → clamped. See safety rule 8.
- **Motion:** lerobot **0.5.2** Python API (conda env `miniforge3/envs/lerobot`, Python 3.12.13, editable install — never upgrade or add a parallel env). `SO101Follower` is an alias of `SOFollower`; calibration lives under `calibration/robots/so_follower/`. **Units: body joints in DEGREES (`use_degrees=True`, pinned in config), gripper always 0–100 percent — NOT normalized -100..100.** lerobot does not clamp degree targets itself, so `safety.clamp_action()` is the only guard; keep `max_relative_target` set as defence-in-depth. Gestures recorded by the operator with the 0.5.2 record CLI (verify the entry point name before use); replayed by our wrapper reading the dataset's action stream (subprocess replay is the fallback path).
- **Pixel→robot:** plane homography. Primary: ChArUco board (adapt `google-gemini/robotics-pointing-sample`, Apache-2.0). Fallback: gripper-tip 6-point method (jog tip to 6 spots, pair robot XY from kinematics with tip pixels, `cv2.findHomography`).
- **Grasp:** top-down scripted — hover 10cm over target → descend to grasp height → close gripper → lift. Fixed Z from table height + per-object height in config.
- **IK ladder:** robotics-pointing-sample approach / lerobot `RobotKinematics` (Placo + `so101_new_calib.urdf`) → Plan B: phosphobot REST `/move/absolute` → Plan C: taught zones (5 marked spots, one recorded pick macro each; VLM only chooses the zone). Plan switch decisions belong to the human architect, not you.
- **Config:** `config.py` + `.env` via python-dotenv. Keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`. `.env` is gitignored, never committed, never printed.
- **Decision log:** every tool call, gate evaluation, confidence score, approval, and outcome → append JSONL to `logs/decisions.jsonl`. This feeds the dashboard and is our "audit trail" artifact for judges.

## Repo layout

```
armani/            # package
  config.py safety.py motion.py gestures.py improvise.py
  eyes.py calibrate.py grasp.py gates.py agent.py dashboard.py logutil.py
tests/             # smoke_01_ports.py ... numbered, standalone, each with --dry-run
scripts/doctor.py  # runs all smoke tests in order with operator prompts
docs/env_report.md
logs/  README.md
```

## Working agreement (how you, Claude Code, operate)

- Work arrives as numbered **stage prompts** from the external architect (who also reviews your code after each stage). Do ONLY the current stage. No scope creep into future stages, no "while I'm here" refactors.
- Before coding a stage: restate your plan in ≤5 lines.
- **Definition of Done — every stage, no exceptions:**
  1. All stage tests / dry-runs pass on your machine.
  2. **Self-review: re-read every file you wrote or changed, end to end, as a hostile reviewer.** Check: safety rules honored, error paths handled, nothing hardcoded that belongs in config, no dead code, imports clean. Fix what you find.
  3. `README.md` "How to run" section updated for the stage.
  4. Git commit with message `stage N: <summary>`.
  5. Final report in exactly this format: **WHAT I BUILT / HOW I TESTED IT** (including which tests need the operator + hardware) **/ KNOWN LIMITATIONS / QUESTIONS FOR REVIEWER**.
  You never say "done" without all five. If something doesn't work, you say so plainly instead of papering over it.
- Anything requiring motors, camera position, or physical objects: write the test, then STOP and hand it to the operator with exact run instructions. Never simulate a hardware pass.
- Style: boring, readable code. Small functions, type hints, one module one job. No premature abstraction, no framework ceremony. Comments only where the *why* is non-obvious.
- If blocked >20 min on any single issue: stop, write up the blocker precisely, move to the next task in the stage, flag it in QUESTIONS FOR REVIEWER.

## Stage roadmap

1. **Foundation & smoke tests** — repo, config, safety core, every hardware/API dependency proven.
2. **Gestures** — record/replay macros + keyframe engine + improvise_move.
3. **Voice brain** — Realtime agent, PTT, persona, tools wired. ⭐ MILESTONE: talking, roasting, bowing robot.
4. **Eyes + calibration** — Gemini pointing, homography, "hover over the named object" proof.
5. **Grasp** — full pick pipeline on 5 objects.
6. **Trust gates** — G1 seen / G2 ambiguous / G3 reachable / G4 confidence+approval+timeout / G5 verify.
7. **Dashboard + demo hardening** — mind screen, kill switch drill, decision log replay, code freeze.

## Key references

- Google SO-101 pointing sample (pattern for Gemini+homography+IK): https://github.com/google-gemini/robotics-pointing-sample
- Gemini robotics docs: https://ai.google.dev/gemini-api/docs/robotics-overview
- OpenAI Agents SDK realtime: https://openai.github.io/openai-agents-python/realtime/quickstart/
- LeRobot SO-101: https://huggingface.co/docs/lerobot/so101 · record/replay: https://huggingface.co/docs/lerobot/il_robots · IK/kinematics: https://huggingface.co/docs/lerobot/phone_teleop
- SO-101 URDF: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
- Safety/validation pattern to imitate (MIT): https://github.com/cyberwave-os/cyberwave-python → `examples/nl_arm_controller/motion.py`, `planner.py`
- Architecture-to-copy for motion thread + tool dispatch: https://github.com/pollen-robotics/reachy_mini_conversation_app
- phosphobot (Plan B IK): https://github.com/phospho-app/phosphobot
