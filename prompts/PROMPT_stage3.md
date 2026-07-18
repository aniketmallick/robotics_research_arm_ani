# STAGE 3 — Voice brain ⭐ MILESTONE: the talking, roasting, bowing robot

Re-read `CLAUDE.md` (rule 8 now states the ratified improvise budget: ≤8 keyframes, ≤5s each, ≤15s total — your interpretation was approved, no change needed). Stage 2 passed review; all 46 unit tests were independently re-run and passed. This stage wires the personality on top of the proven motion layer.

## Pre-work (~10 min)

1. `git add armani/data/home_pose.json` and commit — it is not a secret, and losing it silently un-verifies home.
2. Delete `tests/out/review/` (architect's staging snapshot, no longer needed).
3. `clamp_action` escalates to ERROR level when `profile == "physical"` — but the send boundary uses `"backstop"`. Escalate on both ("physical", "backstop"): a send-boundary clamp firing is always an anomaly worth an ERROR.
4. Rotate the decision log: move `logs/decisions.jsonl` to `logs/decisions_dev.jsonl` and start fresh — stage 7 replays this log for judges and 900 dev-time clamp entries would bury the story.

## Architecture (follow this; deviations go in QUESTIONS FOR REVIEWER)

**New: `armani/agent.py` + `scripts/run_agent.py`.**

- **Motion worker thread.** One dedicated thread owns the arm and consumes a `queue.Queue(maxsize=1)` of motion jobs. The realtime event loop NEVER blocks on motion. Long-action tools enqueue and return immediately with `{"status": "started", "action": ..., "eta_s": ...}`; if the worker is busy they return `{"status": "busy", "doing": <current>}` — no pileup, the robot does one thing at a time. When a job finishes (or fails), the worker posts a completion event that the agent loop injects back into the session so the model can comment on the result. Every tool call and outcome → `log_event`.
- **Session.** OpenAI Agents SDK (`openai-agents` 0.18.3 is installed) RealtimeAgent/RealtimeRunner over WebSocket, model `config.REALTIME_MODEL`, voice from config. **Turn detection OFF — push-to-talk**: global spacebar via pynput (hold = stream mic, release = commit + request response), plus an ENTER-to-type text fallback in the terminal for when audio is impossible. Pressing space while the model is speaking = barge-in: cancel/interrupt via the SDK and flush local playback.
- **Audio.** `sounddevice` for mic capture and speaker playback at the rates the SDK expects — read the installed SDK's own realtime/audio example to get formats right rather than guessing. PTT means mic and speaker are naturally half-duplex; no echo cancellation needed.
- **Reality check first:** the SDK is installed at a specific version — read its actual API surface (module source / bundled examples) before coding against it. If the SDK's realtime layer fights you for >20 min, fall back to a manual WebSocket session via the `openai` client per the realtime docs (PCM16 audio, function calling) and record which path shipped.
- **PERSONA** constant in `agent.py`. ARM-ANI: Gen-Z, deadpan, straightforward; roasts the user when they slip (lovingly, never mean); replies ≤2 sentences unless asked for more; no emoji in speech. Hard rules inside the prompt: NEVER claim abilities it lacks — it has NO vision until stage 4, so `look()` questions get an honest "eyes arrive in the next build"; always announce motion BEFORE it starts; when a tool returns busy/refused/error, own it plainly and move on. Personality is style — the tools are law.
- **Tools** (each validates args, logs to the decision log, returns compact JSON):
  - `list_gestures()` — instant.
  - `play_gesture(name)` — unknown name or unrecorded episode → return the exact error text so the model can riff honestly ("bow's in my repertoire; 'backflip' is not").
  - `improvise_move(description)` — run `request_plan` off-thread (it's a network call), then enqueue `perform`; return started + keyframe count; `ImproviseError` text goes back to the model verbatim.
  - `go_home()`, `get_status()` (pose summary, busy/idle, gestures available).
  - `stop_motion()` — LLM-invoked stop: worker aborts the current job and the arm HOLDS. It does NOT open the freeze menu — that menu belongs to the human kill switch only (rule 7). Return "stopped and holding".
- **Kill switch vs agent console.** ESC/Ctrl-C still freezes per rule 7 — the freeze menu prompts on stdin, so the agent loop must pause its own stdin/PTT handling while `handle_freeze` owns the terminal, then resume cleanly.
- **Operator gate.** At startup, `require_operator("start the voice session with motion enabled")` ONCE. Declined → NO-MOTION mode: session still runs, motion tools return `{"status": "refused", "reason": "motion not enabled"}` — personality still demos.
- **Config additions:** realtime voice name, PTT key, audio device override env vars, response token cap (keep replies snappy and cheap).

## Smoke test

`tests/smoke_08_agent.py`:
- `--dry-run`: build the agent with a DryRunArm, verify every tool schema registers, run ONE text round-trip ("introduce yourself in one sentence") and print the reply. No audio, no motion. Add to doctor.
- Live (operator): scripted checklist printed at start — introduce yourself → list gestures → "take a bow" (verify it TALKS WHILE THE ARM MOVES) → "improvise a tiny robot dance" → "pick up the banana" (must honestly decline: no eyes yet) → interrupt it mid-sentence with spacebar → "stop" mid-gesture → Ctrl-C freeze drill → [s] recovery.

## Constraints

- No vision, no gates, no dashboard (stages 4–7). No changes to safety/motion semantics beyond pre-work item 3.
- Wired-headset reminder printed at session start if the default input device name contains "AirPods" or "MacBook Pro Microphone".
- Never print API keys; log token usage events if the SDK exposes them.

## Definition of done

Standard five (CLAUDE.md), plus: smoke_08 dry-run green; live voice session witnessed by the operator with talk-while-moving demonstrated; README updated with `scripts/run_agent.py` instructions. Commit `stage 3: realtime voice agent`. Four-part report — flag every SDK surprise: stage 6 wires trust gates INTO these tools, so the tool-result plumbing you build here must be clean.
