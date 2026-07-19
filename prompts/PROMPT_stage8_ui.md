# STAGE 8 — The face: wire the avatar UI to the live agent

Act as a distinguished engineer and Re-read `CLAUDE.md`. This is a **presentation-only** stage: build the operator-facing voice screen (the animated mascot), driven by the agent's real state. **Touch NO gate, safety, motion, or perception logic** — the Stage-6 invariant and the whole pipeline stay byte-identical. This stage adds a state publisher and a static server, nothing more. Keep the audit dashboard (stage 7) too — it's the *proof* screen; this is the *delight* screen. Both ship.

A working, self-contained prototype is already in the repo at **`armani/data/avatar.html`** (light-green glassmorphism, an original sapling mascot — deliberately NOT Marvel Groot for IP reasons; keep it original). It currently simulates the reply cycle. Your job is to feed it the agent's *real* state and serve it. Do not redesign it; wire it.

## The five states the UI shows

`idle` · `listening` · `thinking` · `talking` · `doing`. The agent already has every one of these signals — you're publishing them, not inventing them.

## What to build

1. **`armani/uistate.py` — the state publisher.** `publish(state, **fields)` writes `logs/ui_state.json` **atomically** (write temp + `os.replace`, so the poller never reads a half-written file): `{"state": "...", "object": "...", "text": "...", "ts": ...}`. Cheap, no deps, never raises (a UI-state write must never take down the voice loop — wrap and swallow like `logutil`). A `current()` reader for the server.

2. **Publish transitions from the agent** (`run_agent.py` / `agent.py`) — map to the moments that already exist:
   - **listening** — on PTT key-down (mic opens). You already detect this in the voice input handler.
   - **thinking** — on PTT key-up / commit, before the first audio arrives.
   - **talking** — on the first `audio` event of a response (in `_pump_events`).
   - **doing** — when a motion job starts on the worker (gesture, improvise, or a gated pick's `perform`); include the object/action in `fields`.
   - **idle** — when the turn ends and the worker is not busy.
   These are one-line `uistate.publish(...)` calls at points that already fire. Do not restructure the agent; just annotate it. In `--text` and `--no-motion` modes, publish the same states so the screen still animates.

3. **`scripts/run_avatar.py` — serve it** (mirror `run_dashboard.py`: stdlib `http.server`, localhost default). Routes: `/` → `armani/data/avatar.html`; `/state` → the JSON from `uistate.current()` (send `Cache-Control: no-store`). Add `--host 0.0.0.0` support so a phone on the same wifi can open it (print the LAN URL when bound non-local) — this is how you get it on a second screen or a projector. `--port` from config.

4. **Wire the page to `/state`** — in `avatar.html`, replace the `runPreviewReply()` simulation with a poll: `fetch('/state')` ~8×/sec, and `set(s.state)` on change. Keep the simulation behind `?demo=1` so the page still previews with no backend. Keep the spacebar + button handlers exactly as they are — the spacebar already drives the real agent through its global listener; the on-screen button's REAL HOOKs stay no-ops unless you do the stretch below.

5. **(Stretch, only if the spacebar path is solid and time remains) `/ptt` endpoint** so the on-screen button can drive the mic on a touch device with no keyboard. POST `/ptt {down|up}` → inject a press/release into the same path the pynput listener uses. Skip if it risks the working spacebar flow; the button can stay a visual affordance for the demo.

## Constraints

- Presentation only. No changes to `gates.py`, `safety.py`, `motion.py`, `pick.py`, `zones.py`, `eyes.py`'s logic. If you find yourself editing a gate, stop.
- The state file is best-effort telemetry: if it's missing or stale, the UI shows `idle`, never crashes.
- Don't open the camera or the arm from the avatar server — it's a read-only mirror, like the dashboard.

## Definition of done

Standard five (CLAUDE.md), plus: `run_avatar.py` serves the page; running the real voice session drives the mascot through listening→thinking→talking→doing→idle in the browser (operator-witnessed); `?demo=1` still previews standalone; a phone on the same wifi can open the LAN URL. A couple of tests for `uistate` (atomic write, missing-file → idle). Commit `stage 8: avatar UI wired to agent state`. Re-tag the freeze. Four-part report — and confirm in writing that no gate/safety/motion file changed (a one-line `git diff --stat` against the demo-freeze tag).
