# STAGE 7 — Dashboard + demo hardening (the final stage)

Act as a distinguished engineer and Re-read `CLAUDE.md`. Stage 6 passed review — 214 tests (all re-run independently, including the two that matter most: the model-never-answers stand-down and the late-approval-rejected path), and the invariant holds: the model's only inputs are a clarify string and an approve bool, both re-checked in Python, and the sole `worker.submit` is reachable only after G1–G4. This stage makes it a demo that wins: the screen judges watch, and the hardening so nothing dies on stage. **Do NOT change any gate, safety, or motion logic — this stage is presentation + robustness only.**

## Pre-work — the two demo-day survival items (do first, flag loudly in the report)

1. **Billing on the Google API key.** The free tier (20 req/day/model) is exhausted and is a hard demo blocker — G1 can't run, so no pick can even start. This is operational (the operator enables pay-as-you-go in Google AI Studio); cost is fractions of a cent per call. Nothing ships until this is confirmed. Add a check to the pre-flight (below) that makes one cheap Gemini call and FAILS RED if it 429s.
2. **Back up the datasets.** `armani_gestures` (8 eps) and `armani_pick_place` (5 eps) live in `~/.cache/huggingface/lerobot/` — outside the repo, irreplaceable at the venue. Add `scripts/backup_datasets.sh` (or a Python util) that copies both dataset dirs into `armani/data/dataset_backup/` (gitignored) with a timestamp, and a restore note in the README. Run it.

## What to build

1. **`armani/dashboard.py` + `scripts/run_dashboard.py` — the "robot's mind" screen.** A local web page (FastAPI/Flask + a single self-contained HTML/JS page, or a clean OpenCV window if a browser is too much — your call, browser preferred for judges) that shows, live:
   - the C920 frame with the detected object's point/label and the marked zones drawn on it;
   - the current/last **gated pick**: object, chosen zone, the **confidence number** as a bar against the `CONF_APPROVAL` line, and **which gate fired** (G1→G5, with the STOP point highlighted);
   - a scrolling feed of the decision log (`logs/decisions.jsonl`) — the `gated_pick` records rendered as a gate-by-gate audit trail.
   It READS the decision log the agent already writes (tail it); it does not re-run anything or touch the arm. If it can share the persona's colour language (seen/ambiguous/approved/verified), even better. This is the artifact judges stare at — make it legible from 3 metres.

2. **`scripts/preflight.py` — the 90-second go/no-go before you present.** One script, clear PASS/FAIL table: Gemini billing live (a real cheap call, red on 429), all three API keys, zones.json present + 5 macros in the pick dataset + 8 gestures, home verified, camera index reachable at 640×480, mic default = the wired headset (warn on AirPods/built-in), Accessibility + Input Monitoring granted (ESC kill switch), decision log writable. Reuse the smoke tests' checks; don't reinvent them.

3. **Decision-log replay (`scripts/replay_log.py` or a dashboard mode)** — feed a past run's `decisions.jsonl` through the dashboard so you can show the full gate story even if the live call flakes. This is your insurance if venue wifi or quota misbehaves mid-pitch.

4. **Demo hardening in the agent:** a handful of pre-written zero-latency persona quips cached for the slow moments (while a pick runs, while Gemini thinks); make sure a Gemini 429 / timeout surfaces to the model as an honest, in-character line ("my eyes are buffering, give me a sec") not a stack trace; confirm NO-MOTION mode and the freeze kill switch still behave. No new gate logic.

5. **`docs/demo_runbook.md`** — the exact three-act script (banter+gesture+improvise / clean pick with confidence / ambiguous→"which one?" + low-confidence→approve→stand-down), the object/zone placement per act, the preflight command, what to do if X fails (quota → replay; wifi → hotspot; overload → power-cycle), and the code-freeze checkpoint.

## Verification / definition of done

Standard five (CLAUDE.md), plus: `preflight.py` runs green with billing enabled; the dashboard renders a real `gated_pick` record live (drive it with a real or replayed log); `backup_datasets` has run and the backup exists; the three-act runbook is written; a backup video of a clean run is recorded (operator). Commit `stage 7: dashboard + demo hardening`, then **tag a code freeze** (`git tag demo-freeze`). Four-part report. After this stage, ARM-ANI is demo-ready — the last thing you do is rehearse the three acts until they're boring.

## The bar

This is what a VC sees. The arm can fail a grasp and you still win if the *screen* shows it saw the object, stated 34%, asked, and stood down when you stayed silent. Build the dashboard to make the trust story legible even when the hardware has an off day.
