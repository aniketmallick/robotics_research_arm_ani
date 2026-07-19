# STAGE 6 — Trust gates: the product. G1 seen · G2 ambiguous · G3 reachable · G4 confidence+approval+timeout · G5 verify

Re-read `CLAUDE.md`, especially **safety rule 6** (gates live in OUR Python inside the pick path — NEVER in the LLM prompt; the model cannot bypass a gate no matter what it decides) and the trust-gate roadmap. Stage 5 passed: 160 tests, taught-zone pick with four honest refusal paths, `return_home=False` locked. This stage turns those refusals into the interaction that IS the pitch: the robot clarifies, states a confidence number, asks approval when unsure, stands down on silence, and verifies its own grasp.

**This is the stage the VCs are paying for. Build it to be watched.**

## Pre-work

1. Confirm `armani/data/zones.json` is committed (rig-specific but reproducible). Remind the operator in the report to back up the gesture AND pick datasets from the HF cache — they live outside the repo and are irreplaceable at the venue.
2. Confidence for taught zones: define it in one place. `pick`/`eyes` already produce a vision confidence; temper it by assignment clarity (a barely-in-one-zone object is less certain). Put the formula in config with a comment; `CONF_APPROVAL = 0.60` is the G4 line.

## Architecture — gates in Python, dialogue through the model

The gates are an ordered, Python-enforced pipeline wrapped around `pick.pick_object`. The voice model NEVER decides whether a gate passes — it only (a) speaks the question a gate produces and (b) relays the human's answer back in. The arm does not move until the Python gate is satisfied. That separation is the whole safety story; keep it airtight and testable.

1. **`armani/gates.py` — the ordered pipeline.** `run_gated_pick(arm, object_name, *, clarify, approve, verify_vlm=True) -> GatedResult`, where `clarify(question, options)` and `approve(prompt, timeout_s)` are INJECTED callables (console versions for the smoke test; voice versions from the agent). The pipeline, each step logging `gate`, inputs, decision, and outcome to `logs/decisions.jsonl`:
   - **G1 seen** — `eyes.locate`; not seen → say so, stop. (Already in pick; surface it as a named gate.)
   - **G2 ambiguous** — two candidates for the name (`Detection.candidates` / `list_visible`) OR an object between two spots (`ZoneMatch.ambiguous`). Fire `clarify("I see two cups — the left one or the right one?", [...])`, take the human's answer, RE-RESOLVE to a single zone, and only then continue. If still ambiguous or no answer → stand down. The model relays the answer; Python does the resolving.
   - **G3 reachable** — a taught macro exists for the chosen zone (`macro_available`). None → "I don't have a taught pick for that spot," stop.
   - **G4 confidence + approval + timeout** — compute the confidence number. If ≥ `CONF_APPROVAL`: state it and proceed ("Banana, 92%. Easy."). If < threshold: state the number and fire `approve(prompt, APPROVAL_TIMEOUT_S)`. Approved → proceed. Not approved, or **no answer within 10s → STAND DOWN, no motion** (fail-closed is the default: the arm never moves without a satisfied gate, and the timeout is enforced in Python, not by trusting the model to wait).
   - **G5 verify** — after the macro, run the real VLM check (below) and announce the outcome honestly, including failure ("Hmm — I don't think I got it.").
   - Return a `GatedResult` recording which gate stopped it (or success), the confidence, the approval, and the verification — this is the audit trail.

2. **Wire G5 for real** — replace `pick.verify_held`'s `TODO(stage 6)` with a Gemini call over the re-captured frame: "Is the `<object>` held in the gripper, and gone from its marked spot? Reply JSON {held: bool, confidence: 0-1, reason}." Fold it with the gripper-closure signal (VLM wins; closure is the tie-breaker). Defensive parse like `eyes.py`. Announce verified / not-verified.

3. **Agent integration — the return-to-model dialogue pattern.** Add ONE tool `pick(object_name)` to `agent.py`, bound behind `run_gated_pick`. Because the realtime session owns the audio, implement clarify/approve as the **conversational-turn pattern**, not a blocking listen:
   - When a gate needs input, the tool returns a status the model speaks: `{"status":"need_clarification","question":...}` or `{"status":"needs_approval","confidence":0.34,"prompt":...}`. The model says it out loud; the human answers by voice; the model calls a small `answer_pick(text)` / `approve_pick(yes)` tool that feeds the answer back into the pending Python gate, which then resolves and (if satisfied) enqueues the macro on the motion worker.
   - **The 10s stand-down is Python's, not the model's:** when an approval is pending, `gates.py` arms a deadline; if `approve_pick` doesn't arrive in time, the pending pick is discarded and the robot announces the stand-down. Do NOT rely on the model to count seconds.
   - The persona may make the number funny ("34%, that's a coin toss") but the number, the gate, and the timeout come from Python. Prompt style ≠ gate law.
   - Keep `stop_motion`, the freeze kill-switch, and NO-MOTION mode working exactly as stage 3.

4. **Config:** confidence formula weights, `CONF_APPROVAL` (0.60), `APPROVAL_TIMEOUT_S` (10), G5 VLM prompt/threshold. Document new `ARMANI_*` vars.

## Smoke test — prove every gate, without needing the venue

`tests/smoke_12_gates.py` (add to doctor) — console clarify/approve callables with SCRIPTED answers so the whole pipeline is exercised with no voice and (dry-run) no arm:
- clean pick → all gates pass (mock/av high confidence).
- ambiguous ("two cups") → clarify → answer "left" → resolves → proceeds.
- low confidence → approve "yes" → proceeds.
- low confidence → **no answer → 10s (use a short test timeout) → stands down, ZERO sends.**
- unseen → G1 stops, no motion.
- G5 mismatch → macro runs but verify says not-held → reported honestly.
Assert the decision log contains one clean gate-by-gate record per run — that log is the judges' artifact, so its shape matters.
Then the live voice witness (operator): the three demo acts end-to-end.

## Constraints

- Gates in `gates.py` (Python). The LLM prompt gets NO gate logic — it only speaks questions and relays answers (rule 6). A prompt jailbreak must not be able to move the arm past a gate.
- Fail-closed everywhere: unseen, unresolved, unapproved, timed-out, or unverified all mean the arm does not (or did not usefully) act, and says so.
- No dashboard yet (stage 7). No new motion primitives — reuse the stage-5 pick and the stage-3 worker.

## Definition of done

Standard five (CLAUDE.md), plus: smoke_12 exercises all six scenarios green (including the timeout stand-down with zero sends); G5 makes a real Gemini call; the live three-act demo witnessed by the operator; the decision log shows a clean gate-by-gate audit trail for each run. Commit `stage 6: trust gates`. Four-part report — and confirm the one invariant in writing: there is no code path where the model's output alone moves the arm past a gate. Stage 7 (dashboard + demo hardening) renders exactly the log this stage produces.
