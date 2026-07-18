# STAGE 2 — Gestures: recorded macros, verified home, improvised moves

Re-read `CLAUDE.md` first — safety rules 2, 4 and 7 changed after the stage-1 review (clamp profiles, return-to-entry recovery, freeze-first kill switch). Stage 1 passed review; the items below under Pre-work are the review's required amendments. Do them before the new feature work.

## Pre-work: review amendments (~45 min)

1. **Recovery semantics (updated rules 4 & 7).**
   - `SafeMotion` captures the arm's pose on entry. On exception it returns THERE (a zero-length move if nothing moved yet) — never to `HOME_POSE`. Delete the home-on-error path.
   - `goto`'s kill-switch path: stop commanding, hold position, then interactive prompt: `[s]` return to this motion's start, `[h]` home (only offered when home is verified, see task 3), `[t]` torque off (operator supports the arm first), `[l]` leave as-is. Non-tty → hold + log, no motion. Remove the auto-home.
   - Update smoke_02's kill-switch warning text to match.
2. **Config:** `wrist_roll` policy limit → `(-150.0, 150.0)` (reviewer ruling: pure rotation, collision-free, calibrated ±180 leaves margin). Implement the two clamp profiles from rule 2: `clamp_action(action, profile="policy")` and `"recorded"` = physical limits minus 2° margin. Every existing call keeps `policy`.
3. **`tests/test_safety.py`** (pytest, ≤ ~120 lines, no hardware imports): clamp — NaN/inf/unknown-joint rejection, boundary clamping, both profiles; interp_move — final step lands exactly on target, speed limit stretches step count correctly, envelope refusal fires, zero-delta yields one step. Timebox 30 minutes; these tests guard the only code standing between a bad number and the motors.

## Stage work

4. **`scripts/capture_home.py`** — the operator physically poses the arm, we record it as the verified home. Connect, then inside `bus.torque_disabled()` stream live joint positions at ~2 Hz while the operator moves the arm by hand to a good resting pose (upright-ish, gripper mid, nothing over the table edge); ENTER captures, torque re-enables, pose is written to `armani/data/home_pose.json` as `{"pose": {...}, "verified": true, "captured_at": ...}`. `config` loads this file when present (else keeps the placeholder with `HOME_VERIFIED = False`). Homing anywhere in the codebase requires `HOME_VERIFIED`. Operator-gated, dry-run supported.
5. **`armani/gestures.py`** — replay recorded teleop episodes as named macros.
   - Config: `GESTURES = {"bow": 0, "wave": 1, "dance": 2, "nod_yes": 3, "shake_no": 4, "look_around": 5, "celebrate": 6, "sad_droop": 7}` mapping name → episode index in one local dataset (`armani_gestures`), plus the dataset root path.
   - Load the action stream with lerobot 0.5.2's `LeRobotDataset` (local, no hub). If the dataset API fights you for >20 min, fall back to invoking the stock replay CLI per episode via subprocess — still behind `require_operator`.
   - `play_gesture(arm, name)`: pre-position to the episode's first action frame with a slow `goto` (2s), then stream frames at the recorded fps through `arm.send` with the `recorded` clamp profile, checking the stop flag every frame, then `goto` back to home (verified) or the episode's last frame.
   - `list_gestures()` for later use by the voice agent.
6. **`docs/recording_gestures.md`** — the operator's recording runbook, then STOP for the human. First verify which record entry point lerobot 0.5.2 actually ships (`lerobot-record --help`, else `python -m lerobot.record --help`) and write the exact command: follower + leader ports/ids from `.env`, camera optional (gestures don't need vision), fps 30, `episode_time_s` ≈ 10 with early-stop, 8 episodes in the exact `GESTURES` order, local-only (no hub push). Recording rules for the operator: every gesture STARTS and ENDS at the captured home pose (this is what makes replays chainable and pre-positioning trivial); slow and exaggerated reads better than fast; re-record a bad take immediately (left-arrow) rather than accepting it.
7. **`armani/improvise.py`** — Claude Sonnet writes novel moves, we trust nothing.
   - Prompt Claude (model from config) for strict JSON: `[{"pose": {joint: degrees...}, "seconds": s}, ...]`, ≤ 8 keyframes, each `0.3 ≤ seconds ≤ 5`, subset of known joints only. Parse defensively (strip fences, first `{`/`[` to last), validate schema hard (unknown key/joint/type → reject with the error, one retry with the error appended, then give up gracefully).
   - Execute as sequential `goto`s under the `policy` clamp profile inside `SafeMotion`.
   - `scripts/improvise_cli.py "do a slow clap" --dry-run` prints the validated, clamped plan without hardware; live execution operator-gated.
8. **`tests/smoke_07_gestures.py`** — dry-run: dataset present → load every configured episode, print frame counts and first/last poses, verify all frames pass the `recorded` clamp without alteration; dataset absent → SKIP pointing at the runbook. Live: operator replays `bow` once. Add to doctor.

## Constraints

- No voice, no Realtime API, no vision (stages 3–4). No changes to eyes/grasp/gates.
- Anthropic key check: when the operator has added the key, confirm the configured `ANTHROPIC_MODEL` id actually answers (smoke_05 covers it) before building improvise on top; if the id is rejected, list available models and pick the closest Sonnet — record the change in the report.
- Motion beyond dry-run only via the operator: capture_home, gesture recording, and the live bow replay are theirs.

## Definition of done

Per CLAUDE.md (all five steps) plus: pytest green; smoke_07 dry-run green against the recorded dataset; capture_home produces a verified home_pose.json; recording runbook delivered and the 8 episodes recorded by the operator; live bow replay witnessed. Commit `stage 2: gestures + verified home + improvise`. Four-part report; flag anything about 0.5.2's dataset/replay API that surprised you — stage 3 builds on this layer.
