# STAGE 4 — Eyes + calibration: "hover over the named object"

Act as a distinguished engineer and Re-read `CLAUDE.md`, especially the IK ladder and safety rules 2–4. Stage 3 passed review: 46/46 unit tests still green, the agent drove real tools live (gestures, improvise, go_home all logged `done`), and the persona lands. This stage gives ARM-ANI eyes and a pixel→robot map. **It stops at HOVER — no descent, no gripper, no grasp. That is stage 5.** Keeping the arm 10cm above the table all stage means vision can be wrong without breaking anything.

## Pre-work (~10 min)

1. Commit the working tree first — `run_agent.py` and `smoke_08_agent.py` have uncommitted edits past the stage-3 commit, and `CLAUDE.md` has the ratified rule-8 wording. Commit them (`chore: commit stage-3 working-tree fixups`) so review starts from a clean tree.
2. `logs/` hygiene: confirm `decisions.jsonl` is the live one and `decisions_dev.jsonl` is the archive — leave both.

## What you're building

The pipeline from a spoken object name to the arm hovering over it: **C920 frame → Gemini points at the named object → homography maps pixel to table (X,Y) → IK to joint angles → `goto` a hover pose 10cm above.** Plus honest confidence, because stage 6 will gate on it.

1. **`armani/eyes.py` — perception, no motion.**
   - `google-genai` client, model from `config.GEMINI_MODELS` (walk the fallback list; `gemini-robotics-er-1.6-preview` is confirmed live from smoke_05). Capture one C920 frame at the pinned index (640×480, AVFoundation) — reuse the smoke_03 capture path, don't reinvent it.
   - `locate(object_name) -> Detection | None`: prompt Gemini for points on the named object; points come back normalized 0–1000 as `[y, x]` → convert to pixels. Return the point, a confidence in 0–1, and the raw reply for the log. Structured-output/JSON parse must be defensive (you already have the pattern in `improvise.py` — reuse the spirit, don't copy blindly).
   - `list_visible(candidates) -> list[Detection]`: point at several named objects in one call, for the stage-6 "which one?" disambiguation later. Build the plumbing now; don't wire dialogue yet.
   - **Confidence, honest:** primary signal = Gemini returning a point at all + its self-reported score. If OWLv2 installs cleanly into the lerobot env in ≤20 min (CLAUDE.md marks it optional, dependency-conflict risk — do NOT fight pip past the timebox), add it as a second opinion: agreement between Gemini's point and an OWLv2 box = high confidence, disagreement = low. If it won't install, say so in the report and proceed Gemini-only; confidence then also folds in IK reachability margin (below). Never fabricate a calibrated number — document exactly what the score means.
   - `eyes.py` NEVER moves the arm and imports nothing from `motion`. Pure perception.

2. **`armani/calibrate.py` + `scripts/calibrate_camera.py` — the pixel→robot map (operator-run).**
   - Plane homography, `cv2.findHomography`, saved to `armani/data/homography.json` with a timestamp and the frame size it was computed at.
   - **Primary: ChArUco/ArUco board** — adapt the approach in `google-gemini/robotics-pointing-sample` (Apache-2.0; you may vendor a small, attributed helper). Operator lays the board flat on the table, one capture builds the map.
   - **Fallback: gripper-tip 6-point** — jog the tip to ≥6 spots spread across the table; at each, read robot (X,Y) from FK and click/record the tip pixel; `findHomography` on the pairs. Offer this when no board is available.
   - **Populate `config.TABLE_POLYGON`** (currently empty → gates fail closed) from the calibrated corners, in robot metres. Save alongside the homography. Include a `homography_health` check: reproject the calibration points, report mean pixel error, and REFUSE to save if error exceeds a sane threshold (say >15px) — a bad map is worse than none.
   - Loud reminder on success: the camera and board plane must not move afterward, or recalibrate (~5 min).

3. **IK for the hover — the ladder, timeboxed.**
   - Plan A: lerobot's own `RobotKinematics` (Placo + `so101_new_calib.urdf`) if it's importable in the 0.5.2 env, OR the sample repo's IK helper. Discover which exists before coding; record it in `docs/env_report.md`.
   - `armani/grasp.py` (hover half only this stage): `hover_over(arm, x, y) -> bool` — table height + `HOVER_HEIGHT_M` (config, ~0.10) gives Z; IK → joint target; **target must pass `clamp_action(profile="policy")` and a `TABLE_POLYGON` membership check BEFORE motion**; then `goto(..., profile="policy")`. Returns False (no motion) if IK fails or the point is outside the polygon — that's a real confidence signal, not an error.
   - If the IK layer fights you >20 min, STOP and write the blocker precisely — the Plan A→B→C switch is the human architect's call, not yours (CLAUDE.md). Do not silently fall back to taught zones.

4. **`config` additions:** `HOVER_HEIGHT_M`, `TABLE_HEIGHT_M` (measured by operator), homography path, object catalog (the 5 demo objects with per-object grasp height for stage 5 — define now, use the height later), Gemini confidence threshold, reprojection-error ceiling.

## Smoke tests (add both to doctor)

- `tests/smoke_09_vision.py` — dry-run: with a saved frame, call `eyes.locate("banana")` (or whatever's on the table) and print the point + confidence + raw reply; draw the point on the frame → `tests/out/detect.jpg` for the operator to eyeball. No motion. SKIP cleanly if no homography yet (vision alone still works).
- `tests/smoke_10_hover.py` — **operator + hardware, the stage deliverable.** Full chain on ONE named object: locate → homography → IK → hover 10cm above → hold 2s → return to start (SafeMotion). Prints the pixel, the robot XY, the IK solution, and whether the gripper visibly ends up over the object. Requires homography + verified home. **Asserts it never commands Z below hover height** — a coding guard that the descent can't happen this stage.

## Constraints

- HOVER ONLY. No gripper actuation, no descent below hover height, no pick. Stage 5.
- No changes to the voice agent, gestures, or safety semantics (you may ADD `hover_over` and vision tools' plumbing, but do not wire vision into `agent.py`'s toolset yet — stage 6 does that behind the trust gates).
- Camera stays at the pinned index; if calibration needs the camera, reuse the locked tripod position — do not move it.
- `.env`: document any new `ARMANI_*` vars in `.env.example`.

## Definition of done

Standard five (CLAUDE.md), plus: smoke_09 shows a correct point on the real object (operator eyeballs `detect.jpg`); homography saved with reprojection error under the ceiling; smoke_10 hovers the gripper visibly over the named object and returns to start, witnessed by the operator; `TABLE_POLYGON` populated. Commit `stage 4: eyes + homography + hover`. Four-part report — and in KNOWN LIMITATIONS state the measured detection accuracy and hover error in cm, because stage 5's grasp reliability rides entirely on how good this map is.
