# STAGE 5 — Pick via TAUGHT ZONES (the reliable demo path)

Act as a distinguished engineer and Re-read `CLAUDE.md` — the **Grasp** decision changed. After stage-4's homography/IK proved too fragile (tip-calibration couldn't beat 15px; gripper-lean + click imprecision), the architect has **ratified taught zones (Plan C) as the demo pick path.** Homography + IK + hover stays in the repo as an untouched STRETCH; do not delete `eyes.py`, `calibrate.py`, `kinematics.py`, or `grasp.py`'s hover path — just stop building on them for the demo.

Why this path: the operator teleop-records a *working* pick at each marked spot (reusing your proven stage-2 record/replay), and the arm replays it. That erases the coordinate-precision, IK-verticality, riser, and camera-bump problems in one move. Vision's job shrinks to **identity** — "which of the marked spots holds the thing you asked for?" — a coarse call that tolerates large error. Every trust gate still works because they are about judgment, not millimetres.

## The architecture

Fixed marked spots on the table = **zones**. Each zone has a recorded pick macro (geometry) and a pixel location (for assigning a detected object to it). Which object sits in which zone is decided live by Gemini per frame — objects can be swapped between zones and it still works.

1. **`armani/zones.py` — the zone registry (no robot-frame coordinates anywhere).**
   - Load `armani/data/zones.json`: a list of zones, each `{id, label, pixel_center: [u,v], pick_episode: int}`. `pixel_center` is where that spot appears in the C920 frame; `pick_episode` indexes the pick-macro dataset.
   - `assign_zone(detection) -> ZoneMatch`: given an `eyes.Detection` (object pixel), return the nearest zone by pixel distance, PLUS the margin to the second-nearest. A small margin (object sits between two zones) is the ambiguity signal for stage-6's G2 — expose it, don't resolve it here.
   - `list_zone_objects(candidates) -> {zone_id: object_name}`: run `eyes.list_visible`, assign each detected object to its nearest zone, for "what's on the table?" and disambiguation.
   - Pure logic + a saved frame; imports nothing from motion.

2. **`scripts/define_zones.py` — the one-time setup (operator, ~2 min, near-impossible to get wrong).**
   - Capture a C920 frame, show it, operator clicks each marked spot's center ONCE and types its label. Save `zones.json`. That's the entire "calibration" — one tolerant click per zone, no ruler, no robot coordinates. Re-runnable. Dry-run supported.
   - Reminder on save: keep the camera fixed (a bump is far less catastrophic than for homography, but still recalibrate the clicks if it moves a lot).

3. **Pick macros — reuse the stage-2 record/replay engine, do not reinvent it.**
   - The operator teleop-records one pick per zone into a `armani_picks` dataset (episode order = zone order), each macro: start at home → approach the spot → descend → close gripper → lift → return toward home holding. Same recording discipline as gestures (start and end at the captured home pose so replays are safe and chainable).
   - `docs/recording_picks.md`: the runbook. Verify the record entry point again, pin `--dataset.root`, list the exact command, and the rule that every macro starts/ends at home. The operator records with a demo object physically at each spot so the grasp is real.
   - Replay: generalize `gestures.py`'s loader or add a thin `zones`-side wrapper that loads and streams a pick episode through `arm.send` with the `recorded` clamp profile and the kill-switch checks — exactly as `play_gesture` already does. Reuse that code; don't fork it.

4. **`armani/pick.py` — tie identity to macro.**
   - `pick_object(arm, object_name) -> PickResult`: `eyes.locate(object_name)` → `zones.assign_zone` → replay that zone's macro → return a structured result (zone chosen, object, vision confidence, assignment margin, macro status, every reason string). If the object isn't seen, or is ambiguous between zones, or the zone has no recorded macro: return the honest reason and DO NOT move. Trust gates in stage 6 read these fields.
   - `PickResult` carries what stage 6 needs: `seen`, `ambiguous` (+ the two candidate zones), `zone`, `confidence`, `assignment_margin`, `moved`, `held_guess`.
   - **Post-pick verify hook (partial G5):** after the lift, re-capture a frame and expose a `verify_held()` that will (stage 6) ask Gemini "is the <object> in the gripper / gone from its spot?". Build the hook + the re-capture; leave the VLM call stubbed with a clear TODO. Also read gripper closure as a cheap secondary signal (near-fully-closed = probably empty).
   - `pick.py` is a standalone callable + smoke test. It does NOT become an agent tool this stage — stage 6 wires it behind the gates.

5. **Config:** `ZONES_PATH`, `PICK_DATASET_REPO_ID`/root, gripper open/close percents for the macros if the recording engine needs them, an `ASSIGNMENT_MARGIN_PX` threshold below which an assignment counts as ambiguous. Document new `ARMANI_*` vars.

## Smoke test

`tests/smoke_11_pick.py` (add to doctor):
- **dry-run:** with a saved frame + a `zones.json`, run `pick_object` against a DryRunArm — print the detected object, chosen zone, confidence, and assignment margin; assert NO motion and that an unseen/ambiguous object returns a falsy result with a reason.
- **live (operator + hardware), the deliverable:** with zones defined and macros recorded, say a demo object → it identifies the zone, replays the pick, operator confirms the object is actually lifted; then the **identity check**: place each of the 5 objects at each of a few spots and confirm Gemini assigns it to the correct zone. Tally the identity accuracy to the decision log — that number (not a grasp-coordinate number) is the demo's competence bar.

## Constraints

- No robot-frame coordinates, no homography, no IK in the demo path. Zones are pixel-space + recorded macros only.
- No voice-agent changes, no trust-gate dialogue (stage 6). `pick_object` is standalone.
- Reuse record/replay and the home-pose discipline; don't reimplement motion.
- Camera stays fixed; if it's bumped hard, re-run `define_zones.py` (cheap).

## Definition of done

Standard five (CLAUDE.md), plus: `define_zones.py` produces a `zones.json`; the operator records the 5 pick macros; smoke_11 dry-run green (no motion, honest refusals); one live pick witnessed (object actually lifted); the 5-object identity-accuracy tally in the decision log. Commit `stage 5: taught-zone pick`. Four-part report — give me the identity accuracy and the top confusion (which objects Gemini mixes up), because stage 6's ambiguity gate gets tuned against exactly that.
