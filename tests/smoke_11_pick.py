#!/usr/bin/env python
"""Smoke 11 — taught-zone pick. The stage-5 deliverable.

Four modes, in increasing order of what they touch:

    --dry-run     no network, no camera, no motion. Checks the decision path
                  refuses honestly (unseen / ambiguous / off-the-spots / no
                  macro) using synthetic detections. This is what doctor runs.

    (default)     real vision against the camera or --frame, replayed against a
                  DryRunArm. Prints the detected object, chosen zone, confidence
                  and assignment margin, and ASSERTS nothing was commanded.

    --live        OPERATOR + HARDWARE. The real pick: identify the zone, replay
                  the recorded macro, and have the operator confirm the object
                  was actually lifted.

    --identity N  the competence bar. Place each object on each spot in turn;
                  every trial is scored and the tally goes to the decision log.
                  This is the number stage 6's ambiguity gate gets tuned against
                  — a grasp-coordinate number would be meaningless here, because
                  the grasp is a human recording.

    python tests/smoke_11_pick.py --dry-run
    python tests/smoke_11_pick.py --frame tests/out/frame.jpg --object "red block"
    python tests/smoke_11_pick.py --live --object "red block"
    python tests/smoke_11_pick.py --identity 10
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import EXIT_SKIP, banner, fail, ok, skip

from armani import config, eyes, gestures, motion, pick, safety, zones
from armani.logutil import log_event

DEFAULT_OBJECT = next(iter(config.OBJECT_CATALOG))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="offline decision-path checks only")
    parser.add_argument("--live", action="store_true", help="OPERATOR + HARDWARE: really pick")
    parser.add_argument("--identity", type=int, metavar="N", help="run N identity trials")
    parser.add_argument("--object", default=DEFAULT_OBJECT, help="what to pick")
    parser.add_argument("--frame", help="use this image instead of the camera")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    banner("Smoke 11: taught-zone pick")

    zone_set = zones.load_zones()
    print(f"zones   : {zones.describe(zone_set)}")
    recorded = gestures.episode_count(config.PICK_DATASET_ROOT)
    print(f"macros  : {recorded} pick episode(s) at {config.PICK_DATASET_ROOT}")

    if args.dry_run:
        return _dry_run()
    if args.identity is not None:
        if args.identity < 1:
            return fail("--identity needs at least 1 trial")
        return _identity(args, zone_set)
    if args.live:
        return _live(args, zone_set)
    return _offline_vision(args, zone_set)


# --- dry run: no network, no camera, no motion ---------------------------


def _dry_run() -> int:
    """Prove the refusals with synthetic detections. Doctor-safe."""
    print("\n[dry-run] no camera, no network, no motion — checking the decision path")

    frame_size = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    zone_set = zones.ZoneSet(
        zones=(
            zones.Zone("z1", "front-left", (100, 300), 0),
            zones.Zone("z2", "front-centre", (320, 300), 1),
        ),
        frame_size=frame_size,
        created="dry-run",
    )

    def detection(point):
        return eyes.Detection(
            label="red block", point=point, confidence=0.9, frame_size=frame_size, model="dry-run"
        )

    checks: list[tuple[str, bool, str]] = []

    on_spot = zones.assign_zone(detection((104, 303)), zone_set)
    checks.append((
        "object on a spot is assigned to it",
        on_spot.ok and on_spot.zone is not None and on_spot.zone.id == "z1",
        f"zone={on_spot.zone.id if on_spot.zone else None} margin={on_spot.margin_px:.0f}px",
    ))

    between = zones.assign_zone(detection((210, 300)), zone_set)
    checks.append((
        "object between two spots reads as AMBIGUOUS",
        between.ambiguous and not between.ok,
        between.reason,
    ))

    far = zones.assign_zone(detection((320, 15)), zone_set)
    checks.append((
        "object off every spot is REFUSED",
        far.zone is None,
        far.reason,
    ))

    wrong_size = zones.assign_zone(
        eyes.Detection(label="red block", point=(104, 303), confidence=0.9,
                       frame_size=(1280, 960), model="dry-run"),
        zone_set,
    )
    checks.append((
        "detection from a different frame size is REFUSED",
        wrong_size.zone is None,
        wrong_size.reason,
    ))

    # Serialisability: these all reach logs/decisions.jsonl, which the dashboard
    # parses. An inf or nan here would be a broken log line at demo time.
    try:
        for match in (on_spot, between, far, wrong_size):
            json.dumps(match.as_log(), allow_nan=False)
        serialises = True
        detail = "all zone matches are strict JSON"
    except (TypeError, ValueError) as exc:
        serialises, detail = False, str(exc)
    checks.append(("zone matches serialise for the decision log", serialises, detail))

    print()
    failed = 0
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"         {detail}")
        failed += 0 if passed else 1

    print("\n[dry-run] no arm was constructed and nothing was commanded.")
    if failed:
        return fail(f"{failed} decision-path check(s) failed")
    return ok(f"{len(checks)} decision-path checks passed, no motion")


# --- real vision, simulated arm ------------------------------------------


def _offline_vision(args, zone_set) -> int:
    """Real Gemini call, DryRunArm. Identity without risk."""
    if zone_set is None:
        return skip("no zones defined — run: python scripts/define_zones.py")
    if config.api_key("GOOGLE_API_KEY") is None:
        return skip("GOOGLE_API_KEY is not set")

    frame = _load_frame(args)
    if frame is None:
        return skip("no frame available (camera unavailable and no --frame given)")

    arm = motion.DryRunArm()
    result = pick.pick_object(arm, args.object, frame=frame, verify=False)
    _report(result)

    if arm.sends:
        return fail(f"DryRunArm was commanded {arm.sends} times; this mode must not move")
    print("\nnothing was commanded (DryRunArm recorded 0 sends)")

    if not result.seen:
        return skip(f"Gemini did not see {args.object!r}: {result.reason}")
    if not result:
        # A refusal here is the system working, not a broken test.
        return ok(f"refused honestly without moving: {result.reason}")
    return ok(
        f"would pick {args.object!r} from {result.zone_label!r} "
        f"(confidence {result.confidence:.2f}, margin {result.assignment_margin:.0f} px)"
    )


# --- live pick -----------------------------------------------------------


def _live(args, zone_set) -> int:
    if zone_set is None:
        return skip("no zones defined — run: python scripts/define_zones.py")
    if config.api_key("GOOGLE_API_KEY") is None:
        return skip("GOOGLE_API_KEY is not set")

    recorded = gestures.episode_count(config.PICK_DATASET_ROOT)
    if recorded == 0:
        return skip(
            f"no pick macros recorded at {config.PICK_DATASET_ROOT}. See {pick.RUNBOOK}."
        )
    if not config.HOME_VERIFIED:
        return skip("home pose is not verified — run scripts/capture_home.py (safety rule 4)")

    print(f"\nPut the {args.object!r} on one of the marked spots, and clear the others.")
    if not safety.require_operator(f"replay a recorded pick macro to grasp the {args.object!r}"):
        return skip("operator did not confirm presence")

    safety.install_kill_switch()
    try:
        arm = motion.connect()
    except Exception as exc:
        return fail(f"could not connect: {type(exc).__name__}: {exc}")

    try:
        result = pick.pick_object(arm, args.object)
        _report(result)
        log_event("smoke_11_live", **result.as_log())

        if not result.seen:
            return skip(f"not seen, so nothing moved: {result.reason}")
        if not result.moved:
            return skip(f"refused without moving: {result.reason}")
        if not result:
            return fail(f"the pick started but did not finish: {result.reason}")

        print("\n" + "=" * 68)
        print(f"  OPERATOR: is the {args.object} actually held in the gripper?")
        print("=" * 68)
        try:
            answer = input("  Lifted and held? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = ""
        held = answer in ("y", "yes")
        log_event("smoke_11_verdict", object=args.object, zone=result.zone, held=held)

        if not held:
            return fail(
                "the macro ran but the object was not lifted. Re-record that zone's episode "
                f"with the object exactly on the mark — see {pick.RUNBOOK}."
            )
        return ok(f"picked {args.object!r} from {result.zone_label!r}")
    except KeyboardInterrupt:
        print("\ninterrupted — the arm holds where it is.")
        return EXIT_SKIP
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})")


# --- identity accuracy ---------------------------------------------------


def _identity(args, zone_set) -> int:
    """The competence bar: does Gemini put each object on the right spot?

    No motion at all — this measures identity, which is the only thing vision is
    responsible for on the taught-zone path.
    """
    if zone_set is None:
        return skip("no zones defined — run: python scripts/define_zones.py")
    if config.api_key("GOOGLE_API_KEY") is None:
        return skip("GOOGLE_API_KEY is not set")

    objects = list(config.OBJECT_CATALOG)
    print(
        f"\n{args.identity} identity trials. NO MOTION.\n"
        f"  Objects: {', '.join(objects)}\n"
        f"  Spots  : {', '.join(f'{z.id}={z.label}' for z in zone_set.zones)}\n"
        "  For each trial: place the named object on the named spot, then press ENTER.\n"
    )

    trials: list[dict] = []
    for index in range(args.identity):
        obj = objects[index % len(objects)]
        expected = zone_set.zones[(index // len(objects)) % len(zone_set.zones)]
        print(f"\n--- trial {index + 1}/{args.identity} ---")
        print(f"  place the {obj!r} on spot {expected.id} ({expected.label!r})")
        try:
            if input("  ENTER when placed, or 's' to skip: ").strip().lower() == "s":
                continue
        except (EOFError, KeyboardInterrupt):
            print()
            break

        try:
            detection = eyes.locate(obj)
        except eyes.EyesError as exc:
            print(f"  vision failed: {exc}")
            trials.append({"object": obj, "expected": expected.id, "got": None, "correct": False})
            continue

        if detection is None:
            print(f"  NOT SEEN — counts as wrong for {obj!r}")
            trials.append({"object": obj, "expected": expected.id, "got": None, "correct": False})
            continue

        match = zones.assign_zone(detection, zone_set)
        got = None if match.zone is None else match.zone.id
        correct = got == expected.id
        print(
            f"  -> {got or 'none'} ({'CORRECT' if correct else 'WRONG'}), "
            f"confidence {detection.confidence:.2f}, margin "
            f"{'inf' if match.zone is None else f'{match.margin_px:.0f} px'}"
            f"{', AMBIGUOUS' if match.ambiguous else ''}"
        )
        trials.append({
            "object": obj,
            "expected": expected.id,
            "got": got,
            "correct": correct,
            "ambiguous": match.ambiguous,
            "confidence": round(detection.confidence, 3),
        })

    return _tally(trials)


def _tally(trials: list[dict]) -> int:
    if not trials:
        return skip("no trials completed")

    correct = sum(1 for t in trials if t["correct"])
    accuracy = correct / len(trials)

    # Which objects get mixed up — the thing stage 6's ambiguity gate is tuned against.
    confusions: dict[str, int] = {}
    for trial in trials:
        if not trial["correct"]:
            key = f"{trial['object']} -> {trial['got'] or 'not seen'} (expected {trial['expected']})"
            confusions[key] = confusions.get(key, 0) + 1

    print("\n" + "=" * 68)
    print(f"  IDENTITY ACCURACY: {correct}/{len(trials)} = {accuracy:.0%}")
    print("=" * 68)
    if confusions:
        print("  Confusions, most common first:")
        for key, count in sorted(confusions.items(), key=lambda kv: -kv[1]):
            print(f"    {count}x  {key}")
    else:
        print("  No confusions.")
    ambiguous = sum(1 for t in trials if t.get("ambiguous"))
    print(f"  Flagged ambiguous: {ambiguous}/{len(trials)}")

    log_event(
        "identity_accuracy",
        trials=len(trials),
        correct=correct,
        accuracy=round(accuracy, 3),
        ambiguous=ambiguous,
        confusions=confusions,
        detail=trials,
    )
    print("\n  Written to the decision log as `identity_accuracy`.")

    if accuracy < 0.8:
        return fail(
            f"identity accuracy {accuracy:.0%} is below 80% — the demo's competence bar. "
            "Check the confusions above: rename objects to something Gemini distinguishes, "
            "or move the spots further apart."
        )
    return ok(f"identity accuracy {accuracy:.0%} over {len(trials)} trials")


# --- shared --------------------------------------------------------------


def _load_frame(args):
    if args.frame:
        import cv2

        return cv2.imread(args.frame)
    try:
        return eyes.capture_frame()
    except eyes.EyesError as exc:
        print(f"  camera unavailable: {exc}")
        return None


def _report(result: pick.PickResult) -> None:
    print("\n--- decision ---")
    print(f"  object     : {result.object!r}")
    print(f"  seen       : {result.seen}")
    print(f"  confidence : {result.confidence:.2f}")
    print(f"  zone       : {result.zone} ({result.zone_label})")
    print(f"  distance   : {result.distance_px:.0f} px from the spot")
    print(f"  margin     : {result.assignment_margin:.0f} px clear of the runner-up")
    print(f"  ambiguous  : {result.ambiguous}"
          + (f"  candidates={list(result.candidate_zones)}" if result.ambiguous else ""))
    print(f"  moved      : {result.moved}")
    if result.verify.gripper_percent is not None:
        print(f"  gripper    : {result.verify.gripper_percent:.1f}% (held_guess={result.held_guess})")
    if result.reason:
        print(f"  reason     : {result.reason}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(EXIT_SKIP) from None
