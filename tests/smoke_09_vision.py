#!/usr/bin/env python
"""Smoke 09 — vision. Gemini points at a named object. NO MOTION AT ALL.

Captures a frame (or reads one you pass with --frame), asks eyes.locate() where
the named object is, prints the point, the confidence and the raw reply, and
draws the point on the frame so the operator can eyeball whether it is actually
on the object:

    tests/out/detect.jpg

If the camera has been mapped to the table, the pixel is also converted to robot
coordinates and checked against the table polygon — but a missing homography is
not a failure here. Vision works on its own; that part is simply skipped.

    python tests/smoke_09_vision.py --object "red block"
    python tests/smoke_09_vision.py --frame tests/out/frame.jpg --object "red block"
"""

from __future__ import annotations

import argparse

from _bootstrap import EXIT_SKIP, banner, fail, ok, permission_hint, skip

from armani import calibrate, config, eyes
from armani.logutil import log_event

DEFAULT_OBJECT = next(iter(config.OBJECT_CATALOG))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="explain the plan, call nothing")
    parser.add_argument("--object", default=DEFAULT_OBJECT, help="what to look for")
    parser.add_argument("--frame", help="use this image instead of the camera")
    parser.add_argument(
        "--list", action="store_true",
        help="also exercise list_visible() over the whole object catalog",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    banner("Smoke 09: vision (no motion)")

    out_path = config.TEST_OUT_DIR / "detect.jpg"
    if args.dry_run:
        print(f"[dry-run] would capture a frame from camera index {config.CAMERA_INDEX}")
        print(f"[dry-run] would ask {config.GEMINI_MODELS[0]} to point at {args.object!r}")
        print(f"[dry-run] would ask {config.EYES_SAMPLES} independently-worded queries and fuse them")
        print(f"[dry-run] would draw the result to {out_path}")
        print("[dry-run] no motion in this test under any circumstances")
        return ok("dry run complete")

    if config.api_key("GOOGLE_API_KEY") is None:
        return skip("GOOGLE_API_KEY is not set — see .env.example")

    try:
        import cv2
    except ImportError as exc:
        return fail(f"opencv not importable: {exc}")

    # --- frame ---
    if args.frame:
        frame = cv2.imread(args.frame)
        if frame is None:
            return fail(f"could not read {args.frame}")
        print(f"frame  : {args.frame} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        try:
            frame = eyes.capture_frame()
        except eyes.EyesError as exc:
            if "Camera" in str(exc) or "authorized" in str(exc):
                permission_hint("Camera", str(exc))
            return skip(f"no camera frame: {exc}")
        print(f"frame  : camera index {config.CAMERA_INDEX} ({frame.shape[1]}x{frame.shape[0]})")

    # --- vision ---
    print(f"looking for: {args.object!r}  ({config.EYES_SAMPLES} independent queries)")
    try:
        detection = eyes.locate(args.object, frame=frame)
    except eyes.EyesError as exc:
        return fail(f"vision call failed: {exc}")

    if detection is None:
        # A real answer, not a crash. Still write the frame so the operator can
        # see what the camera saw when it said "not there".
        config.TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        return skip(
            f"Gemini did not see {args.object!r} in the frame. Look at {out_path} — is it "
            "actually visible, lit, and unobstructed? Try --object with a plainer name."
        )

    print("\n--- detection ---")
    print(f"  model      : {detection.model}")
    print(f"  point      : {detection.point} px")
    print(f"  confidence : {detection.confidence:.2f}   (threshold {config.EYES_CONF_THRESHOLD:.2f})")
    print(f"    found_rate  {detection.found_rate:.2f}  (queries that saw it)")
    print(f"    agreement   {detection.agreement:.2f}  (how closely they agreed on where)")
    print(f"    self_report {detection.self_report:.2f}  (what the model claims — uncalibrated)")
    print("\n--- raw reply ---")
    for line in detection.raw.splitlines():
        print(f"  {line}")

    detections = [detection]
    if args.list:
        print("\n--- list_visible over the catalog (one query) ---")
        try:
            listed = eyes.list_visible(list(config.OBJECT_CATALOG), frame=frame)
        except eyes.EyesError as exc:
            print(f"  list_visible failed: {exc}")
        else:
            for item in listed:
                print(f"  {item.label:<12} {item.point} conf={item.confidence:.2f}")
            detections = listed or detections

    # --- optional: pixel -> robot ---
    _report_mapping(detection)

    config.TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), eyes.annotate(frame, detections)):
        return fail(f"could not write {out_path}")
    print(f"\nWrote {out_path}")
    print("OPERATOR: open it. Is the marker actually ON the object? That is the real result —")
    print("          a confident number on the wrong object is the failure this test exists to catch.")

    log_event("smoke_09", object=args.object, **detection.as_log())
    return ok(f"pointed at {args.object!r} with confidence {detection.confidence:.2f}")


def _report_mapping(detection: eyes.Detection) -> None:
    """Convert the pixel to robot XY when calibration exists; skip cleanly if not."""
    print("\n--- pixel -> robot ---")
    homography = calibrate.load()
    if homography is None:
        print("  SKIPPED: not calibrated yet (no armani/data/homography.json).")
        print("  Vision works without it; run scripts/calibrate_camera.py to map pixels to the table.")
        return

    print(f"  {calibrate.describe(homography)}")
    if homography.frame_size != detection.frame_size:
        print(
            f"  WARNING: calibrated at {homography.frame_size[0]}x{homography.frame_size[1]} but this "
            f"frame is {detection.frame_size[0]}x{detection.frame_size[1]}. The map does not apply."
        )
        return
    try:
        x, y = homography.to_robot(detection.point)
    except calibrate.CalibrationError as exc:
        print(f"  could not map the point: {exc}")
        return

    inside = calibrate.point_in_polygon(x, y, margin_m=config.POLYGON_MARGIN_M)
    print(f"  robot XY : ({x:+.3f}, {y:+.3f}) m")
    print(f"  on table : {'yes' if inside else 'NO — outside the calibrated polygon'}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(EXIT_SKIP) from None
