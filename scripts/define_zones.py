#!/usr/bin/env python
"""Define the taught-zone map: click each marked spot once. OPERATOR REQUIRED.

This is the entire "calibration" for the demo pick path — about two minutes, one
tolerant click per spot, no ruler and no robot coordinates. Compare that with the
stage-4 homography it replaces, which needed a printed board, a measured origin
and sub-15-pixel reprojection to be usable at all.

You click where each marked spot APPEARS in the camera frame. Zone labels name
the SPOT, not the object on it ("front-left", not "banana") — which object sits
where is decided live by Gemini on every frame, so objects can be swapped
between spots at demo time and everything still works.

The order you click is the order the pick macros must be recorded in: zone 1's
macro is episode 0, zone 2's is episode 1, and so on. See docs/recording_picks.md.

    python scripts/define_zones.py --dry-run
    python scripts/define_zones.py
    python scripts/define_zones.py --zones 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import _bootstrap  # noqa: E402,F401  (verifies the interpreter, exits if wrong)

from armani import config, eyes, zones  # noqa: E402
from armani.logutil import log_event  # noqa: E402

WINDOW = "ARM-ANI zones — click the marked spot's centre  (s = skip, ESC = stop)"

# Offered as defaults so the operator can just press ENTER. They name positions
# on the table, never objects.
SUGGESTED_LABELS = ("front-left", "front-centre", "front-right", "back-left", "back-right")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="explain the plan, write nothing")
    parser.add_argument("--zones", type=int, default=5, help="how many spots to define")
    parser.add_argument("--frame", help="use this image instead of the camera")
    args = parser.parse_args()

    print("=== ARM-ANI zone definition ===")
    print(f"output : {config.ZONES_PATH}")
    existing = zones.load_zones()
    if existing is not None:
        print(f"current: {zones.describe(existing)}")

    if args.dry_run:
        print(f"\n[dry-run] would capture a frame from camera index {config.CAMERA_INDEX}")
        print(f"[dry-run] would ask you to click {args.zones} marked spots and label each")
        print(f"[dry-run] would write {config.ZONES_PATH}")
        print("[dry-run] zone order == pick macro episode order (see docs/recording_picks.md)")
        print("[dry-run] nothing written — that would falsely mark the zones defined.")
        return 0

    if args.zones < 1:
        print("--zones must be at least 1", file=sys.stderr)
        return 1

    try:
        import cv2
    except ImportError as exc:
        print(f"opencv not importable: {exc}", file=sys.stderr)
        return 1

    if args.frame:
        frame = cv2.imread(args.frame)
        if frame is None:
            print(f"could not read {args.frame}", file=sys.stderr)
            return 1
    else:
        try:
            frame = eyes.capture_frame()
        except eyes.EyesError as exc:
            print(f"no camera frame: {exc}", file=sys.stderr)
            return 1

    height, width = frame.shape[:2]
    print(f"frame  : {width}x{height}")
    print(
        "\nPut a physical marker at each spot you want to pick from, with the demo\n"
        "objects on them, so you can see exactly what you are clicking.\n"
        "Click the CENTRE of each marked spot. Order matters: it is the episode order\n"
        "you must record the pick macros in.\n"
    )

    try:
        defined = _collect(frame, args.zones)
    except KeyboardInterrupt:
        print("\nInterrupted — nothing saved.", file=sys.stderr)
        return 1

    if not defined:
        print("No zones defined — nothing saved.", file=sys.stderr)
        return 1

    return _save(defined, (width, height), frame)


def _collect(frame, wanted: int) -> list[dict]:
    """Click-then-label, once per spot."""
    defined: list[dict] = []
    for index in range(wanted):
        print(f"\n--- zone {index + 1} of {wanted} ---")
        point = _click_point(frame, defined, index + 1, wanted)
        if point is None:
            print("  stopped clicking.")
            break

        suggested = SUGGESTED_LABELS[index] if index < len(SUGGESTED_LABELS) else f"spot-{index + 1}"
        try:
            label = input(f"  label for this SPOT [{suggested}]: ").strip() or suggested
        except (EOFError, KeyboardInterrupt):
            print()
            break

        defined.append(
            {
                "id": f"z{index + 1}",
                "label": label,
                "pixel_center": [int(point[0]), int(point[1])],
                # Episode order IS click order. The runbook records them in this
                # order, and getting it wrong means the arm picks the wrong spot.
                "pick_episode": index,
            }
        )
        print(f"  zone z{index + 1} {label!r} at pixel ({int(point[0])}, {int(point[1])}) "
              f"-> pick macro episode {index}")
    return defined


def _click_point(frame, defined: list[dict], index: int, total: int):
    """Show the frame with the zones so far drawn, and return the next click.

    A near-copy of the click helper in scripts/calibrate_camera.py. Left
    duplicated on purpose: that script is the frozen stage-4 stretch path, and
    coupling it to new code to save twenty lines of overlay drawing would put
    the demo path at risk of a change made for the stretch path.
    """
    import cv2

    clicked: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((float(x), float(y)))

    canvas = frame.copy()
    for zone in defined:
        x, y = zone["pixel_center"]
        cv2.circle(canvas, (x, y), 12, (0, 255, 0), 2)
        cv2.putText(canvas, zone["label"], (x - 30, y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"zone {index}/{total}: click the marked spot's centre",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW, canvas)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while not clicked:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("s"), 27):  # s or ESC
                return None
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyWindow(WINDOW)
        # macOS needs a few event-loop turns before the window actually goes.
        for _ in range(4):
            cv2.waitKey(1)
    return clicked[0]


def _save(defined: list[dict], frame_size: tuple[int, int], frame) -> int:
    close = _too_close(defined)
    if close:
        # Two spots closer than the ambiguity margin can never be told apart:
        # every assignment between them would be flagged ambiguous and the
        # demo would ask "which one?" forever.
        print(
            f"\nWARNING: {close} are closer together than the "
            f"{config.ASSIGNMENT_MARGIN_PX:.0f} px ambiguity margin.\n"
            "  Objects on them will always read as ambiguous. Move the spots further\n"
            "  apart, or lower ARMANI_ASSIGNMENT_MARGIN_PX.",
            file=sys.stderr,
        )
        try:
            if input("Save anyway? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Not saved.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print()
            return 1

    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "zones": defined,
        "note": (
            "Pixel positions of the marked table spots. Valid for this camera pose and "
            "frame size. zone order == pick macro episode order."
        ),
    }
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.ZONES_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    log_event("zones_defined", count=len(defined), zones=defined)

    # A reference image of what was clicked, so a later "did the camera move?"
    # question can be answered by looking rather than guessing.
    try:
        import cv2

        config.TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
        overlay = frame.copy()
        for zone in defined:
            x, y = zone["pixel_center"]
            cv2.drawMarker(overlay, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
            cv2.circle(overlay, (x, y), 12, (0, 255, 0), 2)
            cv2.putText(overlay, f"{zone['id']} {zone['label']}", (x - 34, y - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        reference = config.TEST_OUT_DIR / "zones.jpg"
        cv2.imwrite(str(reference), overlay)
        print(f"\nReference image: {reference}")
    except Exception as exc:  # a missing reference image must not fail the setup
        print(f"  (could not write the reference image: {exc})", file=sys.stderr)

    print("\n" + "=" * 68)
    print("  ZONES SAVED")
    print("=" * 68)
    print(f"  {zones.describe(zones.load_zones())}")
    print(
        "\n  KEEP THE CAMERA WHERE IT IS. These are pixel positions, so moving the\n"
        "  tripod moves every zone. A bump here is far cheaper than it was for the\n"
        "  stage-4 homography — just re-run this script — but it is not free.\n"
    )
    print("  Next: record one pick macro per zone, IN THIS ORDER:")
    for zone in defined:
        print(f"    episode {zone['pick_episode']}  ->  {zone['id']} {zone['label']}")
    print("\n  See docs/recording_picks.md, then: python tests/smoke_11_pick.py --dry-run")
    return 0


def _too_close(defined: list[dict]) -> str:
    """Pairs of spots that sit inside the ambiguity margin of each other."""
    import math

    pairs = []
    for i, first in enumerate(defined):
        for second in defined[i + 1 :]:
            distance = math.dist(first["pixel_center"], second["pixel_center"])
            if distance < config.ASSIGNMENT_MARGIN_PX:
                pairs.append(f"{first['label']!r} and {second['label']!r} ({distance:.0f} px)")
    return "; ".join(pairs)


if __name__ == "__main__":
    raise SystemExit(main())
