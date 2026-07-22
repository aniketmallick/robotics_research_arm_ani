#!/usr/bin/env python
"""Spike S1 — generate the printable ChArUco board.

Writes armani/data/charuco_board.png at an exact DPI so that printing at 100%
("Actual size") produces squares of the intended physical size, and marks the
THREE corners the calibration will ask you to touch with the gripper tip.

    python scripts/make_charuco.py            # write the board
    python scripts/make_charuco.py --dpi 600  # finer print

THE PRINTER TRAP: printers rescale. After printing, MEASURE one square with a
ruler and put the measured value in the environment:

    export ARMANI_CHARUCO_SQUARE_MM=31.4     # whatever your ruler says

The measured number is truth; the intended number below is only a hope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import _bootstrap  # noqa: E402,F401  (verifies the interpreter, exits if wrong)

from armani import calibrate, config  # noqa: E402

OUT_PATH = config.DATA_DIR / "charuco_board.png"
A4_MM = (210, 297)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dpi", type=int, default=300, help="print resolution (default 300)")
    parser.add_argument("--dry-run", action="store_true", help="say what it would write, write nothing")
    args = parser.parse_args()

    import cv2
    import numpy as np

    square_mm = calibrate.measured_square_m() * 1000.0
    marker_mm = square_mm * (config.CHARUCO_MARKER_M / config.CHARUCO_SQUARE_M)
    board_w_mm = config.CHARUCO_SQUARES_X * square_mm
    board_h_mm = config.CHARUCO_SQUARES_Y * square_mm
    margin_mm = square_mm * 0.5
    px_per_mm = args.dpi / 25.4

    def mm_to_px(mm: float) -> int:
        return int(round(mm * px_per_mm))

    print("=== ChArUco board generator (Spike S1) ===")
    print(f"  board     : {config.CHARUCO_SQUARES_X}x{config.CHARUCO_SQUARES_Y} squares, "
          f"dict {config.CHARUCO_DICT}")
    print(f"  square    : {square_mm:.2f} mm  (marker {marker_mm:.2f} mm)")
    print(f"  board size: {board_w_mm:.0f} x {board_h_mm:.0f} mm + {margin_mm:.0f} mm margin")
    print(f"  DPI       : {args.dpi}  ->  {mm_to_px(board_w_mm + 2*margin_mm)} x "
          f"{mm_to_px(board_h_mm + 2*margin_mm)} px")
    if board_w_mm + 2 * margin_mm > A4_MM[0] or board_h_mm + 2 * margin_mm > A4_MM[1]:
        print(f"  WARNING: {board_w_mm + 2*margin_mm:.0f}x{board_h_mm + 2*margin_mm:.0f} mm "
              f"exceeds A4 ({A4_MM[0]}x{A4_MM[1]} mm) — it will be cropped. "
              "Lower the square size or squares count.")

    if args.dry_run:
        print(f"\n[dry-run] would write {OUT_PATH} and mark corners {calibrate.calibration_corner_ids()}")
        return 0

    board = calibrate.charuco_board(square_m=square_mm / 1000.0)
    out_w = mm_to_px(board_w_mm + 2 * margin_mm)
    out_h = mm_to_px(board_h_mm + 2 * margin_mm)
    image = board.generateImage((out_w, out_h), marginSize=mm_to_px(margin_mm))

    # Mark the three corners the calibration will touch, by DETECTING the board
    # we just drew and circling the exact corners OpenCV will detect — so the
    # marks sit precisely on the ids we fit against, no manual projection.
    marked = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ids_to_mark = calibrate.calibration_corner_ids()
    try:
        pixels, _, found_ids = calibrate.detect_charuco(image, board)
        by_id = {cid: px for px, cid in zip(pixels, found_ids)}
        for label, cid in enumerate(ids_to_mark, start=1):
            if cid not in by_id:
                continue
            x, y = int(by_id[cid][0]), int(by_id[cid][1])
            cv2.circle(marked, (x, y), mm_to_px(4), (0, 0, 220), max(2, mm_to_px(0.6)))
            cv2.putText(marked, str(label), (x + mm_to_px(3), y - mm_to_px(3)),
                        cv2.FONT_HERSHEY_SIMPLEX, px_per_mm * 0.9, (0, 0, 220),
                        max(2, mm_to_px(0.5)), cv2.LINE_AA)
    except calibrate.CalibrationError as exc:
        print(f"  (could not self-detect to mark corners: {exc}; board still written unmarked)")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _save_with_dpi(marked, OUT_PATH, args.dpi, np)

    print(f"\nWrote {OUT_PATH}")
    print("  1. Print at 100% / 'Actual size' (NOT 'fit to page').")
    print("  2. Measure one square with a ruler; export ARMANI_CHARUCO_SQUARE_MM=<measured>.")
    print("  3. Tape it FLAT on the table, fully in the C920 view. Do not move the camera after.")
    print("  4. The red-circled corners 1/2/3 are the ones calibrate_charuco.py will ask you to touch.")
    return 0


def _save_with_dpi(image, path: Path, dpi: int, np) -> None:
    """Embed the DPI so the printer honours 100% scale. Falls back to cv2."""
    try:
        from PIL import Image

        rgb = image[:, :, ::-1] if image.ndim == 3 else image  # BGR->RGB for PIL
        Image.fromarray(np.ascontiguousarray(rgb)).save(str(path), dpi=(dpi, dpi))
    except Exception as exc:
        import cv2

        cv2.imwrite(str(path), image)
        print(f"  (Pillow unavailable: {exc}; DPI not embedded — print scale may be off, "
              "verify with the ruler)")


if __name__ == "__main__":
    raise SystemExit(main())
