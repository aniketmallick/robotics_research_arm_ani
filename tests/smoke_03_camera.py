#!/usr/bin/env python
"""Smoke 03 — camera. Identifies the C920, grabs a frame, reports the truth.

This machine exposes several cameras (OBS Virtual, FaceTime HD, the C920 and a
Continuity iPhone camera). OpenCV indices do NOT reliably correspond to the
order macOS lists devices in, and OpenCV cannot report device names, so this
test does not guess: it saves a probe image from every index that works and
asks the operator which one is the C920 view of the table.

Set ARMANI_CAMERA_INDEX in .env once it is known and this becomes a
non-interactive verification of that index.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager

from _bootstrap import banner, fail, ok, parse_args, permission_hint, skip

from armani import config
from armani.logutil import log_event

# OpenCV prints this when macOS Camera access is off. It is not an exception,
# and cv2.VideoCapture just reports "not opened", so the only way to tell
# "permission denied" from "no camera at this index" is to read the message.
UNAUTHORIZED_MARKER = "not authorized to capture video"


@contextmanager
def capture_c_stderr():
    """Capture writes to file descriptor 2, including from native libraries.

    contextlib.redirect_stderr only rebinds Python's sys.stderr object. The
    AVFoundation backend writes straight to fd 2 from C, so it would slip past
    that entirely and the permission check would never fire.
    """
    with tempfile.TemporaryFile(mode="w+") as tmp:
        saved_fd = os.dup(2)
        try:
            os.dup2(tmp.fileno(), 2)
            yield
        finally:
            # Restore and publish inside finally so fd 2 is never left dangling
            # and the captured text survives an exception raised in the body.
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            tmp.seek(0)
            capture_c_stderr.last = tmp.read()  # type: ignore[attr-defined]


def probe(index: int):
    """Open one index and grab a frame. Returns (frame, (w, h, fps)) or None."""
    import cv2

    payload = None
    with capture_c_stderr():
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                read_ok, frame = cap.read()
                actual = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    cap.get(cv2.CAP_PROP_FPS),
                )
                if read_ok and frame is not None:
                    payload = (frame, actual)
        finally:
            # Always release: a raise here would otherwise hold the device open
            # and every later probe of this index would fail.
            cap.release()
    return payload, getattr(capture_c_stderr, "last", "")


def device_names() -> list[str]:
    """Camera names macOS knows about. Order is a hint, not an index mapping."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    names = []
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith(("Camera", "Model ID", "Unique ID")):
            names.append(stripped.rstrip(":"))
    return names


def main() -> int:
    args = parse_args(__doc__ or "")
    banner("Smoke 03: camera")

    out_path = config.TEST_OUT_DIR / "frame.jpg"
    if args.dry_run:
        print(f"[dry-run] would scan OpenCV indices 0..{config.CAMERA_MAX_PROBE_INDEX - 1} (AVFoundation)")
        print(f"[dry-run] would request {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}@{config.CAMERA_FPS}")
        print(f"[dry-run] would save a probe image per index, then {out_path}")
        return ok("dry run complete")

    try:
        import cv2
    except ImportError as exc:
        return fail(f"opencv not importable: {exc}")

    names = device_names()
    if names:
        print("Cameras macOS reports (order is a hint only, not the OpenCV index):")
        for name in names:
            print(f"  - {name}")

    config.TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    indices = (
        [config.CAMERA_INDEX]
        if config.CAMERA_INDEX is not None
        else list(range(config.CAMERA_MAX_PROBE_INDEX))
    )
    if config.CAMERA_INDEX is not None:
        print(f"\nUsing ARMANI_CAMERA_INDEX={config.CAMERA_INDEX} from .env")

    print("\nProbing indices:")
    working: dict[int, tuple] = {}
    unauthorized = False
    for index in indices:
        payload, stderr_text = probe(index)
        if UNAUTHORIZED_MARKER in stderr_text:
            unauthorized = True
        if payload is None:
            print(f"  index {index}: no frame")
            continue
        frame, actual = payload
        probe_path = config.TEST_OUT_DIR / f"probe_{index}.jpg"
        cv2.imwrite(str(probe_path), frame)
        print(f"  index {index}: {actual[0]}x{actual[1]} @ {actual[2]:.0f}fps -> {probe_path.name}")
        working[index] = (frame, actual)

    if unauthorized and not working:
        permission_hint("Camera", "OpenCV reported 'not authorized to capture video'")
        return fail("camera access denied by macOS")
    if not working:
        return fail(
            "No camera produced a frame. Check the C920 is plugged in and not held open "
            "by another app (Zoom, Photo Booth, OBS, Chrome)."
        )

    chosen = _choose_index(working)
    if chosen is None:
        return skip(
            f"multiple cameras responded and the C920 was not identified. Open "
            f"{config.TEST_OUT_DIR}/probe_*.jpg, find the C920's view of the table, "
            "and set ARMANI_CAMERA_INDEX in .env."
        )

    frame, (width, height, fps) = working[chosen]
    if not cv2.imwrite(str(out_path), frame):
        return fail(f"could not write {out_path}")

    print(f"\nSaved {out_path} ({frame.shape[1]}x{frame.shape[0]}) from index {chosen}")
    print(f"Negotiated: {width}x{height} @ {fps:.0f}fps  (requested "
          f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}@{config.CAMERA_FPS})")
    log_event("smoke_03", index=chosen, width=width, height=height, fps=fps, path=str(out_path))

    if (width, height) != (config.CAMERA_WIDTH, config.CAMERA_HEIGHT):
        return skip(
            f"camera negotiated {width}x{height}, not the requested "
            f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}. Stage 4 assumes "
            f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}; check the index is the C920."
        )
    return ok(f"camera index {chosen} delivered a {width}x{height} frame")


def _choose_index(working: dict[int, tuple]) -> int | None:
    """Use the configured index, else ask. Never silently pick for the operator."""
    if config.CAMERA_INDEX is not None:
        return config.CAMERA_INDEX
    if len(working) == 1:
        only = next(iter(working))
        print(f"\nOnly index {only} responded; using it.")
        return only

    if not sys.stdin.isatty():
        return None

    print(
        f"\nSeveral cameras responded. Open the probe images in "
        f"{config.TEST_OUT_DIR} and find the one showing the TABLE from the C920 tripod."
    )
    options = sorted(working)
    while True:
        raw = input(f"C920 index {options} (or 's' to skip): ").strip().lower()
        if raw == "s":
            return None
        if raw.isdigit() and int(raw) in working:
            chosen = int(raw)
            print(f"Add ARMANI_CAMERA_INDEX={chosen} to .env so this never has to be asked again.")
            return chosen
        print("Not one of the working indices.")


if __name__ == "__main__":
    raise SystemExit(main())
