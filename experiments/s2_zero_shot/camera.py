"""Persistent C920 stream for the S2 control loop.

``armani.eyes.capture_frame`` opens AND releases the camera on every call — the
right call for one-shot perception, but far too slow to do at 10 Hz (AVFoundation
re-init costs hundreds of ms). So the runner keeps a single handle open for the
whole episode. The AVFoundation backend + 640x480 setup mirrors ``eyes`` exactly
so frames match what the rest of the project sees.

Headless verification uses :func:`synthetic_frame` instead of a real camera, so
the whole inference chain can be exercised on a laptop with no C920 attached.
"""

from __future__ import annotations

import numpy as np

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
WARMUP_FRAMES = 5  # AVFoundation delivers a few dark/green frames before it settles


class CameraError(RuntimeError):
    """The camera would not open or delivered no frame — an operator problem."""


class CameraStream:
    """One open C920 handle, read once per loop step. Use as a context manager."""

    def __init__(self, index: int) -> None:
        import cv2  # local import: headless verification never touches the camera

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            raise CameraError(
                f"camera index {index} would not open. Check the C920 is plugged in, that no "
                "other app (Zoom, Photo Booth, OBS) holds it, and that macOS Camera access is "
                "granted to your terminal (System Settings > Privacy & Security > Camera)."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        for _ in range(WARMUP_FRAMES):
            self._cap.read()

    def read_bgr(self) -> np.ndarray:
        """Latest frame as a BGR uint8 HxWx3 array (OpenCV's native order)."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("camera opened but delivered no frame mid-episode")
        return frame

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass

    def __enter__(self) -> "CameraStream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def synthetic_frame(step: int = 0) -> np.ndarray:
    """A deterministic stand-in frame for headless runs (no camera required).

    Gray field with a red-ish block that slides across as ``step`` advances, so
    successive frames differ (a static frame would let the policy's action queue
    look artificially stable). Deterministic on ``step`` — no RNG.
    """
    frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 127, dtype=np.uint8)
    x = 40 + (step * 13) % (CAMERA_WIDTH - 120)
    frame[200:280, x : x + 80] = (0, 0, 200)  # BGR: a red block on the "table"
    return frame
