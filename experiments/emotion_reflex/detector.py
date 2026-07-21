#!/usr/bin/env python
"""Emotion detector (eyes). SYSTEM-1 PROTOTYPE — experimental spike.

Watches the LAPTOP built-in webcam (never the C920 — that one is locked to the
arm's workspace homography), classifies the operator's facial emotion with a
small FER+ ONNX model, debounces it into a stable signal, and publishes it to
logs/emotion_state.json for reflex.py to react to.

    python detector.py --download    # fetch the models once, then exit
    python detector.py --watch       # show the cam + detection, write NOTHING
    python detector.py               # run for real: publish the emotion state

Runs in its OWN venv (opencv-python + onnxruntime + numpy) — see README. It must
not be run in the lerobot conda env; keeping the vision deps out of that env is
the whole reason the two halves are separate processes.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

SPIKE_DIR = Path(__file__).resolve().parent
if str(SPIKE_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKE_DIR))

import spikeconfig  # noqa: E402
import statefile  # noqa: E402
from emotion_smooth import EmotionSmoother  # noqa: E402

MODEL_URLS = {
    spikeconfig.FERPLUS_MODEL: (
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    ),
    spikeconfig.YUNET_MODEL: (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
}


# --- model download -------------------------------------------------------


def download_models() -> int:
    spikeconfig.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in MODEL_URLS.items():
        if path.is_file() and path.stat().st_size > 1024:
            print(f"  have {path.name} ({path.stat().st_size // 1024} KB)")
            continue
        print(f"  downloading {path.name} ...")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "armani-spike"})
            with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as out:
                out.write(response.read())
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            if path is spikeconfig.YUNET_MODEL:
                print("    (YuNet is optional — the Haar cascade fallback needs no download)")
                continue
            return 1
        print(f"    saved {path.name} ({path.stat().st_size // 1024} KB)")
    return 0


# --- face detection -------------------------------------------------------


class FaceDetector:
    """Largest-face crop, from YuNet if the model is present, else Haar."""

    def __init__(self) -> None:
        self._yunet = None
        if spikeconfig.YUNET_MODEL.is_file():
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(spikeconfig.YUNET_MODEL), "", (320, 320), 0.7, 0.3, 5000
                )
                self.backend = "YuNet"
            except Exception as exc:
                print(f"  YuNet unavailable ({exc}); using the Haar cascade")
                self._yunet = None
        if self._yunet is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                raise RuntimeError(f"could not load the Haar cascade at {cascade_path}")
            self.backend = "Haar"

    def largest_face(self, frame) -> tuple[int, int, int, int] | None:
        """Bounding box (x, y, w, h) of the biggest face, or None."""
        height, width = frame.shape[:2]
        if self._yunet is not None:
            self._yunet.setInputSize((width, height))
            _, faces = self._yunet.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            box = max(faces, key=lambda f: f[2] * f[3])[:4]
            return int(box[0]), int(box[1]), int(box[2]), int(box[3])

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


# --- emotion classification ----------------------------------------------


class EmotionClassifier:
    """FER+ ONNX. Returns a normalised emotion name and a 0-1 score."""

    def __init__(self) -> None:
        import onnxruntime as ort

        if not spikeconfig.FERPLUS_MODEL.is_file():
            raise FileNotFoundError(
                f"FER+ model not found at {spikeconfig.FERPLUS_MODEL}. "
                "Run: python detector.py --download"
            )
        # CoreML first on the Mac, CPU as the guaranteed fallback.
        providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(str(spikeconfig.FERPLUS_MODEL), providers=providers)
        self._input = self._session.get_inputs()[0].name
        self._output = self._session.get_outputs()[0].name
        self.provider = self._session.get_providers()[0]

    def classify(self, face_bgr) -> tuple[str, float]:
        grey = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grey, (64, 64)).astype(np.float32)
        tensor = resized.reshape(1, 1, 64, 64)
        logits = self._session.run([self._output], {self._input: tensor})[0][0]
        probs = _softmax(logits)
        index = int(np.argmax(probs))
        raw = spikeconfig.FERPLUS_CLASSES[index]
        return spikeconfig.normalise_emotion(raw), float(probs[index])


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


# --- camera ---------------------------------------------------------------


def list_cameras(max_index: int) -> list[int]:
    """Indices that actually deliver a frame. Printed so the operator can choose."""
    working = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened() and cap.read()[0]:
                working.append(index)
        finally:
            cap.release()
    return working


# --- main loop ------------------------------------------------------------


def run(watch: bool) -> int:
    print("cameras that respond:", list_cameras(spikeconfig.FACE_CAM_MAX_PROBE) or "none")
    print(f"opening camera index {spikeconfig.FACE_CAM_INDEX} "
          f"(the LAPTOP webcam — NOT the C920)")

    faces = FaceDetector()
    print(f"face backend: {faces.backend}")
    classifier = EmotionClassifier()
    print(f"emotion model: FER+ on {classifier.provider}")
    print(f"config: {spikeconfig.describe()}")
    print("watch mode: NOT writing the state file\n" if watch else "publishing to "
          f"{spikeconfig.EMOTION_STATE_PATH}\n")

    smoother = EmotionSmoother(spikeconfig.WINDOW_FRAMES, spikeconfig.HOLD_S, spikeconfig.MIN_SCORE)
    cap = cv2.VideoCapture(spikeconfig.FACE_CAM_INDEX)
    if not cap.isOpened():
        print(f"could not open camera index {spikeconfig.FACE_CAM_INDEX}. "
              "Try another index (see the list above) or grant Camera permission.",
              file=sys.stderr)
        return 1

    fps_times: deque[float] = deque(maxlen=30)
    last_write = 0.0
    emitted = statefile.NO_EMOTION
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            start = time.perf_counter()

            box = faces.largest_face(frame)
            emotion, score = statefile.NO_EMOTION, 0.0
            if box is not None:
                x, y, w, h = _clamp_box(box, frame.shape)
                if w > 0 and h > 0:
                    emotion, score = classifier.classify(frame[y:y + h, x:x + w])
                    stable = smoother.update(emotion, score, time.monotonic())
                    if stable is not None:
                        emitted = stable
                        if not watch:
                            now = time.monotonic()
                            if now - last_write >= spikeconfig.WRITE_MIN_INTERVAL_S:
                                statefile.write_emotion(stable, score)
                                last_write = now

            fps_times.append(time.perf_counter())
            fps = _fps(fps_times)

            if watch:
                _overlay(frame, box, emotion, score, emitted, fps, faces.backend)
                cv2.imshow("emotion detector (--watch) — q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Headless: a light heartbeat so the operator sees it is alive.
                if int(start) % 2 == 0:
                    print(f"\r  fps {fps:4.1f} | frame {emotion:<8} {score:0.2f} "
                          f"| emitting {emitted:<8}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        cap.release()
        if watch:
            cv2.destroyAllWindows()
    return 0


def _clamp_box(box, shape) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    w, h = min(w, width - x), min(h, height - y)
    return x, y, w, h


def _fps(times: deque[float]) -> float:
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0 else 0.0


def _overlay(frame, box, emotion, score, emitted, fps, backend) -> None:
    if box is not None:
        x, y, w, h = _clamp_box(box, frame.shape)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 120), 2)
        cv2.putText(frame, f"{emotion} {score:0.2f}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 120), 2, cv2.LINE_AA)
    banner = f"emitting: {emitted}   |   {fps:0.1f} FPS   |   {backend}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(frame, banner, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (240, 240, 240), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--download", action="store_true", help="fetch the ONNX models and exit")
    parser.add_argument("--watch", action="store_true",
                        help="show the cam + detection and write NOTHING (validate the eyes)")
    args = parser.parse_args()

    print("=== emotion detector (spike) ===")
    if args.download:
        return download_models()
    try:
        return run(watch=args.watch)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
