"""Shared, dependency-free config for the emotion-reflex spike.

Deliberately pure stdlib so BOTH sides can import it:

* ``detector.py`` runs in its own venv (cv2 + onnxruntime + numpy) and cannot
  see the ``armani`` package.
* ``reflex.py`` runs in the lerobot conda env alongside ``armani``.

So the two processes agree on the state-file path, the camera index and the
emotion→gesture map through this one module rather than through anything in
``armani``. Nothing here imports ``armani`` — that is what keeps the demo
pipeline untouched (``git diff`` against demo-freeze stays inside this folder).

Everything is overridable by ``ARMANI_REFLEX_*`` environment variables so the
operator can tune the spike without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

# experiments/emotion_reflex/spikeconfig.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
SPIKE_DIR = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "logs"

# The atomic hand-off file. detector.py writes it, reflex.py reads it. It lives
# under the repo's logs/ so it sits next to decisions.jsonl and ui_state.json.
EMOTION_STATE_PATH = LOG_DIR / "emotion_state.json"

# Where the downloaded ONNX models live (gitignored). See README for the URLs.
MODELS_DIR = SPIKE_DIR / "models"
FERPLUS_MODEL = MODELS_DIR / "emotion-ferplus-8.onnx"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Camera ---------------------------------------------------------------
# The LAPTOP built-in (FaceTime) webcam, NOT the C920. The C920 is locked to the
# arm's workspace homography and must never be opened here. Indices are not
# stable across machines, so detector.py --watch prints the ones that respond
# and the operator sets this. Default 0 is the usual built-in on a MacBook, but
# on this rig the C920 may also be 0 — CONFIRM with --watch before trusting it.
FACE_CAM_INDEX = _env_int("ARMANI_REFLEX_CAM_INDEX", 0)
FACE_CAM_MAX_PROBE = _env_int("ARMANI_REFLEX_CAM_MAX_PROBE", 4)

# --- Detector debounce ----------------------------------------------------
# A rolling window of the last N per-frame classifications. An emotion is only
# EMITTED when one class holds a majority of the window AND that has been true
# for at least HOLD_S seconds AND its mean score clears MIN_SCORE. This is what
# turns a twitchy per-frame classifier into a stable signal.
WINDOW_FRAMES = _env_int("ARMANI_REFLEX_WINDOW", 15)
HOLD_S = _env_float("ARMANI_REFLEX_HOLD_S", 1.0)
MIN_SCORE = _env_float("ARMANI_REFLEX_MIN_SCORE", 0.55)
# Do not rewrite the state file faster than this, even when the emotion is
# stable — the reader only polls a few Hz and the disk write is pointless spam.
WRITE_MIN_INTERVAL_S = _env_float("ARMANI_REFLEX_WRITE_INTERVAL_S", 0.25)

# --- Reflex (arm side) ----------------------------------------------------
# How often reflex.py polls the state file.
POLL_HZ = _env_float("ARMANI_REFLEX_POLL_HZ", 7.0)
# A state file older than this is "no emotion" — the detector died or the face
# left. Must be comfortably longer than the detector's write interval.
STALE_S = _env_float("ARMANI_REFLEX_STALE_S", 3.0)
# Minimum seconds between two reflex gestures. The whole "don't spam the arm"
# guarantee. 15s is deliberately unhurried for a reflex demo.
COOLDOWN_S = _env_float("ARMANI_REFLEX_COOLDOWN_S", 15.0)
# When False (default), the same emotion held continuously fires ONCE and then
# waits for the emotion to change before reacting again — so a persistently sad
# face is not cheered-up every cooldown. Set True to react every cooldown.
ALLOW_REPEAT = _env_flag("ARMANI_REFLEX_ALLOW_REPEAT", False)

# --- Emotion → gesture map ------------------------------------------------
# Keys are the NORMALISED emotion names the detector emits (see EMOTION_ALIASES
# below). Values are gesture names that MUST exist in armani_gestures — verified
# against config.GESTURES at build time: bow, wave, dance, nod_yes, shake_no,
# look_around, celebrate, sad_droop.
#
# The rationale, so the operator can retune sensibly:
#   happy    -> nod_yes     acknowledge the good mood
#   sad      -> celebrate   the "cheer up" gesture
#   surprise -> look_around mirror the surprise
#   angry    -> bow         placate / apologise
# neutral / disgust / fear are intentionally UNMAPPED — a reflex should stay
# quiet rather than fire something that does not fit.
_DEFAULT_MAP = {
    "happy": "nod_yes",
    "sad": "celebrate",
    "surprise": "look_around",
    "angry": "bow",
}


def _parse_map(raw: str | None) -> dict[str, str]:
    """Parse ARMANI_REFLEX_MAP="happy:nod_yes,sad:celebrate" into a dict.

    A malformed entry is skipped, not fatal — a typo in an env var must not stop
    the reflex from running with the entries that did parse.
    """
    if not raw or not raw.strip():
        return dict(_DEFAULT_MAP)
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        emotion, _, gesture = pair.partition(":")
        emotion, gesture = emotion.strip().lower(), gesture.strip()
        if emotion and gesture:
            result[emotion] = gesture
    return result or dict(_DEFAULT_MAP)


EMOTION_GESTURE_MAP = _parse_map(os.getenv("ARMANI_REFLEX_MAP"))

# FER+ emits eight classes; these fold the awkward ones onto friendly keys and
# leave the rest as-is. contempt collapses to angry (closest expressive match);
# the operator can remap by editing EMOTION_GESTURE_MAP.
EMOTION_ALIASES = {
    "happiness": "happy",
    "sadness": "sad",
    "anger": "angry",
    "contempt": "angry",
}

# The class order the emotion-ferplus-8 model outputs, before aliasing.
FERPLUS_CLASSES = (
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
)


def normalise_emotion(name: str) -> str:
    return EMOTION_ALIASES.get(name, name)


def describe() -> str:
    """One human-readable summary of the live config, for a script banner."""
    mapped = ", ".join(f"{e}->{g}" for e, g in EMOTION_GESTURE_MAP.items())
    return (
        f"cam index {FACE_CAM_INDEX} | window {WINDOW_FRAMES} frames, hold {HOLD_S}s, "
        f"min score {MIN_SCORE} | cooldown {COOLDOWN_S}s, stale {STALE_S}s | map: {mapped}"
    )
