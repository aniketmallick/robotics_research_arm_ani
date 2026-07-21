"""Atomic hand-off of the current emotion between the two processes.

The detector (own venv) writes; the reflex (lerobot env) reads. They never share
memory, so the file is the whole channel. It is written exactly the way
``armani/uistate.py`` writes ``ui_state.json`` — temp file in the same directory
plus ``os.replace`` — so a reader polling mid-write sees the old file or the new
one, never half of either.

Pure stdlib on purpose: this module is imported from both venvs, and the
detector's venv has no ``armani``. It also never raises — a write failing must
not take the detector down, and a read failing must not take the arm down. A bad
read resolves to "no emotion", exactly like a missing file.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from spikeconfig import EMOTION_STATE_PATH, STALE_S

# What the reflex sees when there is nothing usable to react to.
NO_EMOTION = "none"


def write_emotion(emotion: str, score: float, path: Path = EMOTION_STATE_PATH) -> bool:
    """Publish the current stable emotion. Best-effort; never raises."""
    payload = {"emotion": emotion, "score": round(float(score), 3), "ts": time.time()}
    handle = None
    temp_name = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent),
            prefix=".emotion_state-", suffix=".tmp", delete=False,
        )
        temp_name = handle.name
        json.dump(payload, handle)
        handle.flush()
        handle.close()
        handle = None
        os.replace(temp_name, path)
        return True
    except Exception:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        return False


def read_emotion(path: Path = EMOTION_STATE_PATH, stale_s: float = STALE_S) -> dict[str, Any]:
    """Return ``{"emotion", "score", "ts", "stale"}``. Always usable.

    Missing, unreadable, malformed or stale all resolve to ``emotion="none"``
    with ``stale=True`` — the reflex treats that as "react to nothing".
    """
    none = {"emotion": NO_EMOTION, "score": 0.0, "ts": 0.0, "stale": True}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return none
    if not isinstance(payload, dict):
        return none

    emotion = payload.get("emotion")
    if not isinstance(emotion, str) or not emotion:
        return none

    try:
        ts = float(payload.get("ts") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    if time.time() - ts > stale_s:
        return {**none, "ts": ts, "last_emotion": emotion}

    try:
        score = float(payload.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {"emotion": emotion, "score": score, "ts": ts, "stale": False}
