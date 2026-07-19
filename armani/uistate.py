"""What ARM-ANI is doing right now, for the face to mirror. Telemetry only.

Five states — ``idle``, ``listening``, ``thinking``, ``talking``, ``doing`` —
written to ``logs/ui_state.json`` by the agent and polled by the avatar page.

Three properties this file exists to guarantee:

1. **It never raises.** A UI-state write happens inside the voice loop and on
   the motion worker. A full disk must not take down a robot that is holding
   something. Every failure is swallowed, exactly like ``logutil``.
2. **A reader never sees half a file.** The page polls ~8x/second while we
   write; a truncated read would be a JSON error on screen. Writes go to a temp
   file in the same directory and land via ``os.replace``, which is atomic on
   POSIX and Windows alike.
3. **It is cheap to call from a hot path.** ``talking`` is published on every
   audio chunk — many per second. Publishing only on an actual change turns
   that into one write per transition, so call sites stay one-liners and nobody
   has to remember where the state last was.

This is a MIRROR. Nothing here decides anything; if the file is missing, stale
or corrupt the reader reports ``idle`` and the demo carries on.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any

from armani import config
from armani.logutil import get_logger

log = get_logger("uistate")

IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
TALKING = "talking"
DOING = "doing"

STATES: tuple[str, ...] = (IDLE, LISTENING, THINKING, TALKING, DOING)

_lock = threading.Lock()
# The last (state, fields) actually written, so a hot path can call publish()
# on every audio chunk and only pay for real transitions.
_last: tuple | None = None


def publish(state: str, **fields: Any) -> None:
    """Record what the robot is doing. Best-effort; never raises.

    Unknown state names are still written — the page falls back to idle for
    anything it does not recognise, and refusing to publish would be a worse
    failure than showing the wrong face.
    """
    global _last
    if state not in STATES:
        log.debug("publishing unknown ui state %r", state)

    try:
        key = (state, tuple(sorted((k, repr(v)) for k, v in fields.items())))
    except Exception:
        key = None  # unhashable field; just write it

    with _lock:
        if key is not None and key == _last:
            return
        payload = {"state": state, "ts": time.time(), **fields}
        if _write_atomic(payload):
            _last = key


def _write_atomic(payload: dict) -> bool:
    """Write the state so a concurrent reader sees old or new, never half."""
    path = config.UI_STATE_PATH
    handle = None
    temp_name = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Same directory as the target: os.replace is only atomic within a
        # filesystem, and /tmp is frequently a different one.
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent),
            prefix=".ui_state-", suffix=".tmp", delete=False,
        )
        temp_name = handle.name
        json.dump(payload, handle, default=str)
        handle.flush()
        handle.close()
        handle = None
        os.replace(temp_name, path)
        return True
    except Exception as exc:
        log.debug("could not publish ui state: %s", exc)
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


def current() -> dict:
    """What to show. Always a usable dict, whatever the file is doing.

    Missing, unreadable, malformed or stale all resolve to idle: the face
    should go quiet when there is nothing driving it, not freeze mid-sentence
    because a session died while it was talking.
    """
    fallback = {"state": IDLE, "ts": 0.0, "stale": True}
    try:
        raw = config.UI_STATE_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, ValueError):
        return fallback
    if not isinstance(payload, dict):
        return fallback

    state = payload.get("state")
    if not isinstance(state, str) or not state:
        return fallback

    try:
        age = time.time() - float(payload.get("ts") or 0.0)
    except (TypeError, ValueError):
        age = float("inf")
    if age > config.UI_STATE_STALE_S:
        # Keep the original fields for debugging, but say idle.
        return {**payload, "state": IDLE, "stale": True}

    return {**payload, "stale": False}


def reset() -> None:
    """Back to idle, and forget the dedupe cache.

    Called when a session ends so the face does not sit there mid-sentence, and
    at start-up so the first publish always lands even if the previous run left
    the same state behind.
    """
    global _last
    with _lock:
        _last = None
    publish(IDLE)
