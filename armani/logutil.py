"""Append-only JSONL decision log plus a plain console logger.

Every gate evaluation, tool call and safety event lands in logs/decisions.jsonl.
Stage 1 only uses the safety events; the trust gates fill in the rest later.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from typing import Any

from armani import config

_write_lock = threading.Lock()

_console = logging.getLogger("armani")
if not _console.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _console.addHandler(_handler)
    _console.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return _console.getChild(name)


def log_event(kind: str, **fields: Any) -> None:
    """Append one event to the decision log.

    Logging must never take down a motion loop, so any I/O failure here is
    reported to the console and swallowed.
    """
    record = {"ts": time.time(), "kind": kind, **fields}
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str)
        with _write_lock, config.DECISION_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        _console.warning("could not write decision log: %s", exc)
