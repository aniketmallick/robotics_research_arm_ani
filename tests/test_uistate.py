"""The UI state publisher. No server, no agent, no arm."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import config, uistate  # noqa: E402


@pytest.fixture(autouse=True)
def state_file(tmp_path, monkeypatch):
    """Point the publisher at a scratch file and clear its dedupe cache."""
    target = tmp_path / "ui_state.json"
    monkeypatch.setattr(config, "UI_STATE_PATH", target)
    monkeypatch.setattr(uistate, "_last", None)
    return target


# --- publishing ----------------------------------------------------------


def test_publish_writes_the_state(state_file):
    uistate.publish(uistate.LISTENING)
    assert json.loads(state_file.read_text())["state"] == "listening"


def test_publish_carries_extra_fields(state_file):
    uistate.publish(uistate.DOING, action="gesture bow")
    payload = json.loads(state_file.read_text())
    assert payload["action"] == "gesture bow"
    assert payload["ts"] > 0


def test_repeated_publishes_do_not_rewrite(state_file):
    """'talking' is published on every audio chunk; only transitions cost a write."""
    uistate.publish(uistate.TALKING)
    first = state_file.stat().st_mtime_ns
    for _ in range(50):
        uistate.publish(uistate.TALKING)
    assert state_file.stat().st_mtime_ns == first


def test_a_changed_field_is_a_real_transition(state_file):
    uistate.publish(uistate.DOING, action="gesture bow")
    uistate.publish(uistate.DOING, action="pick from front-left")
    assert json.loads(state_file.read_text())["action"] == "pick from front-left"


def test_publish_never_raises_when_the_path_is_unwritable(monkeypatch, tmp_path):
    """A full disk must not take down a robot that is holding something."""
    monkeypatch.setattr(config, "UI_STATE_PATH", tmp_path / "no" / "such" / "dir" / "s.json")
    monkeypatch.setattr(uistate, "_last", None)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(uistate.tempfile, "NamedTemporaryFile", boom)
    uistate.publish(uistate.TALKING)  # must not raise


def test_a_failed_write_is_retried_next_time(state_file, monkeypatch):
    """A publish that failed must not be deduped away as 'already published'."""
    calls = {"n": 0}
    real = uistate._write_atomic

    def flaky(payload):
        calls["n"] += 1
        return False if calls["n"] == 1 else real(payload)

    monkeypatch.setattr(uistate, "_write_atomic", flaky)
    uistate.publish(uistate.TALKING)
    assert not state_file.exists()
    uistate.publish(uistate.TALKING)
    assert json.loads(state_file.read_text())["state"] == "talking"


def test_unknown_states_are_still_written(state_file):
    """Refusing to publish would be a worse failure than an odd face."""
    uistate.publish("rebooting")
    assert json.loads(state_file.read_text())["state"] == "rebooting"


def test_no_temp_files_are_left_behind(state_file):
    for state in uistate.STATES:
        uistate.publish(state)
    leftovers = list(state_file.parent.glob(".ui_state-*"))
    assert leftovers == []


# --- reading -------------------------------------------------------------


def test_missing_file_reads_as_idle(state_file):
    assert state_file.exists() is False
    assert uistate.current()["state"] == "idle"
    assert uistate.current()["stale"] is True


def test_corrupt_file_reads_as_idle(state_file):
    state_file.write_text("{ not json")
    assert uistate.current()["state"] == "idle"


def test_non_object_json_reads_as_idle(state_file):
    state_file.write_text('["listening"]')
    assert uistate.current()["state"] == "idle"


def test_missing_state_key_reads_as_idle(state_file):
    state_file.write_text('{"ts": 1}')
    assert uistate.current()["state"] == "idle"


def test_fresh_state_is_reported(state_file):
    uistate.publish(uistate.DOING, action="pick")
    payload = uistate.current()
    assert payload["state"] == "doing"
    assert payload["stale"] is False
    assert payload["action"] == "pick"


def test_stale_state_reads_as_idle(state_file):
    """A session that died mid-sentence must not leave the mascot talking."""
    state_file.write_text(json.dumps({
        "state": "talking",
        "ts": time.time() - config.UI_STATE_STALE_S - 5,
    }))
    payload = uistate.current()
    assert payload["state"] == "idle"
    assert payload["stale"] is True


def test_a_bad_timestamp_is_treated_as_stale(state_file):
    state_file.write_text(json.dumps({"state": "talking", "ts": "not a number"}))
    assert uistate.current()["state"] == "idle"


def test_reset_returns_to_idle(state_file):
    uistate.publish(uistate.TALKING)
    uistate.reset()
    assert uistate.current()["state"] == "idle"


def test_reset_clears_the_dedupe_cache(state_file):
    """After a reset, re-publishing the same state must land."""
    uistate.publish(uistate.TALKING)
    uistate.reset()
    uistate.publish(uistate.TALKING)
    assert uistate.current()["state"] == "talking"


# --- atomicity -----------------------------------------------------------


def test_a_reader_never_sees_a_partial_file(state_file):
    """The page polls while the agent writes. Prove readers only ever see whole
    documents — this is the entire reason for the temp-file + os.replace dance."""
    stop = threading.Event()
    torn: list[str] = []

    def reader():
        while not stop.is_set():
            try:
                raw = state_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if not raw:
                continue
            try:
                json.loads(raw)
            except ValueError:
                torn.append(raw[:60])

    watcher = threading.Thread(target=reader, daemon=True)
    watcher.start()
    try:
        for index in range(300):
            # Vary a field so every call is a real write, and pad it so a
            # non-atomic implementation would be caught mid-flush.
            uistate.publish(uistate.DOING, action=f"job-{index}", pad="x" * 4000)
    finally:
        stop.set()
        watcher.join(timeout=2)

    assert torn == [], f"reader saw {len(torn)} partial file(s): {torn[:2]}"


def test_concurrent_publishers_do_not_corrupt_the_file(state_file):
    """The event loop and the motion worker both publish."""
    errors: list[Exception] = []

    def spam(state):
        try:
            for index in range(80):
                uistate.publish(state, n=index)
        except Exception as exc:  # pragma: no cover - the point is that it cannot
            errors.append(exc)

    threads = [threading.Thread(target=spam, args=(s,)) for s in
               (uistate.TALKING, uistate.DOING, uistate.LISTENING)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert uistate.current()["state"] in uistate.STATES
