"""Cheap, hardware-free tests for the emotion-reflex spike.

Covers only the pure logic — the atomic state file, the debounce smoother, the
reflex decision, and the config-map parser. No camera, no ONNX, no arm, so it
runs in the lerobot env (or any Python) and does not touch the main test suite
(pytest's testpaths=tests keeps it out of a bare `pytest` run).

    pytest experiments/emotion_reflex/tests/ -q
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

SPIKE_DIR = Path(__file__).resolve().parents[1]
if str(SPIKE_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKE_DIR))

import spikeconfig  # noqa: E402
import statefile  # noqa: E402
from emotion_smooth import EmotionSmoother  # noqa: E402
from reflex_rules import ReflexMemory, decide  # noqa: E402


# --- the atomic state file -----------------------------------------------


def test_write_then_read_round_trip(tmp_path):
    path = tmp_path / "emotion_state.json"
    assert statefile.write_emotion("happy", 0.87, path=path)
    state = statefile.read_emotion(path=path, stale_s=10)
    assert state["emotion"] == "happy"
    assert state["score"] == pytest.approx(0.87)
    assert state["stale"] is False


def test_missing_file_reads_as_no_emotion(tmp_path):
    state = statefile.read_emotion(path=tmp_path / "nope.json")
    assert state["emotion"] == statefile.NO_EMOTION
    assert state["stale"] is True


def test_corrupt_file_reads_as_no_emotion(tmp_path):
    path = tmp_path / "emotion_state.json"
    path.write_text("{ not json")
    assert statefile.read_emotion(path=path)["emotion"] == statefile.NO_EMOTION


def test_non_object_json_reads_as_no_emotion(tmp_path):
    path = tmp_path / "emotion_state.json"
    path.write_text('["happy"]')
    assert statefile.read_emotion(path=path)["emotion"] == statefile.NO_EMOTION


def test_stale_file_reads_as_no_emotion(tmp_path):
    path = tmp_path / "emotion_state.json"
    path.write_text(json.dumps({"emotion": "sad", "score": 0.9, "ts": time.time() - 60}))
    state = statefile.read_emotion(path=path, stale_s=3)
    assert state["emotion"] == statefile.NO_EMOTION
    assert state["stale"] is True
    assert state["last_emotion"] == "sad"  # kept for debugging


def test_bad_timestamp_is_treated_as_stale(tmp_path):
    path = tmp_path / "emotion_state.json"
    path.write_text(json.dumps({"emotion": "happy", "score": 0.9, "ts": "soon"}))
    assert statefile.read_emotion(path=path)["emotion"] == statefile.NO_EMOTION


def test_write_never_raises_on_a_bad_path():
    # A directory that cannot be created must be swallowed, not raised.
    bad = Path("/does/not/exist/and/cannot/emotion_state.json")
    assert statefile.write_emotion("happy", 0.5, path=bad) is False


def test_reader_never_sees_a_partial_file(tmp_path):
    """The reflex polls while the detector writes; prove reads are whole."""
    path = tmp_path / "emotion_state.json"
    statefile.write_emotion("neutral", 0.5, path=path)
    stop = threading.Event()
    torn: list[str] = []

    def reader():
        while not stop.is_set():
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if raw:
                try:
                    json.loads(raw)
                except ValueError:
                    torn.append(raw[:40])

    watcher = threading.Thread(target=reader, daemon=True)
    watcher.start()
    try:
        for i in range(300):
            statefile.write_emotion("happy", i / 300, path=path)
    finally:
        stop.set()
        watcher.join(timeout=2)
    assert torn == []


# --- the debounce smoother -----------------------------------------------


def test_smoother_needs_a_majority():
    s = EmotionSmoother(window=5, hold_s=0.0, min_score=0.0)
    # Alternating classes never give a strict majority.
    assert s.update("happy", 0.9, now=0.0) is None
    assert s.update("sad", 0.9, now=0.1) is None
    assert s.update("happy", 0.9, now=0.2) is None
    assert s.update("sad", 0.9, now=0.3) is None


def test_smoother_needs_the_hold_time():
    s = EmotionSmoother(window=5, hold_s=1.0, min_score=0.0)
    # Majority is reached immediately, but the hold time has not elapsed.
    assert s.update("happy", 0.9, now=0.0) is None
    assert s.update("happy", 0.9, now=0.1) is None
    assert s.update("happy", 0.9, now=0.5) is None
    # ...now it has.
    assert s.update("happy", 0.9, now=1.2) == "happy"


def test_smoother_respects_the_score_floor():
    s = EmotionSmoother(window=3, hold_s=0.0, min_score=0.6)
    assert s.update("happy", 0.4, now=0.0) is None
    assert s.update("happy", 0.4, now=0.1) is None
    assert s.update("happy", 0.4, now=0.2) is None  # majority + held, but score too low


def test_a_changing_majority_resets_the_hold():
    s = EmotionSmoother(window=3, hold_s=1.0, min_score=0.0)
    for t in (0.0, 0.1, 0.2):
        s.update("happy", 0.9, now=t)
    assert s.update("happy", 0.9, now=1.5) == "happy"
    # One sad frame: the window is [happy, happy, sad], so happy is STILL the
    # majority — a single stray frame does not flip the signal (hysteresis).
    assert s.update("sad", 0.9, now=1.6) == "happy"
    # A second sad frame makes sad the majority: now the winner changes, so the
    # hold timer restarts and it stops emitting until sad has held long enough.
    assert s.update("sad", 0.9, now=1.7) is None
    assert s.update("sad", 0.9, now=1.9) is None  # sad held only 0.2s
    assert s.update("sad", 0.9, now=2.9) == "sad"  # sad held >= 1.0s


# --- the reflex decision -------------------------------------------------

MAP = {"happy": "nod_yes", "sad": "celebrate"}


def base(**kw):
    args = dict(
        stale=False, arm_busy=False, memory=ReflexMemory(), now=100.0,
        emotion_gesture_map=MAP, cooldown_s=15.0, allow_repeat=False,
    )
    args.update(kw)
    return args


def test_fires_a_mapped_emotion():
    d = decide("happy", **base())
    assert d.fire and d.gesture == "nod_yes"


def test_no_emotion_does_not_fire():
    assert not decide("none", **base(stale=True)).fire
    assert not decide("", **base()).fire


def test_unmapped_emotion_does_not_fire():
    d = decide("disgust", **base())
    assert not d.fire
    assert "not mapped" in d.reason


def test_cooldown_suppresses():
    mem = ReflexMemory(last_fire_ts=100.0, last_fired_emotion="sad")
    d = decide("happy", **base(memory=mem, now=105.0))  # 5s < 15s cooldown
    assert not d.fire
    assert "cooldown" in d.reason


def test_cooldown_elapsed_allows_a_different_emotion():
    mem = ReflexMemory(last_fire_ts=100.0, last_fired_emotion="sad")
    d = decide("happy", **base(memory=mem, now=120.0))  # 20s > 15s
    assert d.fire and d.gesture == "nod_yes"


def test_same_emotion_is_not_repeated_by_default():
    mem = ReflexMemory(last_fire_ts=100.0, last_fired_emotion="happy")
    d = decide("happy", **base(memory=mem, now=200.0))  # cooldown long past
    assert not d.fire
    assert "already reacted" in d.reason


def test_allow_repeat_reacts_again_after_cooldown():
    mem = ReflexMemory(last_fire_ts=100.0, last_fired_emotion="happy")
    d = decide("happy", **base(memory=mem, now=200.0, allow_repeat=True))
    assert d.fire


def test_arm_busy_never_fires():
    assert not decide("happy", **base(arm_busy=True)).fire


def test_a_skip_always_gives_a_reason():
    for emotion, kw in [("none", {"stale": True}), ("disgust", {}), ("happy", {"arm_busy": True})]:
        assert decide(emotion, **base(**kw)).reason


# --- the config map parser -----------------------------------------------


def test_default_map_is_used_when_env_absent():
    assert spikeconfig._parse_map(None) == spikeconfig._DEFAULT_MAP
    assert spikeconfig._parse_map("") == spikeconfig._DEFAULT_MAP


def test_a_custom_map_parses():
    m = spikeconfig._parse_map("happy:wave, sad:bow")
    assert m == {"happy": "wave", "sad": "bow"}


def test_malformed_entries_are_skipped_not_fatal():
    m = spikeconfig._parse_map("happy:wave, garbage, :nothing, sad:")
    assert m == {"happy": "wave"}


def test_normalise_folds_ferplus_names():
    assert spikeconfig.normalise_emotion("happiness") == "happy"
    assert spikeconfig.normalise_emotion("sadness") == "sad"
    assert spikeconfig.normalise_emotion("anger") == "angry"
    assert spikeconfig.normalise_emotion("neutral") == "neutral"


def test_every_default_mapping_targets_a_real_gesture():
    """The map must only reference gestures the arm actually has. Checked against
    the REAL registry when armani is importable (the lerobot env), so a rename of
    a gesture would fail this rather than drift silently."""
    try:
        sys.path.insert(0, str(SPIKE_DIR.parents[1]))
        from armani import config as armani_config

        real = set(armani_config.GESTURES)
    except Exception:
        pytest.skip("armani not importable here; run this test in the lerobot env")
    for emotion, gesture in spikeconfig._DEFAULT_MAP.items():
        assert gesture in real, f"{emotion} -> {gesture} is not a recorded gesture"
