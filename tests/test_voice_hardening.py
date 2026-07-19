"""Push-to-talk robustness in run_agent.py. No microphone, no session.

Two failure modes that showed up as session errors mid-demo:
  * a stray tap committing an audio buffer the API rejects as too short
  * a barge-in truncate losing the race against playback
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_agent  # noqa: E402

from armani import config  # noqa: E402


# --- the too-short-hold guard -------------------------------------------


def held_frames(frames: int) -> int:
    """Reproduce the counter run_agent uses across its two callback threads."""
    counter = itertools.count()
    start = next(counter)                     # on_press
    for _ in range(frames):
        next(counter)                         # audio_callback, once per block
    return next(counter) - start - 1          # on_release


@pytest.mark.parametrize("frames", [0, 1, 5, 50, 500])
def test_the_counter_reports_exactly_what_was_captured(frames):
    assert held_frames(frames) == frames


def test_minimum_is_derived_from_the_pinned_audio_settings():
    """20 ms blocks at 24 kHz -> 5 blocks to clear the API's 100 ms floor."""
    block_ms = 1000 * config.AUDIO_BLOCKSIZE / config.AUDIO_SAMPLE_RATE
    assert run_agent.MIN_COMMIT_FRAMES * block_ms >= run_agent.MIN_COMMIT_MS
    # ...and not wastefully more than one block over.
    assert (run_agent.MIN_COMMIT_FRAMES - 1) * block_ms < run_agent.MIN_COMMIT_MS


def test_a_tap_that_captured_nothing_is_below_the_commit_threshold():
    assert held_frames(0) < run_agent.MIN_COMMIT_FRAMES


def test_a_very_short_tap_is_still_below_the_threshold():
    """The bug this fixes: >0 frames but still too short for the API."""
    assert held_frames(1) < run_agent.MIN_COMMIT_FRAMES


def test_a_real_hold_clears_the_threshold():
    assert held_frames(25) >= run_agent.MIN_COMMIT_FRAMES  # ~half a second


# --- the harmless-audio-error filter ------------------------------------


@pytest.mark.parametrize("detail", [
    "Error: invalid_value: Audio content of 120.00ms is already shorter than 200.00ms",
    {"type": "invalid_request_error", "code": "invalid_value", "param": "audio_end_ms"},
    "invalid_value",
])
def test_truncate_races_are_recognised_as_harmless(detail):
    assert run_agent._is_harmless_audio_error(detail)


@pytest.mark.parametrize("detail", [
    "input_audio_buffer_commit_empty: the buffer is too small",
    "websocket connection closed unexpectedly",
    "rate limit exceeded",
    "insufficient_quota",
    "",
])
def test_real_errors_are_not_swallowed(detail):
    """Especially the empty-commit error: we now prevent it, so if it ever fires
    again it means the guard broke and it must be visible."""
    assert not run_agent._is_harmless_audio_error(detail)
