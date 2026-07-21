"""Turn a twitchy per-frame classifier into a stable emitted emotion.

A single frame of FER+ output is noisy: a smile flickers to "neutral" for one
frame, a raised eyebrow reads "surprise" for two. Firing the arm on any of that
would be a nervous tic, not a reflex. So we keep a rolling window of the last N
per-frame classifications and only EMIT an emotion when:

  * one class holds a strict majority of the window, AND
  * it has held that majority continuously for at least ``hold_s`` seconds, AND
  * its mean score over the winning frames clears ``min_score``.

Pure stdlib and fully deterministic given a clock, so it is unit-tested without
a camera.
"""

from __future__ import annotations

from collections import deque


class EmotionSmoother:
    """Rolling-window majority vote with a hold time and a score floor."""

    def __init__(self, window: int, hold_s: float, min_score: float) -> None:
        self._window: deque[tuple[str, float]] = deque(maxlen=max(1, window))
        self._hold_s = hold_s
        self._min_score = min_score
        # The class that currently holds the majority, and WHEN it started doing
        # so — the hold timer resets whenever the majority winner changes.
        self._holding: str | None = None
        self._holding_since: float = 0.0

    def update(self, emotion: str, score: float, now: float) -> str | None:
        """Feed one per-frame classification. Returns the emitted emotion or None.

        None means "not stable enough yet" — the caller should not act.
        """
        self._window.append((emotion, score))
        winner, share, mean_score = self._majority()

        if winner is None or share <= 0.5:
            # No strict majority: nothing is holding.
            self._holding = None
            return None

        if winner != self._holding:
            self._holding = winner
            self._holding_since = now
            return None

        held_long_enough = (now - self._holding_since) >= self._hold_s
        if held_long_enough and mean_score >= self._min_score:
            return winner
        return None

    def _majority(self) -> tuple[str | None, float, float]:
        """The most common class in the window, its share, and its mean score."""
        if not self._window:
            return None, 0.0, 0.0
        counts: dict[str, int] = {}
        score_sums: dict[str, float] = {}
        for emotion, score in self._window:
            counts[emotion] = counts.get(emotion, 0) + 1
            score_sums[emotion] = score_sums.get(emotion, 0.0) + score
        winner = max(counts, key=lambda e: counts[e])
        share = counts[winner] / len(self._window)
        mean_score = score_sums[winner] / counts[winner]
        return winner, share, mean_score

    @property
    def window_fill(self) -> int:
        return len(self._window)
