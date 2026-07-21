"""The whole safety of the reflex loop, as one pure function.

``decide()`` answers "should the arm fire a gesture right now, and if not, why
not?" given the current emotion and the loop's memory. It is deliberately pure —
no clock reads, no arm, no I/O — so every branch (mapped, cooldown, repeat
suppression, arm-busy) is unit-tested without hardware. ``reflex.py`` supplies
``now`` from ``time.monotonic()`` and does the actual playing.
"""

from __future__ import annotations

from dataclasses import dataclass

from statefile import NO_EMOTION


@dataclass(frozen=True)
class ReflexMemory:
    """What the loop remembers between polls."""

    last_fire_ts: float = 0.0  # monotonic time of the last gesture, 0 = never
    last_fired_emotion: str = ""  # which emotion it last reacted to


@dataclass(frozen=True)
class Decision:
    fire: bool
    gesture: str | None
    reason: str  # always set — the audit trail wants a why for a skip too


def decide(
    emotion: str,
    *,
    stale: bool,
    arm_busy: bool,
    memory: ReflexMemory,
    now: float,
    emotion_gesture_map: dict[str, str],
    cooldown_s: float,
    allow_repeat: bool,
) -> Decision:
    """Return whether to fire, which gesture, and the reason either way.

    The conditions, in the order a human would check them:
      1. there is a live emotion at all (not stale / none)
      2. the arm is free (single-threaded, so this is really a belt-and-braces)
      3. the emotion is mapped to a gesture
      4. the cooldown since the last gesture has elapsed
      5. unless allow_repeat, it is not the very same emotion we just reacted to
    """
    if stale or emotion in ("", NO_EMOTION):
        return Decision(False, None, "no emotion")

    if arm_busy:
        return Decision(False, None, "arm busy")

    gesture = emotion_gesture_map.get(emotion)
    if gesture is None:
        return Decision(False, None, f"{emotion} is not mapped to a gesture")

    elapsed = now - memory.last_fire_ts
    if memory.last_fire_ts > 0.0 and elapsed < cooldown_s:
        return Decision(False, gesture, f"cooldown ({elapsed:.0f}s of {cooldown_s:.0f}s)")

    if not allow_repeat and emotion == memory.last_fired_emotion:
        # The face has stayed on the same emotion since our last reaction. React
        # once, then wait for it to change — a reflex, not a nag.
        return Decision(False, gesture, f"already reacted to {emotion}")

    return Decision(True, gesture, f"{emotion} -> {gesture}")
