#!/usr/bin/env python
"""Emotion → gesture reflex (arm side). SYSTEM-1 PROTOTYPE — experimental spike.

Reads the emotion the detector publishes to logs/emotion_state.json and, when a
mapped emotion has held and the cooldown has elapsed, replays the matching
recorded gesture. It generates NO motion of its own: it only triggers an already
recorded, already clamped macro through ``gestures.play_gesture``.

    python reflex.py                          # DRY RUN (default): prints, never moves
    python reflex.py --sim-emotion sad        # dry-run the arm path with no camera
    python reflex.py --live --once            # ONE real gesture, operator present
    python reflex.py --live                   # full live loop

Runs in the lerobot conda env (it imports armani). The detector runs separately
in its own venv. Re-read CLAUDE.md before --live: this moves the arm, so the
operator-present gate and the freeze kill switch are wired exactly as the voice
agent wires them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKE_DIR.parents[1]
for _p in (str(SPIKE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spikeconfig  # noqa: E402
import statefile  # noqa: E402
from reflex_rules import Decision, ReflexMemory, decide  # noqa: E402

from armani import gestures, motion, safety  # noqa: E402
from armani.logutil import get_logger, log_event  # noqa: E402

log = get_logger("reflex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--live", action="store_true",
        help="really move the arm (prompts for operator presence, arms the kill switch)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print what it WOULD do and never move (this is the default)",
    )
    parser.add_argument(
        "--sim-emotion", metavar="E",
        help="inject one emotion every poll instead of reading the camera state file",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="fire a single reflex then exit — the safest first live test",
    )
    return parser.parse_args()


def validate_map() -> dict[str, str]:
    """Drop any mapped gesture that is not a real recorded gesture, loudly."""
    known = set(gestures.list_gestures())
    good: dict[str, str] = {}
    for emotion, gesture in spikeconfig.EMOTION_GESTURE_MAP.items():
        if gesture in known:
            good[emotion] = gesture
        else:
            log.warning(
                "emotion %r maps to %r which is not a recorded gesture (%s) — dropping it",
                emotion, gesture, ", ".join(sorted(known)),
            )
    return good


def current_emotion(args: argparse.Namespace) -> dict:
    """Where the emotion comes from: the injected one, or the state file."""
    if args.sim_emotion:
        return {"emotion": args.sim_emotion.strip().lower(), "score": 1.0, "stale": False}
    return statefile.read_emotion()


def _publish_ui(state: str, **fields) -> None:
    """Best-effort mirror to the avatar screen, if uistate is importable."""
    try:
        from armani import uistate

        uistate.publish(state, **fields)
    except Exception:
        pass


def run(args: argparse.Namespace, arm, live: bool, emotion_map: dict[str, str]) -> int:
    period = 1.0 / max(spikeconfig.POLL_HZ, 0.5)
    memory = ReflexMemory()

    print(f"\nreflex loop running ({'LIVE' if live else 'DRY RUN'}). Ctrl-C to stop.")
    print(f"config: {spikeconfig.describe()}\n")

    while True:
        # A Ctrl-C that lands here (during the idle poll, not inside a gesture)
        # has set the stop flag but nobody has shown the freeze menu yet. Present
        # it, then stop. A mid-gesture Ctrl-C is handled below instead, because
        # play_gesture runs the menu itself — presenting it again here would show
        # it twice.
        if live and safety.stop_requested():
            print("\n[kill switch] stopping the reflex loop.")
            safety.handle_freeze(arm, arm.read_positions())
            safety.clear_stop()
            return 0

        state = current_emotion(args)
        emotion, score = state["emotion"], state.get("score", 0.0)
        now = time.monotonic()

        decision = decide(
            emotion,
            stale=state.get("stale", True),
            arm_busy=False,  # single-threaded + blocking: nothing runs while we decide
            memory=memory,
            now=now,
            emotion_gesture_map=emotion_map,
            cooldown_s=spikeconfig.COOLDOWN_S,
            allow_repeat=spikeconfig.ALLOW_REPEAT,
        )

        _log_decision(emotion, score, decision, live)

        if decision.fire:
            assert decision.gesture is not None
            if live:
                _play_live(arm, decision.gesture, emotion, score)
                if safety.stop_requested():
                    # The gesture was frozen by the kill switch, which already
                    # ran the operator menu inside play_gesture. Do not present
                    # it again — just clear the flag and stop.
                    safety.clear_stop()
                    return 0
            else:
                print(f"WOULD play {decision.gesture} for {emotion} (score {score:.2f})")
            memory = ReflexMemory(last_fire_ts=time.monotonic(), last_fired_emotion=emotion)
            if args.once:
                print("\n--once: fired one reflex, exiting.")
                return 0
        elif decision.reason not in ("no emotion",):
            # Skips on a real emotion are worth showing; "no emotion" every poll
            # would just be noise.
            print(f"  suppressed ({decision.reason})")

        time.sleep(period)


def _play_live(arm, gesture: str, emotion: str, score: float) -> None:
    print(f"\n>>> {emotion} (score {score:.2f}) -> playing {gesture}")
    _publish_ui("doing", action=f"reflex {gesture}")
    log_event("reflex_gesture_start", emotion=emotion, gesture=gesture, score=round(score, 3))
    try:
        gestures.play_gesture(arm, gesture)
        outcome = "done"
    except Exception as exc:
        outcome = f"error: {type(exc).__name__}: {exc}"
        log.error("reflex gesture %s failed: %s", gesture, exc)
    finally:
        _publish_ui("idle")
    log_event("reflex_gesture_end", emotion=emotion, gesture=gesture, outcome=outcome)
    print(f"<<< {gesture}: {outcome}")


def _log_decision(emotion: str, score: float, decision: Decision, live: bool) -> None:
    log_event(
        "reflex_decision",
        mode="live" if live else "dry_run",
        emotion=emotion,
        score=round(float(score), 3),
        fire=decision.fire,
        gesture=decision.gesture,
        reason=decision.reason,
    )


def main() -> int:
    args = parse_args()
    live = args.live and not args.dry_run
    if args.live and args.dry_run:
        print("--live and --dry-run are mutually exclusive; refusing to guess.", file=sys.stderr)
        return 2

    print("=== emotion → gesture reflex (spike) ===")
    print(f"mode: {'LIVE — the arm will move' if live else 'DRY RUN — no motion'}")

    emotion_map = validate_map()
    if not emotion_map:
        print("no usable emotion→gesture mappings; nothing to do.", file=sys.stderr)
        return 1

    if not args.sim_emotion and not spikeconfig.EMOTION_STATE_PATH.is_file():
        print(
            f"\nnote: no emotion state at {spikeconfig.EMOTION_STATE_PATH} yet.\n"
            "      start the detector first, or pass --sim-emotion to test the arm path.\n"
        )

    if live and not gestures.dataset_available():
        print(
            "the recorded gesture dataset is not present, so no gesture can be replayed. "
            "See docs/recording_gestures.md.",
            file=sys.stderr,
        )
        return 1

    if not live:
        # Dry run uses a simulated arm and never prompts — it commands nothing.
        arm = motion.DryRunArm()
        try:
            return run(args, arm, live=False, emotion_map=emotion_map)
        except KeyboardInterrupt:
            print("\nstopped.")
            return 0

    # --- live: operator gate + kill switch (CLAUDE.md rules 1 and 7) ---
    if not safety.require_operator("run the emotion reflex — the arm will move"):
        print("operator not confirmed — refusing to arm.")
        return 1

    safety.clear_stop()
    safety.install_kill_switch()
    print("kill switch armed: Ctrl-C freezes the arm and offers the operator menu.")

    try:
        arm = motion.connect()
    except Exception as exc:
        print(f"could not connect to the arm: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        return run(args, arm, live=True, emotion_map=emotion_map)
    except KeyboardInterrupt:
        print("\nstopped by Ctrl-C.")
        return 0
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
