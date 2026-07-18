#!/usr/bin/env python
"""Run ARM-ANI's voice session: talk to it, it talks back and moves.

    python scripts/run_agent.py                # voice, push-to-talk (OPERATOR)
    python scripts/run_agent.py --text         # keyboard only, no audio
    python scripts/run_agent.py --no-motion    # personality only, arm disabled

Push-to-talk: HOLD the spacebar to speak, release to send. The model replies in
voice while the arm moves. Type instead with --text. ESC / Ctrl-C freezes the
arm and hands you the operator menu (safety rule 7).

This script owns all the I/O the agent module deliberately leaves out: the
microphone, the speaker, the WebSocket event loop, and the kill-switch handoff.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import _bootstrap  # noqa: E402,F401  (verifies the interpreter, exits if wrong)

from armani import agent, config, motion, safety  # noqa: E402
from armani.logutil import get_logger, log_event  # noqa: E402

log = get_logger("run_agent")

CHECKLIST = """\
Scripted demo checklist (say these out loud, hold space while you talk):
  1. "Introduce yourself in one sentence."
  2. "What gestures can you do?"
  3. "Take a bow."          <- it should TALK WHILE THE ARM MOVES
  4. "Improvise a tiny robot dance."
  5. "Pick up the banana."  <- it must honestly decline: no eyes yet
  6. Interrupt it mid-sentence by tapping space (barge-in).
  7. "Stop" while a gesture is running.
  8. Ctrl-C to freeze, then [s] to recover.
"""


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARM-ANI realtime voice session.")
    parser.add_argument("--text", action="store_true", help="keyboard input only, no microphone")
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="run the personality with motion tools disabled (no arm needed)",
    )
    return parser.parse_args()


def _warn_about_default_mic() -> None:
    """Nudge the operator to the wired headset (CLAUDE.md demo-mic note)."""
    try:
        import sounddevice as sd

        name = sd.query_devices(kind="input")["name"]
    except Exception as exc:
        log.warning("could not query the input device: %s", exc)
        return
    if any(bad.lower() in name.lower() for bad in config.AUDIO_DEVICE_WARN_SUBSTRINGS):
        print(
            f"\n  NOTE: your default mic is {name!r}. The demo mic is a WIRED HEADSET.\n"
            "  Set ARMANI_AUDIO_INPUT_DEVICE in .env or pick it in System Settings > Sound.\n"
        )


def main() -> int:
    args = parse()
    print("=== ARM-ANI voice session ===")

    motion_enabled = not args.no_motion
    if motion_enabled:
        # ONE operator gate for the whole session (rule 1). Declined -> personality
        # still runs, motion tools return refused.
        if not safety.require_operator("start the voice session with motion enabled"):
            print("Motion not enabled — running personality only. Tools that move will refuse.")
            motion_enabled = False

    try:
        arm = motion.connect(dry_run=not motion_enabled)
    except Exception as exc:
        print(f"could not connect to the arm: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    worker = agent.MotionWorker(arm, motion_enabled=motion_enabled)
    worker.start()

    if motion_enabled:
        safety.clear_stop()
        safety.install_kill_switch()
        print("Kill switch armed: ESC / Ctrl-C freezes the arm and offers the operator menu.")

    if not args.text:
        _warn_about_default_mic()

    print(CHECKLIST)

    try:
        asyncio.run(_run(worker, text_mode=args.text, motion_enabled=motion_enabled))
    except KeyboardInterrupt:
        print("\nSession ended by Ctrl-C.")
    finally:
        worker.shutdown()
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})", file=sys.stderr)
    return 0


async def _run(worker: agent.MotionWorker, *, text_mode: bool, motion_enabled: bool) -> None:
    session_cm = agent.build_session(worker, text_only=text_mode)
    async with await session_cm as session:
        # A short greeting so the operator sees/hears it is live before starting.
        await session.send_message(
            "Greet the operator in one short sentence. If this is a voice session, mention "
            "they hold the spacebar to talk."
        )
        await agent.request_response(session)

        background = [
            asyncio.create_task(_pump_events(session, worker, text_mode=text_mode)),
            asyncio.create_task(_pump_completions(session, worker)),
        ]
        input_coro = _text_input(session) if text_mode else _voice_input(session, worker)
        input_task = asyncio.create_task(input_coro)

        try:
            # The session lives until the operator quits (text) or Ctrl-C (voice).
            await input_task
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            for task in (input_task, *background):
                task.cancel()
            # Await the cancellations so the SDK's session cleanup actually runs,
            # instead of "Task was destroyed but it is pending" on the way out.
            await asyncio.gather(input_task, *background, return_exceptions=True)



# --- Event handling ------------------------------------------------------


async def _pump_events(session, worker: agent.MotionWorker, *, text_mode: bool) -> None:
    """Print transcripts, play audio, and hand the freeze menu to the operator."""
    # No speaker in text mode — there is no audio, and opening an output stream
    # that never plays just risks a device error on machines without one.
    player = None if text_mode else _Speaker()
    printed: set[str] = set()  # item ids already shown, so we print each reply once
    try:
        async for event in session:
            etype = event.type
            if etype == "audio" and player is not None:
                player.play(event.audio.data)
            elif etype == "audio_interrupted" and player is not None:
                player.flush()  # barge-in: drop what we were about to say
            elif etype == "history_added":
                _print_item(event.item, printed)
            elif etype == "history_updated":
                # Text-mode replies arrive here (an assistant item's text is filled
                # in by successive updates), never as a standalone history_added.
                for item in event.history:
                    _print_item(item, printed)
            elif etype == "tool_start":
                log_event("agent_tool_start", tool=getattr(event.tool, "name", "?"))
            elif etype == "error":
                log.error("session error: %s", event.error)
            # A human kill switch fires on another thread; check it every event.
            if safety.stop_requested() and not safety.freeze_suppressed():
                await _run_freeze_menu(session, worker)
    except asyncio.CancelledError:
        pass
    finally:
        if player is not None:
            player.close()



async def _pump_completions(session, worker: agent.MotionWorker) -> None:
    """Feed finished-motion results back to the model so it can react."""
    try:
        while True:
            try:
                done = worker.completions.get_nowait()
            except Exception:
                await asyncio.sleep(0.1)
                continue
            summary = f"[motion:{done.action}] {done.status}"
            if done.detail:
                summary += f" — {done.detail}"
            # System-role context so the model comments naturally, not as the user.
            await session.send_message(summary)
    except asyncio.CancelledError:
        pass


async def _run_freeze_menu(session, worker: agent.MotionWorker) -> None:
    """Rule 7 handoff: pause the agent's I/O, let the human own the terminal."""
    worker.pause()
    start = worker.job_start_pose or worker.pose_snapshot() or {}
    loop = asyncio.get_running_loop()
    try:
        # handle_freeze blocks on input(); run it off the event loop.
        await loop.run_in_executor(None, safety.handle_freeze, worker.arm, dict(start))
    finally:
        safety.clear_stop()
        worker.resume()


# --- Voice (push-to-talk) ------------------------------------------------


class _Speaker:
    """Blocking-free PCM16 playback via a sounddevice output stream."""

    def __init__(self) -> None:
        import numpy as np
        import sounddevice as sd

        self._np = np
        self._stream = sd.RawOutputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="int16",
            device=config.AUDIO_OUTPUT_DEVICE,
        )
        self._stream.start()

    def play(self, pcm: bytes) -> None:
        try:
            self._stream.write(pcm)
        except Exception as exc:
            log.warning("audio playback error: %s", exc)

    def flush(self) -> None:
        try:
            self._stream.stop()
            self._stream.start()
        except Exception as exc:
            log.warning("audio flush error: %s", exc)

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


async def _voice_input(session, worker: agent.MotionWorker) -> None:
    """Stream mic audio while the PTT key is held; commit + respond on release."""
    import numpy as np  # noqa: F401  (sounddevice returns numpy-friendly buffers)
    import sounddevice as sd
    from pynput import keyboard

    loop = asyncio.get_running_loop()
    talking = threading.Event()
    key = _ptt_key(keyboard)

    def on_press(k: object) -> None:
        if k == key and not talking.is_set():
            talking.set()
            # Barge-in: pressing space while it speaks interrupts the model.
            asyncio.run_coroutine_threadsafe(session.interrupt(), loop)

    def on_release(k: object) -> None:
        if k == key and talking.is_set():
            talking.clear()
            asyncio.run_coroutine_threadsafe(_commit_and_respond(session), loop)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    if not listener.running:
        print(
            "\n  Could not start the spacebar listener. Grant Input Monitoring to your terminal\n"
            "  (System Settings > Privacy & Security > Input Monitoring), or use --text.\n"
        )

    print("Hold SPACE to talk. Release to send. Ctrl-C to end.\n")

    def audio_callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            log.debug("mic status: %s", status)
        if talking.is_set():
            asyncio.run_coroutine_threadsafe(session.send_audio(bytes(indata)), loop)

    try:
        with sd.RawInputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="int16",
            blocksize=config.AUDIO_BLOCKSIZE,
            device=config.AUDIO_INPUT_DEVICE,
            callback=audio_callback,
        ):
            while True:
                await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        pass
    finally:
        listener.stop()


async def _commit_and_respond(session) -> None:
    await session.send_audio(b"", commit=True)
    await agent.request_response(session)


def _ptt_key(keyboard) -> object:  # noqa: ANN001
    name = config.PTT_KEY.strip().lower()
    special = getattr(keyboard.Key, name, None)
    if special is not None:
        return special
    return keyboard.KeyCode.from_char(name)


# --- Text fallback -------------------------------------------------------


async def _text_input(session) -> None:
    """Type to ARM-ANI when audio is impossible. Blank line or 'quit' exits."""
    loop = asyncio.get_running_loop()
    print("Text mode. Type a message and press ENTER. Empty line or 'quit' to exit.\n")
    while True:
        try:
            line = await loop.run_in_executor(None, input, "you> ")
        except (EOFError, asyncio.CancelledError):
            return
        line = line.strip()
        if not line or line.lower() in ("quit", "exit"):
            return
        await session.send_message(line)
        await agent.request_response(session)


# --- Printing ------------------------------------------------------------


def _print_item(item: object, printed: set[str]) -> None:
    """Print an assistant/user line once.

    ``history_updated`` re-sends the whole history on every change, and the item
    objects do not carry a stable id across those events, so we dedupe on the
    text itself: the same sentence prints exactly once no matter how many times
    it is re-delivered.
    """
    role = getattr(item, "role", None)
    if role not in ("assistant", "user"):
        return
    text = agent._item_text(item) if role == "assistant" else _user_text(item)
    if not text or text in printed:
        return
    printed.add(text)
    who = "ARM-ANI" if role == "assistant" else "you"
    print(f"{who}> {text}")




def _user_text(item: object) -> str:
    content = getattr(item, "content", None) or []
    parts = [getattr(c, "text", None) or getattr(c, "transcript", None) or "" for c in content]
    return " ".join(p for p in parts if p).strip()


if __name__ == "__main__":
    raise SystemExit(main())
