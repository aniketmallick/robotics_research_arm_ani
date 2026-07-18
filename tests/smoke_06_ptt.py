#!/usr/bin/env python
"""Smoke 06 — global push-to-talk spacebar via pynput.

Stage 3 drives the realtime voice loop from a GLOBAL spacebar listener, so this
must work when the terminal is not focused. On macOS that needs Input
Monitoring (and usually Accessibility) granted to the terminal app.

The failure mode is quiet: without permission pynput starts a listener that
simply never reports a key. This test therefore treats "no events within the
timeout" as a permission failure rather than waiting forever.
"""

from __future__ import annotations

import sys
import threading
import time

from _bootstrap import banner, fail, ok, parse_args, permission_hint, skip

from armani.logutil import log_event

WAIT_SECONDS = 20.0
REQUIRED_HOLDS = 2


def main() -> int:
    args = parse_args(__doc__ or "")
    banner("Smoke 06: push-to-talk spacebar")

    if args.dry_run:
        print("[dry-run] would start a global pynput listener and wait for "
              f"{REQUIRED_HOLDS} spacebar hold/release cycles")
        return ok("dry run complete")

    try:
        from pynput import keyboard
    except Exception as exc:
        return fail(f"pynput unavailable: {type(exc).__name__}: {exc}")

    if not sys.stdin.isatty():
        return skip("not an interactive terminal; push-to-talk needs a human to press space")

    holds: list[float] = []
    pressed_at: dict[str, float] = {}
    done = threading.Event()
    # Any key event at all proves the listener is receiving input. Without this
    # we would blame macOS permissions for a run where the operator simply
    # pressed ESC, and send them to a settings pane that is already correct.
    saw_any_event = threading.Event()

    def on_press(key: object) -> None:
        saw_any_event.set()
        if key == keyboard.Key.space and "t" not in pressed_at:
            pressed_at["t"] = time.perf_counter()
            print("  space DOWN  (talking...)")

    def on_release(key: object) -> bool | None:
        saw_any_event.set()
        if key == keyboard.Key.space and "t" in pressed_at:
            duration = time.perf_counter() - pressed_at.pop("t")
            holds.append(duration)
            print(f"  space UP    (held {duration:.2f}s)  [{len(holds)}/{REQUIRED_HOLDS}]")
            if len(holds) >= REQUIRED_HOLDS:
                done.set()
                return False
        if key == keyboard.Key.esc:
            done.set()
            return False
        return None

    print(
        f"\nHold SPACE and release, {REQUIRED_HOLDS} times. ESC to give up.\n"
        f"Waiting up to {WAIT_SECONDS:.0f}s...\n"
    )

    try:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
    except Exception as exc:
        permission_hint("Input Monitoring", f"pynput could not start a listener: {exc}")
        return fail(f"listener failed to start: {type(exc).__name__}: {exc}")

    try:
        done.wait(timeout=WAIT_SECONDS)
    finally:
        listener.stop()

    if not holds and not saw_any_event.is_set():
        permission_hint(
            "Input Monitoring",
            "no key events arrived at all — pynput started but never saw a key",
        )
        print(
            "  Also check System Settings > Privacy & Security > Accessibility.\n"
            "  Grant BOTH to your terminal app and restart it.\n",
            file=sys.stderr,
        )
        return fail("no key events received — this is a permissions problem")

    if not holds:
        # Keys ARE arriving, so permissions are fine; the operator just never
        # pressed space (or pressed ESC). That is a skip, not a failure.
        return skip("key events are arriving, but no spacebar hold was captured")

    log_event("smoke_06", holds=[round(h, 3) for h in holds])

    if len(holds) < REQUIRED_HOLDS:
        return skip(f"only {len(holds)}/{REQUIRED_HOLDS} holds captured before timeout or ESC")
    return ok(f"captured {len(holds)} spacebar holds ({', '.join(f'{h:.2f}s' for h in holds)})")


if __name__ == "__main__":
    raise SystemExit(main())
