"""Safety primitives. Nothing in this project commands a motor without going
through this module.

Actions are plain ``{joint_name: value}`` dicts in this layer. The translation
to lerobot's ``{"<joint>.pos": value}`` feature keys happens once, at the
hardware boundary in ``motion.py``.
"""

from __future__ import annotations

import math
import signal
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Protocol

from armani import config
from armani.logutil import get_logger, log_event

log = get_logger("safety")

Action = dict[str, float]

# Set by the kill switch. Long motions check it between interpolation steps.
_stop_requested = threading.Event()


class OutsideEnvelopeError(RuntimeError):
    """The arm is physically resting outside its policy limits.

    Raised instead of moving, because every legal target is then far away and
    any command would be a large jump. A human has to walk the arm back into
    the envelope (or the limit has to be widened deliberately) first.
    """


class Homeable(Protocol):
    """The slice of the robot API the safety layer needs."""

    def read_positions(self) -> Action: ...
    def send(self, action: Action) -> None: ...


# --- Clamping ------------------------------------------------------------


def clamp_action(action: Action) -> Action:
    """Return a NEW action with every joint clamped to its configured limit.

    Raises on unknown joints and on non-finite values. NaN is rejected
    explicitly because ``min(hi, max(lo, nan))`` evaluates to ``nan`` — every
    comparison against NaN is False, so a naive clamp would hand NaN straight
    to the motors.
    """
    clamped: Action = {}
    for joint, value in action.items():
        if joint not in config.JOINT_LIMITS:
            raise ValueError(f"unknown joint {joint!r}; expected one of {sorted(config.JOINT_LIMITS)}")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"joint {joint!r} got non-numeric value {value!r}") from None
        if not math.isfinite(numeric):
            raise ValueError(f"joint {joint!r} got non-finite value {numeric!r}")
        low, high = config.JOINT_LIMITS[joint]
        result = min(high, max(low, numeric))
        if result != numeric:
            log.warning("clamped %s: %.2f -> %.2f", joint, numeric, result)
            log_event("clamp", joint=joint, requested=numeric, applied=result)
        clamped[joint] = result
    return clamped


# --- Interpolation -------------------------------------------------------


def interp_move(
    current: Action,
    target: Action,
    duration: float,
    hz: int = config.CONTROL_HZ,
) -> Iterator[Action]:
    """Yield clamped intermediate actions from ``current`` to ``target``.

    The caller is responsible for sleeping ``1/hz`` between yielded actions;
    keeping the timing out of here makes the trajectory testable without
    hardware. ``duration`` is therefore a lower bound: if honouring
    ``MAX_JOINT_SPEED`` needs more steps than ``duration`` allows, more steps
    are yielded and the move simply takes longer. It never takes less.

    Only joints present in ``target`` are moved.
    """
    if hz <= 0:
        raise ValueError(f"hz must be positive, got {hz}")
    if duration < 0:
        raise ValueError(f"duration must be non-negative, got {duration}")

    safe_target = clamp_action(target)
    missing = [j for j in safe_target if j not in current]
    if missing:
        raise ValueError(f"no current position for joint(s) {missing}; cannot interpolate")

    # Interpolate from the MEASURED position, never from a clamped copy of it.
    # Clamping the origin would invent a start point the arm is not at, and the
    # first commanded step would then be a jump of however far the real arm sits
    # outside the limit — exactly the raw jump safety rule 2 forbids.
    start = {j: float(current[j]) for j in safe_target}
    outside = [
        f"{j}={start[j]:.1f} (limit {config.JOINT_LIMITS[j][0]:g}..{config.JOINT_LIMITS[j][1]:g})"
        for j in safe_target
        if not (
            config.JOINT_LIMITS[j][0] - config.ENVELOPE_TOLERANCE
            <= start[j]
            <= config.JOINT_LIMITS[j][1] + config.ENVELOPE_TOLERANCE
        )
    ]
    if outside:
        raise OutsideEnvelopeError(
            "arm is resting outside its policy limits: "
            + "; ".join(outside)
            + ". Refusing to move — every legal target is a long way from here. "
            "Power down the servos and move the arm back by hand, or widen the limit "
            "in config.JOINT_LIMITS deliberately."
        )

    largest_delta = max((abs(safe_target[j] - start[j]) for j in safe_target), default=0.0)
    if largest_delta == 0.0:
        yield dict(safe_target)
        return

    steps_for_duration = math.ceil(duration * hz)
    # Speed limit: no single step may exceed MAX_JOINT_SPEED / hz.
    steps_for_speed = math.ceil(largest_delta / (config.MAX_JOINT_SPEED / hz))
    steps = max(1, steps_for_duration, steps_for_speed)

    if steps > max(1, steps_for_duration):
        log.info(
            "speed limit stretched move: %d -> %d steps (%.1f deg over %.1fs)",
            max(1, steps_for_duration),
            steps,
            largest_delta,
            duration,
        )

    for step in range(1, steps + 1):
        fraction = step / steps
        yield {j: start[j] + (safe_target[j] - start[j]) * fraction for j in safe_target}


# --- Kill switch ---------------------------------------------------------


def stop_requested() -> bool:
    return _stop_requested.is_set()


def request_stop(reason: str) -> None:
    if not _stop_requested.is_set():
        log.warning("STOP requested: %s", reason)
        log_event("stop_requested", reason=reason)
    _stop_requested.set()


def clear_stop() -> None:
    _stop_requested.clear()


def install_kill_switch(on_stop: object = None) -> None:
    """Register Ctrl-C (and ESC when permitted) as a stop request.

    Ctrl-C only sets the stop flag; it does not kill the process mid-motion.
    The motion loop notices the flag, stops commanding new targets and returns
    the arm home under control. A second Ctrl-C raises KeyboardInterrupt so the
    operator is never trapped.

    ``on_stop`` is accepted for symmetry with later stages and unused for now.
    """
    del on_stop

    def _handle_sigint(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        if _stop_requested.is_set():
            raise KeyboardInterrupt("second Ctrl-C — aborting immediately")
        request_stop("Ctrl-C")
        print("\n[kill switch] stopping and returning home. Ctrl-C again to abort hard.")

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except ValueError:
        # Not on the main thread; the caller keeps the default handler.
        log.warning("could not install SIGINT handler off the main thread")

    _install_esc_listener()


def _install_esc_listener() -> None:
    """Best-effort global ESC listener. Requires macOS Input Monitoring."""
    try:
        from pynput import keyboard
    except Exception as exc:  # pragma: no cover - import failure is environmental
        log.warning("ESC kill switch unavailable (pynput import failed: %s)", exc)
        return

    def _on_press(key: object) -> None:
        if key == keyboard.Key.esc:
            request_stop("ESC")

    try:
        listener = keyboard.Listener(on_press=_on_press)
        listener.daemon = True
        listener.start()
    except Exception as exc:
        log.warning(
            "ESC kill switch unavailable (%s). Grant Input Monitoring to your terminal: "
            "System Settings > Privacy & Security > Input Monitoring. Ctrl-C still works.",
            exc,
        )


# --- Operator presence ---------------------------------------------------


def require_operator(action_description: str = "move the arm") -> bool:
    """Ask the operator to confirm they are present and watching.

    Fails closed: anything other than an explicit yes on an interactive
    terminal returns False. A non-interactive stdin (CI, piped input) can never
    approve motion.
    """
    if config.DRY_RUN:
        print(f"[dry-run] would ask the operator to confirm: {action_description}")
        return True

    if not sys.stdin.isatty():
        log.error("stdin is not a terminal; refusing to move without an operator")
        log_event("operator_check", approved=False, reason="stdin not a tty")
        return False

    print(f"\nAbout to {action_description}. Keep a hand near the arm.")
    try:
        answer = input("Operator present and watching? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        log_event("operator_check", approved=False, reason="input aborted")
        return False

    approved = answer in ("y", "yes")
    log_event("operator_check", approved=approved, action=action_description)
    if not approved:
        print("Not confirmed — no motion will happen.")
    return approved


# --- Safe motion context -------------------------------------------------


@contextmanager
def SafeMotion(robot: Homeable | None, description: str = "motion"):  # noqa: N802
    """Guarantee the arm ends up home if anything goes wrong inside the block.

    Named as a class for readability at call sites (``with SafeMotion(robot):``).
    Exceptions are logged, the arm is walked home slowly, and the original
    exception is re-raised — never swallowed.
    """
    from armani import motion

    log_event("motion_begin", description=description)
    try:
        yield
    except BaseException as exc:
        log.error("%s failed: %s: %s", description, type(exc).__name__, exc)
        log_event("motion_error", description=description, error=f"{type(exc).__name__}: {exc}")

        # The second Ctrl-C is the operator's hard abort. Homing here would
        # start a fresh multi-second move and make that promise a lie, so the
        # arm is left exactly where it is and the exception propagates.
        if isinstance(exc, KeyboardInterrupt) and stop_requested():
            print("\n[safety] hard abort — arm left where it is, NOT homing. Check it physically.")
            log_event("hard_abort", description=description)
            raise

        if robot is not None:
            try:
                print(f"\n[safety] {type(exc).__name__} during {description} — returning home slowly.")
                motion.home(robot, slow=True, ignore_stop=True)
            except Exception as home_exc:
                log.critical("HOMING FAILED after error: %s. Check the arm physically.", home_exc)
                log_event("home_failed", error=str(home_exc))
        raise
    else:
        log_event("motion_end", description=description)
