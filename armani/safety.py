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
from collections.abc import Iterable, Iterator
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
    """The arm reads outside its PHYSICAL range by more than the tolerance.

    This is an encoder or calibration fault, not a parking position: the joint
    claims to be somewhere the hardware cannot go. Refusing to move is the only
    safe response, because every commanded target would be computed from a
    position we cannot trust.

    Note what this is NOT: an arm parked outside the conservative *policy*
    envelope is a perfectly legal starting point. Its rest pose is measured
    reality, and motion out of it must stay possible — otherwise home() and the
    kill switch stop working exactly when they are needed.
    """


class Homeable(Protocol):
    """The slice of the robot API the safety layer needs."""

    def read_positions(self) -> Action: ...
    def send(self, action: Action) -> Action: ...
    def disable_torque(self) -> None: ...


# --- Clamping ------------------------------------------------------------


def clamp_action(
    action: Action,
    profile: str = config.DEFAULT_PROFILE,
    log_clamps: bool = True,
) -> Action:
    """Return a NEW action with every joint clamped to the given limit profile.

    ``profile`` selects which envelope applies — see config.LIMIT_PROFILES:
    "policy" for LLM/IK targets, "recorded" for replayed or measured targets,
    "backstop" for the send boundary.

    ``log_clamps=False`` is for per-frame callers running at 30 Hz. A recording
    that rests against a stop clamps on EVERY frame, and logging each one buries
    the console and writes hundreds of identical entries into the decision log,
    which is meant to be a readable audit trail. Those callers report a single
    summary instead — see gestures.frame_clamp_deviation.

    Raises on unknown joints and on non-finite values. NaN is rejected
    explicitly because ``min(hi, max(lo, nan))`` evaluates to ``nan`` — every
    comparison against NaN is False, so a naive clamp would hand NaN straight
    to the motors.
    """
    try:
        limits = config.LIMIT_PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"unknown limit profile {profile!r}; expected one of {sorted(config.LIMIT_PROFILES)}"
        ) from None

    clamped: Action = {}
    for joint, value in action.items():
        if joint not in limits:
            raise ValueError(f"unknown joint {joint!r}; expected one of {sorted(limits)}")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"joint {joint!r} got non-numeric value {value!r}") from None
        if not math.isfinite(numeric):
            raise ValueError(f"joint {joint!r} got non-finite value {numeric!r}")
        low, high = limits[joint]
        result = min(high, max(low, numeric))
        if result != numeric and log_clamps:
            # At the "physical" profile this must never fire: the interpolator
            # lerps between a measured start and an already-clamped target, so
            # every step is inside the physical range by construction.
            level = log.error if profile == "physical" else log.warning
            level("clamped %s (%s): %.2f -> %.2f", joint, profile, numeric, result)
            log_event("clamp", joint=joint, profile=profile, requested=numeric, applied=result)
        clamped[joint] = result
    return clamped


def check_start_pose(current: Action, joints: Iterable[str]) -> None:
    """Refuse to move when a measured joint reads beyond its PHYSICAL range.

    Checked against the physical limits, not the policy envelope: a parked arm
    outside the policy envelope is legal and must still be movable.
    """
    faults = []
    for joint in joints:
        low, high = config.PHYSICAL_LIMITS[joint]
        value = current[joint]
        if not (low - config.PHYSICAL_TOLERANCE <= value <= high + config.PHYSICAL_TOLERANCE):
            faults.append(f"{joint}={value:.1f} (physical {low:g}..{high:g})")
    if faults:
        raise OutsideEnvelopeError(
            "joint(s) read beyond the physical range by more than "
            f"{config.PHYSICAL_TOLERANCE:g} degrees: " + "; ".join(faults) + ". "
            "This is an encoder or calibration fault, not a parking position — "
            "the arm claims to be somewhere the hardware cannot reach. Refusing to move. "
            "Power-cycle the servos and re-run smoke_01; if it persists the calibration is wrong."
        )


# --- Interpolation -------------------------------------------------------


def interp_move(
    current: Action,
    target: Action,
    duration: float,
    hz: int = config.CONTROL_HZ,
    profile: str = config.DEFAULT_PROFILE,
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

    safe_target = clamp_action(target, profile=profile)
    missing = [j for j in safe_target if j not in current]
    if missing:
        raise ValueError(f"no current position for joint(s) {missing}; cannot interpolate")

    # Interpolate from the MEASURED position, never from a clamped copy of it.
    # Clamping the origin would invent a start point the arm is not at, and the
    # first commanded step would then be a jump of however far the real arm sits
    # outside the limit — exactly the raw jump safety rule 2 forbids.
    start = {j: float(current[j]) for j in safe_target}

    # Only a physically impossible reading blocks motion. A pose outside the
    # policy envelope is legal to start from, and the lerp below walks it back
    # in monotonically: every step lies between the measured start and an
    # already-clamped target, so the arm only ever moves toward legality.
    check_start_pose(start, safe_target)

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
    The motion loop notices the flag, stops commanding new targets and FREEZES,
    holding position while the operator chooses what happens next (safety rule
    7 — nothing auto-drives anywhere). A second Ctrl-C raises KeyboardInterrupt
    so the operator is never trapped, and leaves the arm exactly where it is.

    ``on_stop`` is accepted for symmetry with later stages and unused for now.
    """
    del on_stop

    def _handle_sigint(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        if _stop_requested.is_set():
            raise KeyboardInterrupt("second Ctrl-C — aborting immediately")
        request_stop("Ctrl-C")
        print("\n[kill switch] freezing — the arm will hold position and ask what to do.")
        print("              Ctrl-C again to abort hard and leave it exactly where it is.")

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except ValueError:
        # Not on the main thread; the caller keeps the default handler.
        log.warning("could not install SIGINT handler off the main thread")

    _install_esc_listener()


def handle_freeze(arm: Homeable, motion_start: Action) -> None:
    """Safety rule 7: the kill switch FREEZES, it never auto-drives anywhere.

    Commanding has already stopped by the time we get here, so the arm is
    holding position. The operator is present (rule 1) and chooses what happens
    next. A non-interactive session holds and does nothing at all.
    """
    from armani import motion

    log_event("freeze", at={k: round(v, 2) for k, v in motion_start.items()})
    print("\n[kill switch] FROZEN — holding position. Nothing will move until you choose.")

    if not sys.stdin.isatty():
        log.warning("not a terminal; holding position and taking no further action")
        log_event("freeze_choice", choice="hold", reason="non-interactive")
        return

    home_offer = (
        "  [h] move to home\n" if config.HOME_VERIFIED else "  [h] home — UNAVAILABLE (not yet verified)\n"
    )
    print(
        "\n  [s] return slowly to where this motion started\n"
        + home_offer
        + "  [t] torque OFF — SUPPORT THE ARM FIRST, it will drop\n"
        "  [l] leave it exactly as-is\n"
    )

    while True:
        try:
            choice = input("Choice [s/h/t/l]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            log_event("freeze_choice", choice="leave", reason="input aborted")
            return

        if choice == "s":
            log_event("freeze_choice", choice="return_to_start")
            # "recorded" profile: the start pose is measured reality and the
            # policy envelope must not clip it (safety rule 2).
            motion.goto(
                arm,  # type: ignore[arg-type]
                dict(motion_start),
                duration=config.RECOVERY_DURATION_S,
                ignore_stop=True,
                profile="recorded",
            )
            return
        if choice == "h":
            if not config.HOME_VERIFIED:
                print("  Home is not verified yet — run scripts/capture_home.py first. (rule 4)")
                continue
            log_event("freeze_choice", choice="home")
            motion.home(arm, slow=True, ignore_stop=True)  # type: ignore[arg-type]
            return
        if choice == "t":
            confirm = input("  Are you physically supporting the arm? It WILL drop. [y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                print("  Not confirmed — torque left on.")
                continue
            log_event("freeze_choice", choice="torque_off")
            try:
                arm.disable_torque()
                print("  Torque off. The arm is limp.")
            except Exception as exc:
                log.error("could not disable torque: %s", exc)
            return
        if choice == "l":
            log_event("freeze_choice", choice="leave")
            return
        print("  Not one of s/h/t/l.")


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
    """Return the arm to where this block STARTED if anything goes wrong.

    Safety rule 4: recovery retraces the corridor the arm already traversed
    safely, rather than driving to HOME_POSE. The entry pose is always
    known-reachable — the arm was just there a moment ago — and if nothing has
    moved yet this is a zero-length move.

    Named as a class for readability at call sites (``with SafeMotion(arm):``).
    The original exception is always re-raised, never swallowed.
    """
    from armani import motion

    entry: Action | None = None
    if robot is not None:
        try:
            entry = robot.read_positions()
        except Exception as exc:
            # Without an entry pose there is nothing safe to return to, so the
            # block still runs but recovery degrades to "leave it where it is".
            log.error("could not read the entry pose; recovery will not move the arm: %s", exc)

    log_event("motion_begin", description=description, entry=_rounded(entry))
    try:
        yield
    except BaseException as exc:
        log.error("%s failed: %s: %s", description, type(exc).__name__, exc)
        log_event("motion_error", description=description, error=f"{type(exc).__name__}: {exc}")

        # Second Ctrl-C is the operator's hard abort (rule 7). Starting a
        # recovery move here would make that promise a lie.
        if isinstance(exc, KeyboardInterrupt) and stop_requested():
            print("\n[safety] hard abort — arm left exactly where it is. Check it physically.")
            log_event("hard_abort", description=description)
            raise

        if robot is not None and entry is not None:
            try:
                moved = _joints_that_moved(robot, entry)
            except Exception as read_exc:
                log.error("could not read the current pose; leaving the arm as-is: %s", read_exc)
                moved = {}

            if not moved:
                print(f"\n[safety] {type(exc).__name__} during {description} — nothing moved, arm left as-is.")
                log_event("recovery_skipped", description=description, reason="no joint moved")
            else:
                try:
                    print(
                        f"\n[safety] {type(exc).__name__} during {description} — "
                        f"returning {', '.join(sorted(moved))} to where it started."
                    )
                    motion.goto(
                        robot,  # type: ignore[arg-type]
                        moved,
                        duration=config.RECOVERY_DURATION_S,
                        ignore_stop=True,
                        profile="recorded",
                    )
                except Exception as recovery_exc:
                    log.critical(
                        "RECOVERY FAILED after error: %s. Check the arm physically.", recovery_exc
                    )
                    log_event("recovery_failed", error=str(recovery_exc))
        raise
    else:
        log_event("motion_end", description=description)


def _rounded(pose: Action | None) -> dict[str, float] | None:
    return None if pose is None else {k: round(v, 2) for k, v in pose.items()}


def _joints_that_moved(robot: Homeable, entry: Action) -> Action:
    """Entry values for the joints that actually left their starting position.

    Recovery must not drag joints that never moved. That matters most for a
    joint parked at its mechanical stop: including it would command a small
    move on every single error recovery, when the correct answer is to leave it
    alone. If nothing moved, this is empty and recovery is a no-op.
    """
    now = robot.read_positions()
    return {
        joint: value
        for joint, value in entry.items()
        if joint in now and abs(now[joint] - value) > config.MOVED_EPSILON
    }
