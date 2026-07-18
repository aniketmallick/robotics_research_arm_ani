"""Motor connection and interpolated motion for the SO-101 follower.

Two things about lerobot 0.5.2 drive the shape of this module:

1. There is no distinct ``SO101Follower`` class any more — SO-100/101 were
   merged into ``SOFollower`` (``name = "so_follower"``, which is why the
   existing calibration lives in ``calibration/robots/so_follower/``).
   ``SO101Follower`` still exists as an alias, so we import that name.
2. ``SOFollower.connect(calibrate=True)`` will start lerobot's INTERACTIVE
   recalibration if the motors disagree with the calibration file. We must
   never recreate this robot's calibration, so we always pass
   ``calibrate=False`` and handle the mismatch ourselves.
"""

from __future__ import annotations

import glob
import time
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from armani import config, safety
from armani.logutil import get_logger, log_event

log = get_logger("motion")

Action = dict[str, float]

POS_SUFFIX = ".pos"


def to_feature_keys(action: Action) -> dict[str, float]:
    """``{"gripper": 50}`` -> ``{"gripper.pos": 50}`` (lerobot's action keys)."""
    return {f"{joint}{POS_SUFFIX}": value for joint, value in action.items()}


def from_feature_keys(observation: dict[str, object]) -> Action:
    """Inverse of :func:`to_feature_keys`, ignoring camera and other entries."""
    return {
        key.removesuffix(POS_SUFFIX): float(value)  # type: ignore[arg-type]
        for key, value in observation.items()
        if key.endswith(POS_SUFFIX)
    }


class Arm(Protocol):
    """The only robot surface the rest of ARM-ANI is allowed to touch."""

    def read_positions(self) -> Action: ...
    def send(self, action: Action) -> Action: ...
    def disconnect(self) -> None: ...
    def disable_torque(self) -> None: ...
    def torque_disabled(self) -> AbstractContextManager[None]: ...
    @property
    def label(self) -> str: ...


class RealArm:
    """Thin wrapper over lerobot's follower, speaking plain joint-name dicts."""

    def __init__(self, robot: object, port: str, robot_id: str) -> None:
        self._robot = robot
        self._port = port
        self._id = robot_id

    @property
    def label(self) -> str:
        return f"SO-101 follower id={self._id} on {self._port}"

    def read_positions(self) -> Action:
        return from_feature_keys(self._robot.get_observation())  # type: ignore[attr-defined]

    def send(self, action: Action) -> Action:
        # Hard backstop at the last line before the motors, so safety rule 2
        # holds structurally: no code path can send a value outside what the
        # hardware can plausibly reach. This clamps to BACKSTOP, not policy —
        # policy belongs on targets, and applying it here would forbid the arm
        # from ever leaving a legal-but-conservative-envelope-violating pose.
        # It should never actually change anything; clamp_action logs an ERROR
        # if it does.
        # send_action returns what lerobot ACTUALLY sent after applying
        # max_relative_target, which can differ from what we asked for.
        safe = safety.clamp_action(action, profile="backstop")
        sent = self._robot.send_action(to_feature_keys(safe))  # type: ignore[attr-defined]
        return from_feature_keys(sent)

    def disconnect(self) -> None:
        self._robot.disconnect()  # type: ignore[attr-defined]
        if not config.DISABLE_TORQUE_ON_DISCONNECT:
            print("  (arm still holding position — power it down or use torque-off to release)")

    def disable_torque(self) -> None:
        """Go limp. The arm WILL drop — the caller must have warned the operator."""
        self._robot.bus.disable_torque()  # type: ignore[attr-defined]

    def torque_disabled(self) -> AbstractContextManager[None]:
        """Torque off inside the block, guaranteed back on after it.

        Used by capture_home so the operator can pose the arm by hand.
        """
        return self._robot.bus.torque_disabled()  # type: ignore[attr-defined,no-any-return]


class DryRunArm:
    """Stand-in arm that prints instead of moving.

    Exists so that dry-run exercises the same code path as real motion rather
    than a pile of ``if DRY_RUN`` branches around every send.
    """

    def __init__(self, start: Action | None = None) -> None:
        self._pose: Action = dict(start or config.HOME_POSE)
        self._sends = 0

    @property
    def label(self) -> str:
        return "DRY-RUN arm (no hardware)"

    def read_positions(self) -> Action:
        return dict(self._pose)

    def send(self, action: Action) -> Action:
        # Same clamp as RealArm.send, so dry-run exercises the real guard.
        action = safety.clamp_action(action, profile="backstop")
        self._pose = {**self._pose, **action}
        self._sends += 1
        # One line per send is unreadable at 25 Hz; sample it.
        if self._sends % config.CONTROL_HZ == 1:
            pretty = " ".join(f"{j}={v:+.1f}" for j, v in sorted(action.items()))
            print(f"  [dry-run] send #{self._sends}: {pretty}")
        return action

    def disconnect(self) -> None:
        print(f"  [dry-run] disconnect after {self._sends} sends")

    def disable_torque(self) -> None:
        print("  [dry-run] torque OFF (arm would go limp)")

    def torque_disabled(self) -> AbstractContextManager[None]:
        print("  [dry-run] torque OFF for hand-posing; back ON when done")
        return nullcontext()


# --- Ports ---------------------------------------------------------------


def find_serial_ports() -> list[str]:
    """All candidate SO-101 serial ports, sorted. macOS: /dev/tty.usbmodem*."""
    return sorted(glob.glob(config.SERIAL_PORT_GLOB))


def resolve_follower_port(port: str | None = None, interactive: bool = True) -> str:
    """Work out which serial port is the follower, asking rather than guessing."""
    if port:
        return port
    if config.FOLLOWER_PORT:
        return config.FOLLOWER_PORT

    ports = find_serial_ports()
    if not ports:
        raise RuntimeError(
            f"No serial ports matching {config.SERIAL_PORT_GLOB}. "
            "Plug in the SO-101 follower (and check the cable is data, not charge-only)."
        )
    if len(ports) == 1:
        # Do NOT silently accept the only port: with just the LEADER plugged in
        # it would be the only port too, and we would then configure the leader
        # and offer to write the follower's calibration onto its motors.
        if not interactive:
            raise RuntimeError(
                f"Only {ports[0]} is present and it has not been confirmed as the follower. "
                "Set ARMANI_FOLLOWER_PORT in .env, or run interactively."
            )
        print(f"\nExactly one serial port is present: {ports[0]}")
        answer = input("Is this the FOLLOWER arm (not the leader)? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            raise RuntimeError("Follower port not confirmed; refusing to guess which arm this is.")
        print(f"Add ARMANI_FOLLOWER_PORT={ports[0]} to .env to skip this next time.")
        return ports[0]

    if not interactive:
        raise RuntimeError(
            f"Multiple serial ports found ({', '.join(ports)}) and no ARMANI_FOLLOWER_PORT set. "
            "Set it in .env or run interactively."
        )

    print("\nMultiple serial ports found. Which one is the FOLLOWER arm?")
    print("(Unplug the follower, re-run this list, and see which entry disappears.)")
    for index, candidate in enumerate(ports):
        print(f"  [{index}] {candidate}")
    while True:
        raw = input(f"Follower port index [0-{len(ports) - 1}]: ").strip()
        if raw.isdigit() and int(raw) < len(ports):
            chosen = ports[int(raw)]
            print(f"Using {chosen}. Add ARMANI_FOLLOWER_PORT={chosen} to .env to skip this next time.")
            return chosen
        print("Not a valid index.")


# --- Connection ----------------------------------------------------------


def connect(
    port: str | None = None,
    robot_id: str | None = None,
    dry_run: bool | None = None,
    interactive: bool = True,
) -> Arm:
    """Connect to the follower using the calibration that already exists.

    Never recreates calibration. If the motors have lost their calibration
    registers, the operator is offered a write of the EXISTING file back to the
    motors — which is not the same as recording new ranges.
    """
    if dry_run is None:
        dry_run = config.DRY_RUN
    if dry_run:
        print("[dry-run] not opening a serial port; using a simulated arm.")
        return DryRunArm()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    resolved_port = resolve_follower_port(port, interactive=interactive)
    resolved_id = robot_id or config.FOLLOWER_ID

    robot_config = SO101FollowerConfig(
        port=resolved_port,
        id=resolved_id,
        use_degrees=config.USE_DEGREES,
        max_relative_target=config.MAX_RELATIVE_TARGET,
        disable_torque_on_disconnect=config.DISABLE_TORQUE_ON_DISCONNECT,
    )
    robot = SO101Follower(robot_config)

    if not robot.calibration_fpath.is_file():
        raise RuntimeError(
            f"No calibration file at {robot.calibration_fpath}. Expected the existing "
            f"calibration for id={resolved_id!r}. Refusing to create one — check "
            "ARMANI_FOLLOWER_ID against ~/.cache/huggingface/lerobot/calibration/robots/so_follower/."
        )

    log.info("connecting to %s (id=%s)", resolved_port, resolved_id)
    robot.connect(calibrate=False)

    # From here the motors are energised, so every failure path must disconnect.
    # is_calibrated reads from the bus and can raise on a flaky cable.
    try:
        if not robot.is_calibrated:
            _recover_calibration(robot, interactive=interactive)
    except Exception:
        _safe_disconnect(robot)
        raise

    arm = RealArm(robot, resolved_port, resolved_id)

    # Park Goal_Position at where the arm actually is. lerobot's enable_torque
    # writes only Torque_Enable and Lock, so until something sets a goal the
    # servos hold whatever target was last written — possibly from a previous
    # session. NOTE this cannot prevent a twitch during connect() itself:
    # configure() runs inside bus.torque_disabled() and re-enables torque before
    # we get control. It does stop the arm drifting to a stale target afterwards.
    try:
        arm.send(arm.read_positions())
    except Exception as exc:
        log.warning("could not park the goal position after connect: %s", exc)

    log_event("robot_connected", port=resolved_port, robot_id=resolved_id)
    return arm


def _recover_calibration(robot: object, interactive: bool) -> None:
    """Push the existing calibration file to the motors, with consent.

    The motors' calibration registers can drift out of sync with the file.
    Writing the saved file back is safe and non-destructive. Recording NEW
    ranges is what we must never do, and is not what this does.

    Raises on every unhappy path and never disconnects: the caller owns the
    connection and disconnects once, so cleanup cannot happen twice.
    """
    message = (
        "The motors' stored calibration does not match "
        f"{robot.calibration_fpath}.\n"  # type: ignore[attr-defined]
        "Writing the EXISTING file to the motors fixes this without recalibrating."
    )
    log.warning(message)
    if not interactive:
        raise RuntimeError(message + " Re-run interactively to apply it.")

    print("\n" + message)
    print(
        "  WARNING: only do this if you are certain this port is the FOLLOWER.\n"
        "  Writing follower calibration onto the leader's motors would corrupt it."
    )
    try:
        answer = input(f"Write {robot.calibration_fpath} to the motors? [y/N] ").strip().lower()  # type: ignore[attr-defined]
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer not in ("y", "yes"):
        raise RuntimeError("Calibration mismatch not resolved; refusing to move an uncalibrated arm.")

    # Homing_Offset and the position limits live in EEPROM, which Feetech
    # servos only accept reliably with torque off. connect() ran configure(),
    # which leaves torque ON, so a write here would otherwise be silently
    # dropped while we printed success.
    with robot.bus.torque_disabled():  # type: ignore[attr-defined]
        robot.bus.write_calibration(robot.calibration)  # type: ignore[attr-defined]

    if not robot.is_calibrated:  # type: ignore[attr-defined]
        raise RuntimeError(
            "Calibration write did not take — the motors still disagree with the file. "
            "Do not move this arm. Check the port really is the follower and power-cycle it."
        )

    log_event("calibration_rewritten", path=str(robot.calibration_fpath))  # type: ignore[attr-defined]
    print("Existing calibration written to the motors and verified. The file was not modified.")


def _safe_disconnect(robot: object) -> None:
    """Disconnect without masking the error that got us here."""
    try:
        robot.disconnect()  # type: ignore[attr-defined]
    except Exception as exc:
        log.warning("disconnect during error handling failed: %s", exc)


# --- Motion --------------------------------------------------------------


def read_positions(arm: Arm) -> Action:
    return arm.read_positions()


def goto(
    arm: Arm,
    target: Action,
    duration: float,
    ignore_stop: bool = False,
    profile: str = config.DEFAULT_PROFILE,
) -> Action:
    """Clamp, interpolate and stream ``target`` to the arm.

    ``profile`` picks the envelope the TARGET is clamped against: "policy" for
    LLM/IK-derived targets, "recorded" for replayed or measured ones (see
    config.LIMIT_PROFILES).

    Returns the last action actually sent. If the kill switch fires mid-move,
    the arm stops where it is and is walked home before returning.
    """
    current = arm.read_positions()
    steps = list(
        safety.interp_move(current, target, duration, hz=config.CONTROL_HZ, profile=profile)
    )
    period = 1.0 / config.CONTROL_HZ

    log_event(
        "goto",
        target={k: round(v, 2) for k, v in target.items()},
        profile=profile,
        steps=len(steps),
        seconds=round(len(steps) * period, 2),
    )

    last = current
    next_deadline = time.perf_counter()
    for step in steps:
        if not ignore_stop and safety.stop_requested():
            # Safety rule 7: freeze. Stop commanding, hold position, and let the
            # operator decide. Never auto-drive anywhere from the kill switch.
            log.warning("stop requested mid-move; freezing at %s", {k: round(v, 1) for k, v in last.items()})
            log_event("goto_interrupted", at={k: round(v, 2) for k, v in last.items()})
            safety.handle_freeze(arm, current)  # type: ignore[arg-type]
            return last
        last = arm.send(step)
        next_deadline += period
        remaining = next_deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            # Fell behind schedule; don't accumulate debt.
            next_deadline = time.perf_counter()
    return last


def home(arm: Arm, slow: bool = True, ignore_stop: bool = False) -> Action:
    """Move to the verified home pose. Slow by default.

    Safety rule 4: refuses outright until the home pose has been captured on
    hardware by scripts/capture_home.py. Auto-driving to a pose nobody has ever
    checked is exactly the failure this rule exists to prevent.

    Uses the "recorded" profile because the verified home is a pose the operator
    physically set — measured reality, like a recording. capture_home warns at
    capture time if it falls outside the policy envelope.
    """
    if not config.HOME_VERIFIED:
        raise RuntimeError(
            "home pose is not verified — refusing to move to it (safety rule 4). "
            f"Run: python scripts/capture_home.py   (writes {config.HOME_POSE_PATH})"
        )
    duration = config.HOME_DURATION_S if slow else config.HOME_DURATION_S / 2
    log.info("homing over %.1fs", duration)
    return goto(
        arm, dict(config.HOME_POSE), duration, ignore_stop=ignore_stop, profile="recorded"
    )
