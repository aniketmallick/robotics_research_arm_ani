"""Forward and inverse kinematics for the SO-101. Pure geometry — no motion.

Plan A of the IK ladder: lerobot 0.5.2's ``RobotKinematics`` (placo + the
SO-101 URDF). Nothing here talks to a motor; ``grasp.py`` owns that.

Three facts drive the design, all of them measured rather than assumed:

1. ``RobotKinematics.inverse_kinematics`` is ONE soft-QP step, not an
   iterate-to-convergence solve, and it returns NO success flag. lerobot calls
   it once per teleop tick because each delta is tiny; a one-shot absolute
   solve has to iterate and then CHECK. So every solution here is verified by
   forward kinematics before it is handed back, and a solution that misses is
   reported as a failure rather than returned as a plausible-looking pose.

2. The solver is told about the POLICY envelope (config.JOINT_LIMITS) via
   placo's own joint limits, so its solutions are policy-legal by construction
   and ``safety.clamp_action`` has nothing left to do. That ordering matters:
   solving first and clamping afterwards would move the arm somewhere the
   verification never checked.

3. Inside the policy envelope a near-vertical approach only exists close to the
   table (z <= 0 m); at a 10 cm hover the best achievable lean is roughly 35
   degrees. Hover therefore constrains POSITION hard and treats the approach
   direction as a soft preference with a reported, bounded tilt.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from armani import config
from armani.logutil import get_logger

log = get_logger("kinematics")

# The QP has settled once a step moves every joint less than this (degrees).
# Far below IK_POSITION_TOLERANCE_M in effect: 1e-4 deg over a 0.4 m arm is
# well under a micron of tool travel, so stopping here cannot cost accuracy.
IK_SETTLED_DEG = 1e-4

# Reentrant on purpose: get_solver() takes it to build the cache, and the solve
# and FK helpers take it around every call because placo's solver carries mutable
# state — the stage-3 agent runs motion on a worker thread, so two concurrent
# solves would interleave and quietly return each other's answers.
_solver_lock = threading.RLock()
_solver = None  # cached RobotKinematics; building it parses the URDF


class KinematicsUnavailable(RuntimeError):
    """placo or the URDF is missing, so Plan A of the IK ladder cannot run.

    Deliberately distinct from "this target is unreachable": one is a broken
    install the operator must fix, the other is a normal answer about geometry.
    """


@dataclass(frozen=True)
class IKSolution:
    """The result of asking for a pose. ``ok`` is the only thing callers trust."""

    ok: bool
    joints: dict[str, float]  # policy-legal joint targets, degrees
    position_error_m: float  # FK(joints) vs the requested point
    tilt_deg: float  # approach axis away from straight down
    reason: str = ""

    @property
    def reachability_margin(self) -> float:
        """0..1 — how comfortably this solution met the tolerance.

        1.0 is a bullseye, 0.0 is at or beyond the tolerance. Fed into the
        confidence number so "the arm can barely get there" lowers it.
        """
        if config.IK_POSITION_TOLERANCE_M <= 0:
            return 0.0
        margin = 1.0 - (self.position_error_m / config.IK_POSITION_TOLERANCE_M)
        return max(0.0, min(1.0, margin))


def available() -> bool:
    """True when Plan A can actually run. Never raises — callers use it to SKIP."""
    try:
        get_solver()
    except KinematicsUnavailable:
        return False
    return True


def get_solver():
    """Build (once) a placo solver constrained to the POLICY joint limits.

    Cached because parsing the URDF and building the QP is slow enough to notice
    at 25 Hz, and locked because the stage-3 agent runs motion on a worker
    thread — placo's solver holds mutable state, so two threads solving at once
    would interleave and produce nonsense.
    """
    global _solver
    with _solver_lock:
        if _solver is not None:
            return _solver

        if not config.URDF_PATH.is_file():
            raise KinematicsUnavailable(
                f"no URDF at {config.URDF_PATH}. See docs/env_report.md, 'URDF provenance'."
            )
        try:
            from lerobot.model.kinematics import RobotKinematics
        except ImportError as exc:
            raise KinematicsUnavailable(f"lerobot kinematics not importable: {exc}") from exc

        try:
            solver = RobotKinematics(
                str(config.URDF_PATH),
                target_frame_name=config.IK_TARGET_FRAME,
                joint_names=list(config.IK_JOINTS),
            )
            # Teach the solver our policy envelope so it cannot return a target
            # that clamp_action would later have to rewrite behind the
            # verification. Inside the same try as construction on purpose: if a
            # placo upgrade renames these, the limits would silently not be
            # applied, and every solve after that could leave the envelope.
            # Failing loudly here is the only safe response.
            for joint in config.IK_JOINTS:
                low, high = config.JOINT_LIMITS[joint]
                solver.robot.set_joint_limits(joint, math.radians(low), math.radians(high))
            solver.solver.enable_joint_limits(True)
        except Exception as exc:
            # placo is an optional lerobot extra and its binary wheels are
            # fussy; a missing dylib surfaces here as ImportError at call time.
            raise KinematicsUnavailable(
                f"could not build the kinematics solver: {type(exc).__name__}: {exc}. "
                "placo must be installed in the lerobot env — see docs/env_report.md."
            ) from exc

        _solver = solver
        log.info("kinematics ready (%s, frame=%s)", config.URDF_PATH.name, config.IK_TARGET_FRAME)
        return _solver


# --- Pose helpers --------------------------------------------------------


def joints_to_vector(pose: dict[str, float]) -> np.ndarray:
    """Pull the IK joints out of a full arm pose, in URDF order."""
    missing = [j for j in config.IK_JOINTS if j not in pose]
    if missing:
        raise ValueError(f"pose is missing joint(s) {missing}")
    return np.array([float(pose[j]) for j in config.IK_JOINTS], dtype=float)


def vector_to_joints(vector: np.ndarray) -> dict[str, float]:
    return {joint: float(vector[i]) for i, joint in enumerate(config.IK_JOINTS)}


def forward(pose: dict[str, float]) -> np.ndarray:
    """4x4 transform of the gripper frame for a measured or planned pose."""
    solver = get_solver()
    with _solver_lock:
        return np.array(solver.forward_kinematics(joints_to_vector(pose)))


def tool_position(pose: dict[str, float]) -> tuple[float, float, float]:
    """Gripper position in robot-base metres. Used to check where the arm IS."""
    translation = forward(pose)[:3, 3]
    return (float(translation[0]), float(translation[1]), float(translation[2]))


def top_down_pose(x: float, y: float, z: float) -> np.ndarray:
    """Target transform: gripper at (x, y, z) with its approach axis pointing down.

    The tool's approach axis is the gripper frame's local Z (verified against FK:
    driving wrist_flex to -90 turns it to face straight up). The remaining
    freedom is roll about that axis; we align the tool's local X with the inward
    radial direction, which is the orientation the arm naturally adopts when it
    reaches out along its own shoulder_pan angle, so the solver is not asked to
    fight its own geometry for nothing.
    """
    yaw = math.atan2(y, x)
    x_axis = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, x_axis)

    target = np.eye(4)
    target[:3, 0] = x_axis
    target[:3, 1] = y_axis
    target[:3, 2] = z_axis
    target[:3, 3] = (x, y, z)
    return target


def tilt_from_down(transform: np.ndarray) -> float:
    """Degrees between the tool's approach axis and straight down. 0 = vertical."""
    # transform[2, 2] is the world-Z component of the tool's local Z axis, so it
    # is -1 when the tool points straight down.
    cosine = float(np.clip(-transform[2, 2], -1.0, 1.0))
    return math.degrees(math.acos(cosine))


# --- Inverse kinematics --------------------------------------------------


def _solve_once(
    target: np.ndarray, start: np.ndarray, orientation_weight: float
) -> tuple[np.ndarray, float, float] | None:
    """Iterate the single-step solver to rest. Returns (joints, error_m, tilt_deg).

    None means the solver produced non-finite joints, which is a broken solve
    rather than an unreachable target.
    """
    solver = get_solver()
    with _solver_lock:
        joints = start
        for _ in range(config.IK_MAX_ITERATIONS):
            previous = joints
            joints = solver.inverse_kinematics(
                joints, target, position_weight=1.0, orientation_weight=orientation_weight
            )
            if not np.all(np.isfinite(joints)):
                return None
            # Each call is a single QP step, so the sequence settles rather than
            # terminating. Stop once it stops moving instead of always paying for
            # the full iteration budget — this runs while the operator waits.
            if np.max(np.abs(joints - previous)) < IK_SETTLED_DEG:
                break
        achieved = np.array(solver.forward_kinematics(joints))

    error = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
    return joints, error, tilt_from_down(achieved)


def solve_top_down(
    x: float,
    y: float,
    z: float,
    start_pose: dict[str, float],
    max_tilt_deg: float | None = None,
) -> IKSolution:
    """Joint targets that put the gripper at (x, y, z) facing as far down as it can.

    Returns a solution whose ``ok`` is False rather than raising when the point
    simply cannot be reached inside the policy envelope — an unreachable target
    is a fact about geometry and a real confidence signal, not an error.

    Never raises for a bad target. Raises only KinematicsUnavailable, which
    means the install is broken.
    """
    if max_tilt_deg is None:
        max_tilt_deg = config.HOVER_MAX_TILT_DEG

    for name, value in (("x", x), ("y", y), ("z", z)):
        if not math.isfinite(value):
            return IKSolution(False, {}, math.inf, math.inf, f"{name} is not a finite number")

    target = top_down_pose(x, y, z)
    start = joints_to_vector(start_pose)

    # The frame task is SOFT, so position and orientation trade against each
    # other: at a fixed orientation weight the solver will happily give up
    # centimetres of position to gain degrees of verticality. At hover height
    # position is what puts the gripper over the object and the lean is
    # cosmetic, so walk the weight down and keep the FIRST (most vertical)
    # solution that still lands inside the position tolerance. Measured cost of
    # the whole ladder is ~130 ms worst case, which is nothing next to the
    # Gemini call that produced the target and is never run in a control loop.
    #
    # Note the deliberate bias: this spends the whole position budget buying
    # verticality. Tighten IK_POSITION_TOLERANCE_M to bias the other way.
    best: tuple[np.ndarray, float, float] | None = None
    for weight in config.IK_ORIENTATION_WEIGHTS:
        attempt = _solve_once(target, start, weight)
        if attempt is None:
            return IKSolution(False, {}, math.inf, math.inf, "solver diverged (non-finite joints)")
        if best is None or attempt[1] < best[1]:
            best = attempt
        if attempt[1] <= config.IK_POSITION_TOLERANCE_M:
            best = attempt
            break

    assert best is not None  # IK_ORIENTATION_WEIGHTS is never empty
    joints, error, tilt = best
    solution = vector_to_joints(joints)

    # The solver was given the policy limits, so this should be a formality.
    # Verify rather than trust: a silent limit change would otherwise let an
    # out-of-envelope target through under the cover of "IK said so".
    outside = [
        f"{joint}={value:.1f}"
        for joint, value in solution.items()
        if not (
            config.JOINT_LIMITS[joint][0] - 0.5
            <= value
            <= config.JOINT_LIMITS[joint][1] + 0.5
        )
    ]
    if outside:
        return IKSolution(
            False, solution, error, tilt,
            f"solution left the policy envelope: {', '.join(outside)}",
        )

    if error > config.IK_POSITION_TOLERANCE_M:
        return IKSolution(
            False, solution, error, tilt,
            f"closest reachable pose is {error * 1000:.0f} mm away "
            f"(tolerance {config.IK_POSITION_TOLERANCE_M * 1000:.0f} mm)",
        )
    if tilt > max_tilt_deg:
        return IKSolution(
            False, solution, error, tilt,
            f"approach leans {tilt:.0f} deg off vertical (limit {max_tilt_deg:.0f} deg)",
        )

    return IKSolution(True, solution, error, tilt)
