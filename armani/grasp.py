"""Hover over a point on the table. Stage 4 stops here — no descent, no gripper.

This is the only module in stage 4 that moves the arm, so it is where safety
rules 2, 3 and 4 are enforced for vision-driven motion:

* rule 3 — the target (x, y) must be inside the calibrated table polygon, checked
  BEFORE any solving or motion. An uncalibrated system has an empty polygon and
  therefore fails closed.
* rule 2 — the joint target comes out of the policy-limited IK solver and is then
  passed through ``clamp_action(profile="policy")`` anyway, and ``goto``
  interpolates from the measured start.
* rule 4 — the move runs inside ``SafeMotion``, so any exception returns the arm
  to the pose the move began at.

The stage-4 floor is structural, not a convention: ``hover_z`` is the lowest Z
this module will compute or command, and ``_assert_hover_only`` raises if
anything asks for less. The descent belongs to stage 5 and cannot happen by
accident from here.

``hover_over`` returns False rather than raising when the target is off the
table or unreachable. That is not an error — it is the answer, and stage 6's
gates will read it as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from armani import calibrate, config, kinematics, motion, safety
from armani.logutil import get_logger, log_event

log = get_logger("grasp")

# How long the arm takes to travel to a hover pose. Slow enough for the operator
# to watch it and reach the kill switch.
HOVER_DURATION_S = 3.0


@dataclass(frozen=True)
class HoverResult:
    """Why the arm did or did not move, in enough detail to log and to speak."""

    ok: bool
    reason: str = ""
    robot_xy: tuple[float, float] | None = None
    joints: dict[str, float] | None = None
    position_error_m: float = float("nan")
    tilt_deg: float = float("nan")
    reachability_margin: float = 0.0
    moved: bool = False

    def __bool__(self) -> bool:
        return self.ok


def _assert_hover_only(z: float) -> None:
    """Stage-4 guard: nothing here may command a Z below the hover plane.

    A coding guard, not a runtime policy — if this ever fires it means someone
    started stage 5 inside stage 4's module.
    """
    floor = config.hover_z()
    if z < floor - 1e-9:
        raise AssertionError(
            f"refusing to command z={z:.4f} m, below the stage-4 hover floor {floor:.4f} m. "
            "Descent and grasping are stage 5."
        )


def plan_hover(x: float, y: float, start_pose: dict[str, float]) -> HoverResult:
    """Everything that decides whether the arm may move — and no motion at all.

    Split out from ``hover_over`` so the whole decision path is testable without
    a robot, and so a caller can ask "could you?" without asking "please do".
    """
    if not calibrate.point_in_polygon(x, y, margin_m=config.POLYGON_MARGIN_M):
        if not config.TABLE_POLYGON:
            reason = (
                "no calibrated table polygon — the camera has not been mapped to the table. "
                "Run: python scripts/calibrate_camera.py"
            )
        else:
            reason = f"({x:.3f}, {y:.3f}) m is outside the calibrated table"
        return HoverResult(False, reason, robot_xy=(x, y))

    z = config.hover_z()
    _assert_hover_only(z)
    try:
        solution = kinematics.solve_top_down(x, y, z, start_pose)
    except kinematics.KinematicsUnavailable as exc:
        return HoverResult(False, f"inverse kinematics unavailable: {exc}", robot_xy=(x, y))

    if not solution.ok:
        return HoverResult(
            False,
            f"cannot reach ({x:.3f}, {y:.3f}) m at hover height: {solution.reason}",
            robot_xy=(x, y),
            joints=solution.joints or None,
            position_error_m=solution.position_error_m,
            tilt_deg=solution.tilt_deg,
        )

    # Belt and braces. The solver was given the policy limits, so this should
    # change nothing — but clamp_action is the single guard safety rule 2 names,
    # and the target must pass through it no matter where it came from.
    joints = safety.clamp_action(solution.joints, profile="policy")

    return HoverResult(
        True,
        "",
        robot_xy=(x, y),
        joints=joints,
        position_error_m=solution.position_error_m,
        tilt_deg=solution.tilt_deg,
        reachability_margin=solution.reachability_margin,
    )


def hover_over(arm, x: float, y: float, duration: float = HOVER_DURATION_S) -> HoverResult:
    """Move the gripper to hover ``HOVER_HEIGHT_M`` above the table point (x, y).

    Returns a falsy HoverResult without moving when the point is off the table
    or out of reach. The gripper is never commanded: stage 4 does not actuate it,
    so it is left out of the target entirely and holds wherever it is.
    """
    start_pose = arm.read_positions()
    plan = plan_hover(x, y, start_pose)

    log_event(
        "hover_plan",
        x=round(x, 4),
        y=round(y, 4),
        z=round(config.hover_z(), 4),
        ok=plan.ok,
        reason=plan.reason,
        # isfinite, not isnan: an unreachable target carries inf, and json.dumps
        # writes that as the literal `Infinity`, which is not valid JSON and
        # would break every reader of the decision log.
        tilt_deg=round(plan.tilt_deg, 1) if math.isfinite(plan.tilt_deg) else None,
        position_error_mm=(
            round(plan.position_error_m * 1000, 1)
            if math.isfinite(plan.position_error_m)
            else None
        ),
        joints=None if plan.joints is None else {j: round(v, 2) for j, v in plan.joints.items()},
    )

    if not plan.ok:
        log.warning("not hovering: %s", plan.reason)
        return plan

    assert plan.joints is not None  # guaranteed by plan.ok
    log.info(
        "hovering over (%.3f, %.3f) m at z=%.3f m — %.0f mm IK error, %.0f deg lean",
        x, y, config.hover_z(), plan.position_error_m * 1000, plan.tilt_deg,
    )

    with safety.SafeMotion(arm, description=f"hover over ({x:.3f}, {y:.3f})"):
        motion.goto(arm, dict(plan.joints), duration=duration, profile="policy")

    # goto returns early and holds when the kill switch fires (safety rule 7).
    # Reporting success then would be a lie, and stage 6's gates would read it
    # as "the arm is over the object" when it is stopped somewhere en route.
    if safety.stop_requested():
        log_event("hover_interrupted", x=round(x, 4), y=round(y, 4))
        return HoverResult(
            False,
            "stopped by the kill switch before reaching the hover pose",
            robot_xy=plan.robot_xy,
            joints=plan.joints,
            position_error_m=plan.position_error_m,
            tilt_deg=plan.tilt_deg,
            moved=True,
        )

    log_event("hover_done", x=round(x, 4), y=round(y, 4))
    return HoverResult(
        True,
        plan.reason,
        robot_xy=plan.robot_xy,
        joints=plan.joints,
        position_error_m=plan.position_error_m,
        tilt_deg=plan.tilt_deg,
        reachability_margin=plan.reachability_margin,
        moved=True,
    )


def combined_confidence(vision_confidence: float, result: HoverResult) -> float:
    """Vision confidence tempered by how comfortably the arm can get there.

    CLAUDE.md's confidence recipe folds in an IK reachability margin, and this
    is where that happens — eyes.py must not know about the arm. A target the
    arm can only just reach lowers the score even when the camera is certain.
    Stage 6 owns what to DO with the number; this only computes it.
    """
    reach = 0.0 if not result.ok else result.reachability_margin
    return round(max(0.0, min(1.0, vision_confidence * (0.5 + 0.5 * reach))), 3)
