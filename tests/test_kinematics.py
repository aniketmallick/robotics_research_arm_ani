"""IK/FK and the hover decision path. Needs placo + the URDF; skips without them.

No arm is touched: plan_hover is deliberately separate from hover_over so the
entire "may it move?" decision is testable on a laptop.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import calibrate, config, grasp, kinematics  # noqa: E402

pytestmark = pytest.mark.skipif(
    not kinematics.available(), reason="placo or the SO-101 URDF is unavailable"
)

REST = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 40.0,
    "wrist_flex": 30.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}

# Inside the measured top-down reach band for the policy envelope (see
# docs/env_report.md): r roughly 0.20..0.35 m at hover height.
REACHABLE = (0.26, 0.0)


# --- forward kinematics --------------------------------------------------


def test_forward_kinematics_is_deterministic():
    first = kinematics.tool_position(REST)
    second = kinematics.tool_position(REST)
    assert first == second


def test_gripper_is_not_part_of_the_ik_chain():
    """Opening the gripper must not move the tool frame."""
    opened = {**REST, "gripper": 100.0}
    assert kinematics.tool_position(REST) == kinematics.tool_position(opened)


def test_missing_joint_is_rejected():
    with pytest.raises(ValueError, match="missing joint"):
        kinematics.forward({"shoulder_pan": 0.0})


def test_reach_is_physically_plausible():
    """A ~0.4 m arm must not claim to reach a metre."""
    x, y, z = kinematics.tool_position(REST)
    assert math.sqrt(x * x + y * y + z * z) < 0.6


# --- orientation helpers -------------------------------------------------


def test_top_down_pose_points_straight_down():
    target = kinematics.top_down_pose(0.25, 0.1, 0.1)
    assert kinematics.tilt_from_down(target) == pytest.approx(0.0, abs=1e-9)


def test_top_down_pose_is_orthonormal():
    rotation = kinematics.top_down_pose(0.25, -0.08, 0.1)[:3, :3]
    product = rotation.T @ rotation
    for i in range(3):
        for j in range(3):
            assert product[i][j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)


def test_tilt_of_an_upward_pointing_tool_is_180_degrees():
    import numpy as np

    pointing_up = np.eye(4)  # local Z along world +Z, i.e. the exact opposite of down
    assert kinematics.tilt_from_down(pointing_up) == pytest.approx(180.0)


# --- inverse kinematics --------------------------------------------------


def test_reachable_target_is_solved_within_tolerance():
    solution = kinematics.solve_top_down(*REACHABLE, config.hover_z(), REST)
    assert solution.ok, solution.reason
    assert solution.position_error_m <= config.IK_POSITION_TOLERANCE_M


def test_solution_is_verified_by_forward_kinematics():
    """The solver reports no success flag, so we must check its answer ourselves."""
    solution = kinematics.solve_top_down(*REACHABLE, config.hover_z(), REST)
    assert solution.ok
    achieved = kinematics.forward({**REST, **solution.joints})
    assert achieved[0][3] == pytest.approx(REACHABLE[0], abs=config.IK_POSITION_TOLERANCE_M)
    assert achieved[1][3] == pytest.approx(REACHABLE[1], abs=config.IK_POSITION_TOLERANCE_M)


def test_solution_never_leaves_the_policy_envelope():
    for x in (0.20, 0.26, 0.32):
        solution = kinematics.solve_top_down(x, 0.0, config.hover_z(), REST)
        for joint, value in solution.joints.items():
            low, high = config.JOINT_LIMITS[joint]
            assert low - 0.5 <= value <= high + 0.5, f"{joint}={value} escaped {low}..{high}"


def test_far_out_of_reach_is_reported_not_raised():
    solution = kinematics.solve_top_down(1.5, 0.0, config.hover_z(), REST)
    assert not solution.ok
    assert solution.reachability_margin == 0.0
    assert solution.reason


def test_non_finite_target_is_rejected_cleanly():
    for bad in (float("nan"), float("inf")):
        solution = kinematics.solve_top_down(bad, 0.0, 0.1, REST)
        assert not solution.ok
        assert "finite" in solution.reason


def test_excessive_tilt_is_refused():
    """A target only reachable by leaning right over is not a hover."""
    solution = kinematics.solve_top_down(*REACHABLE, config.hover_z(), REST, max_tilt_deg=0.0)
    assert not solution.ok
    assert "lean" in solution.reason


def test_reachability_margin_is_bounded():
    for x in (0.15, 0.22, 0.26, 0.30, 0.50):
        solution = kinematics.solve_top_down(x, 0.0, config.hover_z(), REST)
        assert 0.0 <= solution.reachability_margin <= 1.0


# --- the hover decision path ---------------------------------------------


def test_hover_is_refused_without_a_table_polygon(monkeypatch):
    """Safety rule 3 must fail closed on an uncalibrated system."""
    monkeypatch.setattr(config, "TABLE_POLYGON", ())
    plan = grasp.plan_hover(*REACHABLE, REST)
    assert not plan.ok
    assert "calibrat" in plan.reason


def test_hover_is_refused_outside_the_table(monkeypatch):
    monkeypatch.setattr(config, "TABLE_POLYGON", ((0.2, -0.1), (0.3, -0.1), (0.3, 0.1), (0.2, 0.1)))
    plan = grasp.plan_hover(0.9, 0.0, REST)
    assert not plan.ok
    assert "outside" in plan.reason


def test_hover_plans_inside_a_calibrated_table(monkeypatch):
    monkeypatch.setattr(
        config, "TABLE_POLYGON", ((0.20, -0.12), (0.34, -0.12), (0.34, 0.12), (0.20, 0.12))
    )
    plan = grasp.plan_hover(*REACHABLE, REST)
    assert plan.ok, plan.reason
    assert plan.joints is not None
    assert "gripper" not in plan.joints, "stage 4 must never command the gripper"
    assert not plan.moved, "planning must not move anything"


def test_plan_result_is_falsy_when_refused(monkeypatch):
    monkeypatch.setattr(config, "TABLE_POLYGON", ())
    assert not grasp.plan_hover(*REACHABLE, REST)


def test_hover_floor_blocks_a_descent():
    """The stage-4 guard: asking for anything below the hover plane is a bug."""
    with pytest.raises(AssertionError, match="stage-4 hover floor"):
        grasp._assert_hover_only(config.hover_z() - 0.01)
    grasp._assert_hover_only(config.hover_z())  # exactly at the floor is fine


def test_hover_z_is_table_plus_hover_height():
    assert config.hover_z() == pytest.approx(config.TABLE_HEIGHT_M + config.HOVER_HEIGHT_M)


def test_combined_confidence_drops_with_a_marginal_reach():
    solid = grasp.HoverResult(True, reachability_margin=1.0)
    marginal = grasp.HoverResult(True, reachability_margin=0.0)
    assert grasp.combined_confidence(0.9, solid) > grasp.combined_confidence(0.9, marginal)


def test_combined_confidence_is_zero_when_unreachable():
    assert grasp.combined_confidence(0.9, grasp.HoverResult(False, "nope")) == pytest.approx(0.45)


# --- hover_over against a fake arm ---------------------------------------


class FakeArm:
    """Records what was commanded. Never pretends to be hardware."""

    label = "fake arm"

    def __init__(self, pose=None):
        self.pose = dict(pose or REST)
        self.sent: list[dict[str, float]] = []

    def read_positions(self):
        return dict(self.pose)

    def send(self, action):
        self.sent.append(dict(action))
        self.pose.update(action)
        return dict(action)

    def disconnect(self):
        pass

    def disable_torque(self):
        pass


TABLE = ((0.20, -0.12), (0.34, -0.12), (0.34, 0.12), (0.20, 0.12))


def test_hover_over_commands_the_arm_and_reports_moved(monkeypatch):
    monkeypatch.setattr(config, "TABLE_POLYGON", TABLE)
    arm = FakeArm()
    result = grasp.hover_over(arm, *REACHABLE, duration=0.1)
    assert result.ok, result.reason
    assert result.moved
    assert arm.sent, "hover_over must actually command the arm"


def test_hover_never_commands_the_gripper(monkeypatch):
    """Stage 4's hard rule: the gripper is not actuated at all."""
    monkeypatch.setattr(config, "TABLE_POLYGON", TABLE)
    arm = FakeArm()
    grasp.hover_over(arm, *REACHABLE, duration=0.1)
    for action in arm.sent:
        assert "gripper" not in action


def test_hover_never_commands_below_the_hover_plane(monkeypatch):
    """Forward-kinematics every commanded action; none may dip toward the table."""
    monkeypatch.setattr(config, "TABLE_POLYGON", TABLE)
    arm = FakeArm()
    start_z = kinematics.tool_position(REST)[2]
    grasp.hover_over(arm, *REACHABLE, duration=0.1)

    pose = dict(REST)
    for action in arm.sent:
        pose.update(action)
        z = kinematics.tool_position(pose)[2]
        assert z >= min(start_z, config.hover_z()) - 0.02


def test_an_unreachable_target_moves_nothing(monkeypatch):
    monkeypatch.setattr(config, "TABLE_POLYGON", TABLE)
    arm = FakeArm()
    result = grasp.hover_over(arm, 0.9, 0.9, duration=0.1)
    assert not result.ok
    assert not result.moved
    assert arm.sent == [], "a refused hover must not command anything at all"


def test_a_kill_switched_hover_does_not_report_success(monkeypatch):
    """goto returns early and holds when the kill switch fires. Saying 'ok' then
    would tell stage 6's gates the arm is over the object when it is not."""
    from armani import safety

    monkeypatch.setattr(config, "TABLE_POLYGON", TABLE)
    arm = FakeArm()
    safety.request_stop("test")
    try:
        result = grasp.hover_over(arm, *REACHABLE, duration=0.1)
    finally:
        safety.clear_stop()
    assert not result.ok
    assert "kill switch" in result.reason


def test_polygon_and_ik_agree_on_a_calibrated_table(monkeypatch):
    """A point the polygon allows should usually also be reachable — if not, the
    calibrated table is in the wrong place relative to the arm, and stage 5 will
    fail mysteriously. This documents the coupling."""
    polygon = ((0.20, -0.12), (0.34, -0.12), (0.34, 0.12), (0.20, 0.12))
    monkeypatch.setattr(config, "TABLE_POLYGON", polygon)
    centre = (
        sum(p[0] for p in polygon) / len(polygon),
        sum(p[1] for p in polygon) / len(polygon),
    )
    assert calibrate.point_in_polygon(*centre, polygon)
    assert grasp.plan_hover(*centre, REST).ok
