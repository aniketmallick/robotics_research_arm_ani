"""Pure-logic tests for armani.safety. No hardware, no lerobot, no operator.

This is the only code standing between a bad number and the motors.

The anchor pose below is REAL: what the follower reported on 2026-07-18, parked
against its stops. Under the original single-envelope design that pose made
home() — and therefore the kill switch and every recovery path — refuse to move.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import config, safety  # noqa: E402

REST_POSE: dict[str, float] = {
    "shoulder_pan": 3.78,
    "shoulder_lift": -111.69,  # past the calibrated -111.0, as a real stop reads
    "elbow_flex": 96.79,
    "wrist_flex": 68.92,
    "wrist_roll": 1.36,
    "gripper": 31.12,
}
MAX_STEP = config.MAX_JOINT_SPEED / config.CONTROL_HZ


# --- clamp_action --------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [{"shoulder_pan": float("nan")}, {"shoulder_pan": float("inf")},
     {"shoulder_pan": float("-inf")}, {"nope": 1.0}, {"gripper": "x"}],
)
def test_clamp_rejects_bad_input(action):
    with pytest.raises(ValueError):
        safety.clamp_action(action)


def test_clamp_rejects_unknown_profile():
    with pytest.raises(ValueError):
        safety.clamp_action({"shoulder_pan": 0.0}, profile="nonsense")


@pytest.mark.parametrize("profile", sorted(config.LIMIT_PROFILES))
def test_clamp_lands_exactly_on_the_boundary(profile):
    for joint, (low, high) in config.LIMIT_PROFILES[profile].items():
        assert safety.clamp_action({joint: high + 1000}, profile=profile)[joint] == high
        assert safety.clamp_action({joint: low - 1000}, profile=profile)[joint] == low


def test_policy_constrains_llm_targets_but_recorded_does_not():
    """Why two profiles exist: real teleop legitimately exceeds policy."""
    value = REST_POSE["shoulder_lift"]
    policy = safety.clamp_action({"shoulder_lift": value}, profile="policy")["shoulder_lift"]
    recorded = safety.clamp_action({"shoulder_lift": value}, profile="recorded")["shoulder_lift"]
    assert policy == config.JOINT_LIMITS["shoulder_lift"][0]  # clipped ~52 deg
    assert abs(recorded - value) <= config.RECORDED_MARGIN + config.PHYSICAL_TOLERANCE


def test_gripper_keeps_full_travel_in_every_profile():
    """The gripper is a percentage; shrinking it by a degree margin is a unit error."""
    for profile, limits in config.LIMIT_PROFILES.items():
        assert limits["gripper"] == (0.0, 100.0), profile


def test_clamp_does_not_mutate_its_input():
    original = {"shoulder_pan": 250.0}
    safety.clamp_action(original)
    assert original == {"shoulder_pan": 250.0}


# --- interp_move ---------------------------------------------------------


def test_final_step_lands_exactly_on_target():
    steps = list(safety.interp_move({"wrist_roll": 0.0}, {"wrist_roll": 30.0}, 2.0))
    assert steps[-1]["wrist_roll"] == pytest.approx(30.0)


def test_speed_limit_stretches_the_step_count():
    """90 deg asked for in 0.1 s must take as many steps as the speed limit needs."""
    steps = list(safety.interp_move({"shoulder_pan": 0.0}, {"shoulder_pan": 90.0}, 0.1))
    assert len(steps) == pytest.approx(90.0 / MAX_STEP, abs=1)
    values = [0.0] + [s["shoulder_pan"] for s in steps]
    assert max(abs(b - a) for a, b in zip(values, values[1:])) <= MAX_STEP + 1e-9


def test_zero_delta_yields_exactly_one_step():
    steps = list(safety.interp_move({"gripper": 50.0}, {"gripper": 50.0}, 1.0))
    assert steps == [{"gripper": 50.0}]


def test_missing_current_position_is_rejected():
    with pytest.raises(ValueError):
        list(safety.interp_move({}, {"gripper": 10.0}, 1.0))


def test_envelope_refusal_fires_beyond_physical():
    """A reading the hardware cannot produce is an encoder fault, not a pose."""
    broken = {**REST_POSE, "shoulder_lift": -114.0}  # physical -111.0, tolerance 2.0
    with pytest.raises(safety.OutsideEnvelopeError):
        list(safety.interp_move(broken, dict(config.HOME_POSE), 3.0, profile="recorded"))


def test_parked_outside_policy_is_a_legal_start():
    """The stage-1 regression: this pose must NOT block motion."""
    for joint in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        low, high = config.JOINT_LIMITS[joint]
        assert not low <= REST_POSE[joint] <= high, f"{joint} no longer proves anything"
    safety.check_start_pose(REST_POSE, REST_POSE)  # must not raise


def test_recovery_from_the_real_rest_pose_is_safe_and_monotone():
    """Interpolating home from the parked pose: reachable, smooth, in-bounds."""
    steps = list(safety.interp_move(REST_POSE, dict(config.HOME_POSE), 3.0, profile="recorded"))
    assert steps
    for joint in config.HOME_POSE:
        values = [REST_POSE[joint]] + [s[joint] for s in steps]
        deltas = [b - a for a, b in zip(values, values[1:]) if abs(b - a) > 1e-9]
        assert all(d > 0 for d in deltas) or all(d < 0 for d in deltas), f"{joint} backtracked"
        assert max((abs(d) for d in deltas), default=0.0) <= MAX_STEP + 1e-9
        low, high = config.LIMIT_PROFILES["backstop"][joint]
        assert all(low <= s[joint] <= high for s in steps), f"{joint} left the backstop envelope"
