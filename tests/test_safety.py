#!/usr/bin/env python
"""Pure-logic regression tests for armani.safety. No hardware, no operator.

Runs standalone (``python tests/test_safety.py``) so it needs no extra
dependency, and is also collected by pytest if that ever gets installed.

The anchor case is REAL: the pose below is what the follower actually reported
on 2026-07-18, parked against its mechanical stops. Under the original
single-envelope design that pose made home() — and therefore the kill switch
and every error-recovery path — refuse to move. These tests exist so that
cannot come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from armani import config, safety  # noqa: E402

# Measured on hardware 2026-07-18. shoulder_lift, elbow_flex and wrist_flex are
# all far outside the +-60 policy envelope; shoulder_lift even reads past its
# calibrated -111.0 because lerobot's DEGREES conversion does not clamp reads.
REST_POSE: dict[str, float] = {
    "shoulder_pan": 3.78,
    "shoulder_lift": -111.69,
    "elbow_flex": 96.79,
    "wrist_flex": 68.92,
    "wrist_roll": 1.36,
    "gripper": 31.12,
}

MAX_STEP = config.MAX_JOINT_SPEED / config.CONTROL_HZ


def test_home_from_real_rest_pose_is_accepted() -> None:
    """The regression: homing from the parked pose must be allowed."""
    steps = list(safety.interp_move(REST_POSE, dict(config.HOME_POSE), 3.0, profile="recorded"))
    assert steps, "interp_move produced no steps"
    for joint, target in config.HOME_POSE.items():
        assert abs(steps[-1][joint] - target) < 1e-6, f"{joint} did not reach {target}"


def test_home_from_rest_is_monotone_per_joint() -> None:
    """No joint may overshoot or backtrack on the way home."""
    steps = list(safety.interp_move(REST_POSE, dict(config.HOME_POSE), 3.0, profile="recorded"))
    for joint in config.HOME_POSE:
        values = [REST_POSE[joint]] + [s[joint] for s in steps]
        deltas = [b - a for a, b in zip(values, values[1:])]
        moving = [d for d in deltas if abs(d) > 1e-9]
        if not moving:
            continue
        assert all(d > 0 for d in moving) or all(d < 0 for d in moving), (
            f"{joint} changed direction mid-move"
        )


def test_home_from_rest_respects_the_speed_limit() -> None:
    steps = list(safety.interp_move(REST_POSE, dict(config.HOME_POSE), 3.0, profile="recorded"))
    for joint in config.HOME_POSE:
        values = [REST_POSE[joint]] + [s[joint] for s in steps]
        for a, b in zip(values, values[1:]):
            assert abs(b - a) <= MAX_STEP + 1e-9, f"{joint} step {abs(b - a):.3f} exceeds {MAX_STEP:.3f}"


def test_every_step_stays_inside_physical_limits() -> None:
    steps = list(safety.interp_move(REST_POSE, dict(config.HOME_POSE), 3.0, profile="recorded"))
    for step in steps:
        for joint, value in step.items():
            low, high = config.PHYSICAL_LIMITS[joint]
            assert low <= value <= high, f"{joint}={value} outside physical {low}..{high}"


def test_beyond_physical_is_refused() -> None:
    """A reading the hardware cannot produce is a fault, and must block motion."""
    broken = {**REST_POSE, "shoulder_lift": -114.0}  # physical -111.0, tolerance 2.0
    try:
        list(safety.interp_move(broken, dict(config.HOME_POSE), 3.0, profile="recorded"))
    except safety.OutsideEnvelopeError:
        return
    raise AssertionError("a start pose beyond physical+tolerance was accepted")


def test_parked_outside_policy_is_not_refused() -> None:
    """The whole point: outside policy is legal to start from, unlike before."""
    for joint in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        low, high = config.JOINT_LIMITS[joint]
        assert not low <= REST_POSE[joint] <= high, (
            f"{joint} is inside the policy envelope; this test no longer proves anything"
        )
    safety.check_start_pose(REST_POSE, REST_POSE)  # must not raise


def test_policy_profile_still_constrains_llm_targets() -> None:
    """Conservative envelope must survive for LLM/IK-originated targets."""
    clamped = safety.clamp_action({"shoulder_lift": -111.0}, profile="policy")
    assert clamped["shoulder_lift"] == config.JOINT_LIMITS["shoulder_lift"][0]


def test_recorded_profile_does_not_meaningfully_clip_a_measured_entry_pose() -> None:
    """Returning to where the arm actually was must not be clipped by POLICY.

    "recorded" is allowed to pull a target in by up to RECORDED_MARGIN — that is
    a deliberate standoff from the mechanical stop, and it is wanted: on
    2026-07-18 shoulder_lift read 0.69 deg PAST its calibrated stop, and
    returning to 2 deg inside the stop is safer than driving back into it.
    What must not happen is the 50-degree clip the policy envelope would apply.
    """
    worst_allowed = config.RECORDED_MARGIN + config.PHYSICAL_TOLERANCE
    for joint, value in REST_POSE.items():
        recorded = safety.clamp_action({joint: value}, profile="recorded")[joint]
        assert abs(recorded - value) <= worst_allowed + 1e-9, (
            f"recorded profile clipped {joint} by {abs(recorded - value):.2f}, "
            f"more than the {worst_allowed:g} standoff"
        )


def test_recorded_profile_is_far_less_restrictive_than_policy() -> None:
    """The reason the two profiles exist at all."""
    def clip(joint: str, profile: str) -> float:
        value = REST_POSE[joint]
        return abs(safety.clamp_action({joint: value}, profile=profile)[joint] - value)

    for joint in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        assert clip(joint, "recorded") < clip(joint, "policy"), (
            f"{joint}: recorded is not looser than policy"
        )
    # shoulder_lift is the extreme case: ~2.7 deg under recorded vs ~51.7 under policy.
    assert clip("shoulder_lift", "policy") > 40.0, (
        "policy no longer clips the rest pose; this test is stale"
    )


def test_gripper_keeps_full_travel_in_every_profile() -> None:
    """The gripper is a percentage; shrinking it by a degree margin would be a unit error."""
    for profile in config.LIMIT_PROFILES:
        assert config.LIMIT_PROFILES[profile]["gripper"] == (0.0, 100.0), profile


def test_rejects_bad_values_and_unknown_profile() -> None:
    for bad in ({"shoulder_pan": float("nan")}, {"shoulder_pan": float("inf")}, {"nope": 1.0}):
        try:
            safety.clamp_action(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad input {bad}")
    try:
        safety.clamp_action({"shoulder_pan": 0.0}, profile="nonsense")
    except ValueError:
        return
    raise AssertionError("accepted an unknown limit profile")


def test_clamp_does_not_mutate_its_input() -> None:
    original = {"shoulder_pan": 250.0}
    safety.clamp_action(original)
    assert original == {"shoulder_pan": 250.0}


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"  FAIL  {test.__name__}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
