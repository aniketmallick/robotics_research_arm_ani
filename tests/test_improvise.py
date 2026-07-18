"""Tests for the improvise validator — safety rule 8's "never trusted raw".

No network and no hardware: these exercise parsing and validation only, which is
the whole of what stands between a model's output and a joint target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import config, improvise  # noqa: E402

GOOD = '[{"pose": {"wrist_roll": 10}, "seconds": 1}]'


def parse(raw: str):
    return improvise.validate(improvise.extract_json(raw))


@pytest.mark.parametrize(
    "raw",
    [
        GOOD,
        f"```json\n{GOOD}\n```",                       # fenced
        f"Sure! Here you go:\n{GOOD}\nHope that helps.",  # prose either side
        '{"keyframes": ' + GOOD + "}",                  # wrapped in an object
        f"```\nA long explanation, longer than the plan itself\n```\n```json\n{GOOD}\n```",
        f"Here is the plan [see below]: {GOOD}",        # stray bracket in the prose
    ],
)
def test_accepts_the_shapes_models_actually_emit(raw):
    keyframes = parse(raw)
    assert len(keyframes) == 1
    assert keyframes[0].pose == {"wrist_roll": 10.0}
    assert keyframes[0].seconds == 1.0


@pytest.mark.parametrize(
    "raw",
    [
        "[" + ",".join(['{"pose":{"wrist_roll":1},"seconds":1}'] * (config.IMPROVISE_MAX_KEYFRAMES + 1)) + "]",
        '[{"pose":{"wrist_roll":10},"seconds":9}]',        # too slow
        '[{"pose":{"wrist_roll":10},"seconds":0.05}]',     # too fast
        '[{"pose":{"laser_cannon":10},"seconds":1}]',      # invented joint
        '[{"pose":{"wrist_roll":1},"seconds":1,"torque":999}]',  # smuggled key
        '[{"pose":{"wrist_roll":1},"seconds":true}]',      # bool is not a number
        '[{"pose":{"wrist_roll":"lots"},"seconds":1}]',    # string target
        '[{"pose":{},"seconds":1}]',                       # empty pose
        "[]",                                              # nothing to do
        '{"pose":{"wrist_roll":1},"seconds":1}',           # bare object, not a list
        "I refuse to answer.",                             # no JSON at all
        '[{"pose": {"wrist_roll": 1}, "seconds":',         # truncated
        "",                                                # empty reply
        '[{"pose": {"wrist_roll": ' + "9" * 400 + '}, "seconds": 1}]',  # OverflowError in float()
        "[" + ",".join(['{"pose":{"wrist_roll":1},"seconds":5}'] * 8) + "]",  # 40s total
    ],
)
def test_rejects_bad_plans(raw):
    """Rejection must be ImproviseError specifically — anything else escapes the
    retry loop in request_plan and the CLI handler as a raw traceback."""
    with pytest.raises(improvise.ImproviseError):
        parse(raw)


def test_out_of_range_targets_are_clamped_not_rejected():
    """A too-large target is a usable plan pulled inside the policy envelope."""
    keyframes = parse('[{"pose": {"shoulder_lift": -999}, "seconds": 1}]')
    assert keyframes[0].pose["shoulder_lift"] == config.JOINT_LIMITS["shoulder_lift"][0]


def test_improvised_targets_use_the_policy_profile_not_recorded():
    """LLM motion must get the conservative envelope, never the teleop one."""
    keyframes = parse('[{"pose": {"shoulder_lift": -105}, "seconds": 1}]')
    policy_low = config.JOINT_LIMITS["shoulder_lift"][0]
    recorded_low = config.LIMIT_PROFILES["recorded"]["shoulder_lift"][0]
    assert keyframes[0].pose["shoulder_lift"] == policy_low
    assert policy_low > recorded_low, "policy must be tighter than recorded"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_are_refused_as_improvise_errors(literal):
    """json.loads parses these literals happily, and a naive clamp passes NaN through.

    It must surface as ImproviseError specifically: anything else escapes the
    retry loop in request_plan and the CLI's handler, becoming a raw traceback.
    """
    with pytest.raises(improvise.ImproviseError):
        parse(f'[{{"pose": {{"wrist_roll": {literal}}}, "seconds": 1}}]')
