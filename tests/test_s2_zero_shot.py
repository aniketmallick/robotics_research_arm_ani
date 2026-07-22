"""Cheap, hardware-free tests for the Spike S2 zero-shot runner.

No robot, no camera, no model. A fake arm + a fake infer function that emits
out-of-range actions prove the one thing that must never break: every predicted
action is clamped to the policy envelope before any send, and observe-only sends
nothing at all. Also covers clamp-bite detection, the NaN drop path, episode
caps, the positional action map, frame assembly, trial-CSV append, and args.

Runs in either conda env (lerobot or lerobot-vla) — it needs only numpy, torch,
and armani, never the smolvla checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.s2_zero_shot import camera, clamp, run_zero_shot, smolvla_io  # noqa: E402

REST = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}
OUT_OF_RANGE = {
    "shoulder_pan": 200.0,   # policy limit 90
    "shoulder_lift": 10.0,   # in range
    "elbow_flex": 0.0,
    "wrist_flex": -999.0,    # policy limit -60
    "wrist_roll": 0.0,
    "gripper": 250.0,        # limit 100
}
POLICY_LIMITS = clamp._FALLBACK_POLICY_LIMITS


class FakeArm:
    def __init__(self) -> None:
        self.sends: list[dict[str, float]] = []

    def read_positions(self) -> dict[str, float]:
        return dict(REST)

    def send(self, action: dict[str, float]) -> dict[str, float]:
        self.sends.append(dict(action))
        return dict(action)


def fake_clock():
    """now() reads the clock; sleep(dt) advances it. Deterministic, no real time."""
    t = [0.0]
    return (lambda: t[0]), (lambda dt: t.__setitem__(0, t[0] + dt))


def stop_after(n: int):
    calls = {"i": 0}

    def stop() -> bool:
        if calls["i"] >= n:
            return True
        calls["i"] += 1
        return False

    return stop


def const_infer(action):
    return lambda state, frame, task: dict(action)


def frame_fn(step):
    return camera.synthetic_frame(step)


def _within_policy(action: dict[str, float]) -> bool:
    return all(POLICY_LIMITS[j][0] <= v <= POLICY_LIMITS[j][1] for j, v in action.items())


# --- the load-bearing safety test ---------------------------------------
def test_out_of_range_action_is_clamped_before_send():
    arm = FakeArm()
    records: list[dict] = []
    now, sleep = fake_clock()

    stats = run_zero_shot.run_episode(
        arm=arm, read_frame=frame_fn, infer_fn=const_infer(OUT_OF_RANGE),
        task="t", live=True, hz=10.0, seconds=999.0, sink=records.append,
        now=now, sleep=sleep, stop=stop_after(1),
    )

    assert stats.n_steps == 1
    assert len(arm.sends) == 1
    sent = arm.sends[0]
    # Raw values NEVER reach the arm; only clamped, in-range ones do.
    assert _within_policy(sent)
    assert sent["shoulder_pan"] == 90.0
    assert sent["wrist_flex"] == -60.0
    assert sent["gripper"] == 100.0
    # The step record keeps both raw and clamped for the audit trail.
    rec = records[0]
    assert rec["raw"]["shoulder_pan"] == 200.0
    assert rec["clamped"]["shoulder_pan"] == 90.0
    assert rec["clamp_bit"] is True
    assert set(rec["clamp_bit_joints"]) == {"shoulder_pan", "wrist_flex", "gripper"}


def test_observe_only_never_sends():
    arm = FakeArm()
    records: list[dict] = []
    now, sleep = fake_clock()
    run_zero_shot.run_episode(
        arm=arm, read_frame=frame_fn, infer_fn=const_infer(OUT_OF_RANGE),
        task="t", live=False, hz=10.0, seconds=999.0, sink=records.append,
        now=now, sleep=sleep, stop=stop_after(3),
    )
    assert arm.sends == []
    assert all(r["sent"] is None for r in records)


def test_nan_action_is_dropped_not_sent():
    arm = FakeArm()
    records: list[dict] = []
    now, sleep = fake_clock()
    bad = dict(REST, shoulder_pan=float("nan"))
    stats = run_zero_shot.run_episode(
        arm=arm, read_frame=frame_fn, infer_fn=const_infer(bad),
        task="t", live=True, hz=10.0, seconds=999.0, sink=records.append,
        now=now, sleep=sleep, stop=stop_after(2),
    )
    assert arm.sends == []  # never sent
    assert stats.invalid_steps == 2
    assert "invalid_action" in records[0]


def test_episode_capped_by_seconds():
    arm = FakeArm()
    now, sleep = fake_clock()
    # hz=8 -> period 0.125 s, exactly representable in float, so the step count is
    # deterministic (0.1 would accumulate float error and slip an extra step).
    stats = run_zero_shot.run_episode(
        arm=arm, read_frame=frame_fn, infer_fn=const_infer(REST),
        task="t", live=False, hz=8.0, seconds=1.0, sink=lambda r: None,
        now=now, sleep=sleep,
    )
    assert stats.n_steps == 8  # 1.0 s / (1/8 Hz)


def test_seconds_never_exceeds_hard_cap():
    arm = FakeArm()
    now, sleep = fake_clock()
    stats = run_zero_shot.run_episode(
        arm=arm, read_frame=frame_fn, infer_fn=const_infer(REST),
        task="t", live=False, hz=8.0, seconds=100.0, sink=lambda r: None,
        now=now, sleep=sleep,
    )
    assert stats.n_steps == int(run_zero_shot.MAX_EPISODE_SECONDS * 8)  # clamped to 30 s


def test_stop_breaks_loop():
    arm = FakeArm()
    now, sleep = fake_clock()
    stats = run_zero_shot.run_episode(
        arm=arm, read_frame=frame_fn, infer_fn=const_infer(REST),
        task="t", live=False, hz=10.0, seconds=999.0, sink=lambda r: None,
        now=now, sleep=sleep, stop=stop_after(4),
    )
    assert stats.n_steps == 4


# --- clamp module --------------------------------------------------------
def test_clamp_bite_detection():
    in_range = clamp.policy_clamp(dict(REST))
    assert in_range.bit is False and in_range.bit_joints == ()

    out = clamp.policy_clamp(dict(OUT_OF_RANGE))
    assert out.bit is True
    assert set(out.bit_joints) == {"shoulder_pan", "wrist_flex", "gripper"}
    assert _within_policy(out.clamped)


def test_clamp_rejects_nan():
    with pytest.raises(ValueError):
        clamp.policy_clamp(dict(REST, gripper=float("nan")))


def test_fallback_limits_match_armani_source():
    """The embedded fallback must equal the real policy profile, so it can never
    silently drift from safety rule 2 when the real clamp is unavailable."""
    armani_config = pytest.importorskip("armani.config")
    assert clamp._FALLBACK_POLICY_LIMITS == armani_config.JOINT_LIMITS


def test_clamp_source_reports_armani_when_available():
    pytest.importorskip("armani.safety")
    assert "armani" in clamp.clamp_source()


# --- smolvla_io pure logic (no model) -----------------------------------
def _spec(action_dim=6, state_dim=6):
    return smolvla_io.PolicySpec(
        image_keys=("observation.images.camera1", "observation.images.camera2", "observation.images.camera3"),
        state_dim=state_dim, action_dim=action_dim, chunk_size=50, device="cpu",
    )


def test_action_to_joints_positional_full():
    import numpy as np

    joints = smolvla_io.action_to_joints(np.array([1, 2, 3, 4, 5, 6], dtype=float), _spec(6))
    assert joints == {
        "shoulder_pan": 1.0, "shoulder_lift": 2.0, "elbow_flex": 3.0,
        "wrist_flex": 4.0, "wrist_roll": 5.0, "gripper": 6.0,
    }


def test_action_to_joints_truncates_when_dim_smaller():
    import numpy as np

    joints = smolvla_io.action_to_joints(np.array([1, 2, 3], dtype=float), _spec(action_dim=3))
    assert joints == {"shoulder_pan": 1.0, "shoulder_lift": 2.0, "elbow_flex": 3.0}


def test_build_frame_keys_and_shapes():
    frame = smolvla_io.build_frame(REST, camera.synthetic_frame(0), "pick it up", _spec())
    assert frame["task"] == "pick it up"
    assert tuple(frame["observation.state"].shape) == (6,)
    for key in _spec().image_keys:
        assert key in frame
        assert tuple(frame[key].shape) == (3, camera.CAMERA_HEIGHT, camera.CAMERA_WIDTH)
        assert 0.0 <= float(frame[key].min()) and float(frame[key].max()) <= 1.0


def test_synthetic_frame_varies_by_step():
    import numpy as np

    a = camera.synthetic_frame(0)
    b = camera.synthetic_frame(5)
    assert a.shape == (camera.CAMERA_HEIGHT, camera.CAMERA_WIDTH, 3)
    assert not np.array_equal(a, b)


# --- CLI / CSV -----------------------------------------------------------
def test_append_trial_row_writes_header_once(tmp_path):
    csv_path = tmp_path / "trials.csv"
    run_zero_shot.append_trial_row(csv_path, {"episode_tag": "e1", "score": 0, "task": "t"})
    run_zero_shot.append_trial_row(csv_path, {"episode_tag": "e2", "score": 2, "task": "t"})
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].split(",")[0] == "timestamp"  # header
    assert len(lines) == 3  # header + 2 rows
    assert "e1" in lines[1] and "e2" in lines[2]


def test_parser_defaults_observe_only():
    args = run_zero_shot.build_parser().parse_args([])
    assert args.live is False
    assert args.seconds == run_zero_shot.DEFAULT_SECONDS
    assert args.hz == run_zero_shot.DEFAULT_HZ
    assert args.device == "auto"
    assert args.task == run_zero_shot.DEFAULT_TASK


def test_parser_live_flag():
    args = run_zero_shot.build_parser().parse_args(["--live", "--seconds", "5", "--task", "grab the cube"])
    assert args.live is True
    assert args.seconds == 5.0
    assert args.task == "grab the cube"


def test_select_device_explicit_and_auto():
    assert run_zero_shot.select_device("cpu") == "cpu"
    assert run_zero_shot.select_device("auto") in {"mps", "cpu"}


def test_resolve_execution_dry_run_forces_observe_only():
    """The safety-critical coupling: a dry environment can NEVER drive a real arm.
    ARMANI_DRY_RUN=1 with --live must NOT connect a real arm + auto-approve."""
    parser = run_zero_shot.build_parser()
    # normal live run
    assert run_zero_shot.resolve_execution(parser.parse_args(["--live"]), False) == (False, True)
    # --no-arm forces simulated arm + observe-only
    assert run_zero_shot.resolve_execution(parser.parse_args(["--live", "--no-arm"]), False) == (True, False)
    # config DRY_RUN forces observe-only EVEN with --live (the fail-open guard)
    assert run_zero_shot.resolve_execution(parser.parse_args(["--live"]), True) == (True, False)
    # no --live at all
    assert run_zero_shot.resolve_execution(parser.parse_args([]), False)[1] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -5.0])
def test_run_episode_rejects_nonpositive_or_nonfinite_seconds(bad):
    now, sleep = fake_clock()
    with pytest.raises(ValueError):
        run_zero_shot.run_episode(
            arm=FakeArm(), read_frame=frame_fn, infer_fn=const_infer(REST),
            task="t", live=False, hz=10.0, seconds=bad, sink=lambda r: None,
            now=now, sleep=sleep,
        )


def test_main_rejects_nonfinite_seconds():
    # Returns 2 before any model load (the seconds check precedes the heavy imports).
    assert run_zero_shot.main(["--seconds", "nan", "--no-arm", "--synthetic-frame"]) == 2
