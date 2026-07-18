"""Zone assignment and the pick decision path. No camera, no arm, no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import config, eyes, gestures, pick, zones  # noqa: E402

FRAME = (640, 480)

ZONES = zones.ZoneSet(
    zones=(
        zones.Zone("z1", "front-left", (100, 300), 0),
        zones.Zone("z2", "front-centre", (320, 300), 1),
        zones.Zone("z3", "front-right", (540, 300), 2),
    ),
    frame_size=FRAME,
    created="test",
)


def detection(point, label="banana", confidence=0.9, frame_size=FRAME):
    return eyes.Detection(
        label=label, point=point, confidence=confidence, frame_size=frame_size, model="test"
    )


# --- assignment ----------------------------------------------------------


def test_object_on_a_spot_is_assigned_to_it():
    match = zones.assign_pixel((105, 305), ZONES)
    assert match.ok
    assert match.zone.id == "z1"
    assert match.distance_px < 10


def test_runner_up_is_reported():
    match = zones.assign_pixel((105, 305), ZONES)
    assert match.runner_up.id == "z2"
    assert match.margin_px > config.ASSIGNMENT_MARGIN_PX


def test_object_between_two_spots_is_ambiguous_not_resolved():
    """The whole point of the margin: a coin-flip must not read as an answer."""
    match = zones.assign_pixel((210, 300), ZONES)  # exactly between z1 and z2
    assert match.ambiguous
    assert not match.ok
    assert not match
    # The nearest zone is still reported, so stage 6 can name both candidates.
    assert match.zone is not None
    assert match.runner_up is not None
    assert match.reason


def test_object_far_from_every_spot_is_refused():
    match = zones.assign_pixel((320, 20), ZONES)
    assert match.zone is None
    assert not match.ok
    assert "not on a marked spot" in match.reason


def test_no_zones_defined_refuses():
    match = zones.assign_pixel((100, 300), zones.ZoneSet((), FRAME, "test"))
    assert match.zone is None


def test_a_single_zone_is_never_ambiguous():
    """With nothing to be confused with, there is no ambiguity to report."""
    solo = zones.ZoneSet((zones.Zone("z1", "only", (320, 240), 0),), FRAME, "test")
    match = zones.assign_pixel((325, 245), solo)
    assert match.ok
    assert not match.ambiguous
    assert match.runner_up is None


def test_assignment_is_by_distance_not_order():
    assert zones.assign_pixel((535, 295), ZONES).zone.id == "z3"
    assert zones.assign_pixel((315, 295), ZONES).zone.id == "z2"


def test_detection_from_a_different_frame_size_is_refused():
    """Pixels only mean something at the size they were measured at."""
    match = zones.assign_zone(detection((105, 305), frame_size=(1280, 960)), ZONES)
    assert match.zone is None
    assert "1280x960" in match.reason


def test_assign_zone_accepts_a_detection():
    match = zones.assign_zone(detection((105, 305)), ZONES)
    assert match.ok and match.zone.id == "z1"


def test_match_serialises_for_the_decision_log():
    payload = zones.assign_pixel((105, 305), ZONES).as_log()
    json.dumps(payload)  # must not raise: inf/nan would break the log
    assert payload["zone"] == "z1"


def test_far_match_serialises_without_infinities():
    payload = zones.assign_pixel((320, 20), ZONES).as_log()
    json.dumps(payload, allow_nan=False)


# --- zones.json round trip ----------------------------------------------


def write_zones(tmp_path: Path, entries, frame_size=FRAME) -> Path:
    target = tmp_path / "zones.json"
    target.write_text(json.dumps({
        "created": "test",
        "frame_size": list(frame_size),
        "zones": entries,
    }))
    return target


def test_load_round_trip(tmp_path):
    path = write_zones(tmp_path, [
        {"id": "z1", "label": "left", "pixel_center": [100, 300], "pick_episode": 0},
        {"id": "z2", "label": "right", "pixel_center": [540, 300], "pick_episode": 1},
    ])
    loaded = zones.load_zones(path)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded.by_id("z2").pick_episode == 1
    assert loaded.by_id("nope") is None


def test_missing_file_is_none(tmp_path):
    assert zones.load_zones(tmp_path / "nope.json") is None


def test_garbage_is_none_not_an_exception(tmp_path):
    target = tmp_path / "zones.json"
    target.write_text("{ not json")
    assert zones.load_zones(target) is None


def test_missing_field_is_none(tmp_path):
    path = write_zones(tmp_path, [{"id": "z1", "label": "left", "pixel_center": [1, 2]}])
    assert zones.load_zones(path) is None


def test_duplicate_ids_are_refused(tmp_path):
    """by_id would silently return the first, and picks would go to the wrong spot."""
    path = write_zones(tmp_path, [
        {"id": "z1", "label": "left", "pixel_center": [100, 300], "pick_episode": 0},
        {"id": "z1", "label": "right", "pixel_center": [540, 300], "pick_episode": 1},
    ])
    assert zones.load_zones(path) is None


def test_empty_zone_list_is_none(tmp_path):
    assert zones.load_zones(write_zones(tmp_path, [])) is None


# --- pick decisions (no motion) -----------------------------------------


class FakeArm:
    """Records commands. Any send at all is a failure in these tests."""

    label = "fake arm"

    def __init__(self):
        self.pose = {j: 0.0 for j in config.JOINTS}
        self.pose[config.GRIPPER_JOINT] = 40.0
        self.sent = []

    def read_positions(self):
        return dict(self.pose)

    def send(self, action):
        # Applies the action like a real arm would, so a test can ask where the
        # arm ENDED UP and not just what was commanded.
        self.sent.append(dict(action))
        self.pose.update(action)
        return dict(action)

    def disconnect(self):
        pass

    def disable_torque(self):
        pass


@pytest.fixture
def with_zones(tmp_path, monkeypatch):
    path = write_zones(tmp_path, [
        {"id": "z1", "label": "front-left", "pixel_center": [100, 300], "pick_episode": 0},
        {"id": "z2", "label": "front-centre", "pixel_center": [320, 300], "pick_episode": 1},
    ])
    monkeypatch.setattr(config, "ZONES_PATH", path)
    return path


def test_unseen_object_refuses_and_does_not_move(with_zones, monkeypatch):
    monkeypatch.setattr(eyes, "locate", lambda *a, **k: None)
    arm = FakeArm()
    result = pick.pick_object(arm, "banana", verify=False)
    assert not result
    assert not result.seen
    assert not result.moved
    assert arm.sent == []
    assert "cannot see" in result.reason


def test_ambiguous_object_refuses_and_does_not_move(with_zones, monkeypatch):
    monkeypatch.setattr(eyes, "locate", lambda *a, **k: detection((210, 300)))
    arm = FakeArm()
    result = pick.pick_object(arm, "banana", verify=False)
    assert not result
    assert result.seen
    assert result.ambiguous
    assert not result.moved
    assert arm.sent == []
    assert len(result.candidate_zones) == 2


def test_object_off_the_spots_refuses_and_does_not_move(with_zones, monkeypatch):
    monkeypatch.setattr(eyes, "locate", lambda *a, **k: detection((320, 20)))
    arm = FakeArm()
    result = pick.pick_object(arm, "banana", verify=False)
    assert not result
    assert result.seen
    assert not result.ambiguous
    assert not result.moved
    assert arm.sent == []


def test_missing_macro_refuses_and_does_not_move(with_zones, monkeypatch):
    """Zones defined but nothing recorded yet — the most likely demo-day state."""
    monkeypatch.setattr(eyes, "locate", lambda *a, **k: detection((105, 305)))
    # Explicit rather than relying on the dataset being absent: a machine that
    # HAS recorded picks would otherwise silently skip the branch under test.
    monkeypatch.setattr(gestures, "episode_count", lambda root=None: 0)
    arm = FakeArm()
    result = pick.pick_object(arm, "banana", verify=False)
    assert not result
    assert not result.moved
    assert arm.sent == []
    assert "no pick macro" in result.reason


def test_vision_failure_refuses(with_zones, monkeypatch):
    def boom(*a, **k):
        raise eyes.EyesError("no camera")

    monkeypatch.setattr(eyes, "locate", boom)
    arm = FakeArm()
    result = pick.pick_object(arm, "banana", verify=False)
    assert not result
    assert not result.moved
    assert "vision unavailable" in result.reason


def test_no_zones_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ZONES_PATH", tmp_path / "absent.json")
    arm = FakeArm()
    result = pick.pick_object(arm, "banana", verify=False)
    assert not result
    assert not result.moved
    assert "no zones defined" in result.reason


def test_empty_object_name_is_rejected():
    with pytest.raises(ValueError):
        pick.pick_object(FakeArm(), "   ")


def test_result_serialises_for_the_decision_log(with_zones, monkeypatch):
    monkeypatch.setattr(eyes, "locate", lambda *a, **k: detection((210, 300)))
    result = pick.pick_object(FakeArm(), "banana", verify=False)
    json.dumps(result.as_log(), allow_nan=False)


def test_a_lone_zone_produces_a_serialisable_result(tmp_path, monkeypatch):
    """One zone means an infinite margin, and json.dumps writes inf as the
    literal `Infinity` — which is not valid JSON and breaks the log reader."""
    path = write_zones(tmp_path, [
        {"id": "z1", "label": "only", "pixel_center": [320, 300], "pick_episode": 0},
    ])
    monkeypatch.setattr(config, "ZONES_PATH", path)
    monkeypatch.setattr(eyes, "locate", lambda *a, **k: detection((322, 302)))
    monkeypatch.setattr(gestures, "episode_count", lambda root=None: 0)
    result = pick.pick_object(FakeArm(), "banana", verify=False)
    json.dumps(result.as_log(), allow_nan=False)


def test_result_can_be_splatted_into_the_decision_log(with_zones, monkeypatch):
    """log_event(kind, **result.as_log()) is how pick.py logs. Passing any field
    alongside the splat is a duplicate-keyword TypeError — which bit the happy
    path, one line before the arm moves."""
    from armani.logutil import log_event

    monkeypatch.setattr(eyes, "locate", lambda *a, **k: detection((105, 305)))
    monkeypatch.setattr(gestures, "episode_count", lambda root=None: 0)
    result = pick.pick_object(FakeArm(), "banana", verify=False)
    log_event("test_pick_start", **result.as_log())


# --- macro replay (the demo's actual motion path) ------------------------


def synthetic_pick_macro() -> gestures.Gesture:
    """A plausible pick: reach out, close the gripper, lift, come back holding.

    Frame-to-frame deltas stay under config.MAX_FRAME_DELTA so the loader's
    playability check passes, as a real recording's would.
    """
    frames = []
    for step in range(10):  # reach out, gripper open
        frames.append({
            "shoulder_pan": 0.0, "shoulder_lift": -2.0 * step, "elbow_flex": 2.0 * step,
            "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 60.0,
        })
    for step in range(8):  # close on the object
        frames.append({
            "shoulder_pan": 0.0, "shoulder_lift": -18.0, "elbow_flex": 18.0,
            "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 60.0 - 5.0 * step,
        })
    for step in range(10):  # lift and return, still holding
        frames.append({
            "shoulder_pan": 0.0, "shoulder_lift": -18.0 + 1.8 * step,
            "elbow_flex": 18.0 - 1.8 * step, "wrist_flex": 0.0,
            "wrist_roll": 0.0, "gripper": 20.0,
        })
    return gestures.Gesture(name="pick:test", episode=0, fps=30, frames=tuple(frames))


def test_pick_macro_replays_every_frame():
    macro = synthetic_pick_macro()
    arm = FakeArm()
    assert gestures.play_macro(arm, macro, return_home=False, kind="pick")
    # Pre-positioning adds interpolated sends on top of the recorded frames.
    assert len(arm.sent) >= len(macro.frames)


def test_pick_macro_ends_holding_and_never_opens_the_gripper_afterwards():
    """The load-bearing property: home() commands the gripper too, so an
    auto-home after a grasp would open the jaws and drop the object."""
    macro = synthetic_pick_macro()
    arm = FakeArm()
    gestures.play_macro(arm, macro, return_home=False, kind="pick")
    assert arm.sent[-1]["gripper"] == pytest.approx(macro.last["gripper"])
    assert arm.pose["gripper"] == pytest.approx(20.0)


def test_pick_replay_stops_on_the_kill_switch():
    from armani import safety

    macro = synthetic_pick_macro()
    arm = FakeArm()
    safety.request_stop("test")
    try:
        completed = gestures.play_macro(arm, macro, return_home=False, kind="pick")
    finally:
        safety.clear_stop()
    assert completed is False


def test_play_gesture_still_delegates_to_play_macro(monkeypatch):
    """Stage 2/3 entry point must keep working after the generalisation."""
    seen = {}

    def fake_play_macro(arm, macro, return_home=True, kind="gesture"):
        seen.update(name=macro.name, kind=kind, return_home=return_home)
        return True

    monkeypatch.setattr(gestures, "load_gesture", lambda name: synthetic_pick_macro())
    monkeypatch.setattr(gestures, "play_macro", fake_play_macro)
    gestures.play_gesture(FakeArm(), "bow")
    assert seen["kind"] == "gesture"
    assert seen["return_home"] is True


# --- verification hook ---------------------------------------------------


def test_gripper_closed_on_nothing_reads_as_empty(monkeypatch):
    arm = FakeArm()
    arm.pose[config.GRIPPER_JOINT] = 0.5
    outcome = pick.verify_held(arm, "banana", save_frame=False)
    assert outcome.held_guess is False
    assert outcome.gripper_percent == pytest.approx(0.5)


def test_gripper_stopped_by_an_object_reads_as_held(monkeypatch):
    arm = FakeArm()
    arm.pose[config.GRIPPER_JOINT] = 30.0
    outcome = pick.verify_held(arm, "banana", save_frame=False)
    assert outcome.held_guess is True


def test_verify_never_raises_when_the_arm_misbehaves():
    """The arm is holding an object; verification must not throw mid-grasp."""

    class BrokenArm:
        def read_positions(self):
            raise RuntimeError("bus error")

    outcome = pick.verify_held(BrokenArm(), "banana", save_frame=False)
    assert outcome.held_guess is None
    assert "could not read the gripper" in outcome.reason


def test_verify_states_that_the_vlm_check_is_not_implemented():
    """Honesty guard: G5 is not done, and the log must not imply it is."""
    arm = FakeArm()
    outcome = pick.verify_held(arm, "banana", save_frame=False)
    assert "not implemented" in outcome.reason
