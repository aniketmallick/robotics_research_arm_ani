"""The trust-gate pipeline. No camera, no network, no arm.

Every gate is exercised with injected vision and injected clarify/approve
callables, so the ordering, the fail-closed behaviour and — above all — the
Python-enforced approval deadline are testable on a laptop.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import config, eyes, gates, pick, zones  # noqa: E402

FRAME = (640, 480)

# 200 px apart, so the midpoint is 100 px from each: inside
# ZONE_MAX_DISTANCE_PX (on the table) but with a zero margin (between two
# spots). Spread them further and the midpoint stops being ambiguous and starts
# being "not on any spot" — a different gate.
ZONES = zones.ZoneSet(
    zones=(
        zones.Zone("z1", "front-left", (200, 300), 0),
        zones.Zone("z2", "front-right", (400, 300), 1),
    ),
    frame_size=FRAME,
    created="test",
)

ON_Z1 = (202, 302)
ON_Z2 = (402, 302)
BETWEEN = (300, 300)
OFF_THE_TABLE = (320, 10)


def detection(point=ON_Z1, confidence=0.9, candidates=1, label="red block"):
    return eyes.Detection(
        label=label, point=point, confidence=confidence,
        frame_size=FRAME, model="test", candidates=candidates,
    )


class FakeArm:
    """Counts commands. Zero sends is how a stand-down is proven."""

    label = "fake arm"

    def __init__(self, gripper=30.0):
        self.pose = {j: 0.0 for j in config.JOINTS}
        self.pose[config.GRIPPER_JOINT] = gripper
        self.sent = []

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


@pytest.fixture
def rig(monkeypatch):
    """Zones loaded, a macro available for every zone, vision injectable."""
    monkeypatch.setattr(zones, "load_zones", lambda *a, **k: ZONES)
    monkeypatch.setattr(pick, "macro_available", lambda zone: True)
    monkeypatch.setattr(gates.zones, "load_zones", lambda *a, **k: ZONES)
    monkeypatch.setattr(gates.pick, "macro_available", lambda zone: True)
    return ZONES


def see(monkeypatch, det):
    monkeypatch.setattr(gates.eyes, "locate", lambda *a, **k: det)
    monkeypatch.setattr(gates.eyes, "capture_frame", lambda *a, **k: object())


def verifies(monkeypatch, held=True):
    monkeypatch.setattr(
        gates.pick, "verify_held",
        lambda obj, gripper, frame=None, use_vlm=True: pick.VerifyResult(
            gripper_percent=gripper, held=held, reason="test"
        ),
    )


def performed(outcomes: list):
    """A perform() that records the zones it was asked to run."""

    def perform(zone):
        outcomes.append(zone.id)
        return gates.PerformOutcome(completed=True, gripper_percent=30.0)

    return perform


def never_called(*args, **kwargs):
    raise AssertionError("this callable must not be reached")


# --- the confidence number ----------------------------------------------


def test_confidence_is_vision_tempered_by_assignment_clarity():
    wide = gates.confidence_for(0.9, config.CONF_CLEAR_MARGIN_PX)
    tight = gates.confidence_for(0.9, 0.0)
    assert wide == pytest.approx(0.9)
    assert tight == pytest.approx(0.9 * config.CONF_ASSIGNMENT_FLOOR)
    assert tight < wide


def test_confidence_never_exceeds_vision_confidence():
    for margin in (0.0, 30.0, 120.0, 10_000.0, float("inf")):
        assert gates.confidence_for(0.8, margin) <= 0.8 + 1e-9


def test_confidence_is_bounded_and_survives_nonsense():
    assert gates.confidence_for(0.0, 50.0) == 0.0
    assert 0.0 <= gates.confidence_for(1.0, float("nan")) <= 1.0
    assert 0.0 <= gates.confidence_for(1.0, -50.0) <= 1.0


def test_a_lone_zone_is_maximally_clear():
    """No runner-up means an infinite margin, which must not break the maths."""
    assert gates.confidence_for(0.9, float("inf")) == pytest.approx(0.9)


# --- G1 seen -------------------------------------------------------------


def test_g1_unseen_stops_and_never_moves(rig, monkeypatch):
    see(monkeypatch, None)
    arm = FakeArm()
    result = gates.run_gated_pick(
        arm, "red block", clarify=never_called, approve=never_called,
        perform=never_called, verify_vlm=False,
    )
    assert not result
    assert result.stopped_at == gates.G1_SEEN
    assert not result.moved
    assert arm.sent == []


def test_g1_vision_failure_stops(rig, monkeypatch):
    def boom(*a, **k):
        raise eyes.EyesError("camera gone")

    monkeypatch.setattr(gates.eyes, "locate", boom)
    monkeypatch.setattr(gates.eyes, "capture_frame", lambda *a, **k: object())
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=never_called, verify_vlm=False,
    )
    assert result.stopped_at == gates.G1_SEEN
    assert not result.moved


# --- G2 ambiguous --------------------------------------------------------


def test_g2_clear_assignment_does_not_ask(rig, monkeypatch):
    see(monkeypatch, detection())
    verifies(monkeypatch)
    ran: list[str] = []
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=performed(ran), verify_vlm=False,
    )
    assert result.ok
    assert not result.clarified
    assert ran == ["z1"]


def test_g2_between_two_spots_asks_then_resolves(rig, monkeypatch):
    see(monkeypatch, detection(point=BETWEEN))
    verifies(monkeypatch)
    asked: list[str] = []
    ran: list[str] = []

    def clarify(question, options):
        asked.append(question)
        return "the front-right one"

    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=clarify, approve=lambda p, t: True,
        perform=performed(ran), verify_vlm=False,
    )
    assert result.ok
    assert result.clarified
    assert ran == ["z2"], "the human's answer must decide the zone"
    assert asked and "which one" in asked[0].lower()


def test_g2_unresolvable_answer_stands_down(rig, monkeypatch):
    """'um, that one' must not become a coin flip."""
    see(monkeypatch, detection(point=BETWEEN))
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=lambda q, o: "um, that one",
        approve=never_called, perform=never_called, verify_vlm=False,
    )
    assert not result
    assert result.stopped_at == gates.G2_AMBIGUOUS
    assert not result.moved


def test_g2_no_answer_stands_down(rig, monkeypatch):
    see(monkeypatch, detection(point=BETWEEN))
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=lambda q, o: None,
        approve=never_called, perform=never_called, verify_vlm=False,
    )
    assert result.stopped_at == gates.G2_AMBIGUOUS
    assert not result.moved


def test_g2_answer_naming_both_options_is_refused(rig, monkeypatch):
    see(monkeypatch, detection(point=BETWEEN))
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=lambda q, o: "front-left front-right",
        approve=never_called, perform=never_called, verify_vlm=False,
    )
    assert result.stopped_at == gates.G2_AMBIGUOUS


def test_g2_multiple_candidates_asks_even_when_geometrically_clear(rig, monkeypatch):
    """Two red blocks on two spots: the winner is clearly on z1, but 'which
    one?' is still the right question."""
    see(monkeypatch, detection(candidates=2))
    monkeypatch.setattr(
        gates.eyes, "list_visible",
        lambda names, frame=None: [detection(point=ON_Z1), detection(point=ON_Z2)],
    )
    verifies(monkeypatch)
    ran: list[str] = []
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=lambda q, o: "front-right",
        approve=lambda p, t: True, perform=performed(ran), verify_vlm=False,
    )
    assert result.ok
    assert result.clarified
    assert ran == ["z2"]


def test_g2_object_not_on_any_spot_stops(rig, monkeypatch):
    see(monkeypatch, detection(point=OFF_THE_TABLE))
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=never_called, verify_vlm=False,
    )
    assert result.stopped_at == gates.G2_AMBIGUOUS
    assert not result.moved


# --- G3 reachable --------------------------------------------------------


def test_g3_missing_macro_stops(rig, monkeypatch):
    see(monkeypatch, detection())
    monkeypatch.setattr(gates.pick, "macro_available", lambda zone: False)
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=never_called, verify_vlm=False,
    )
    assert result.stopped_at == gates.G3_REACHABLE
    assert not result.moved


# --- G4 confidence, approval, timeout -----------------------------------


def test_g4_high_confidence_proceeds_without_asking(rig, monkeypatch):
    see(monkeypatch, detection(confidence=0.95))
    verifies(monkeypatch)
    ran: list[str] = []
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=performed(ran), verify_vlm=False,
    )
    assert result.ok
    assert not result.approval_required
    assert result.confidence >= config.CONF_APPROVAL
    assert ran == ["z1"]


def test_g4_low_confidence_asks_and_proceeds_when_approved(rig, monkeypatch):
    see(monkeypatch, detection(confidence=0.5))
    verifies(monkeypatch)
    ran: list[str] = []
    prompts: list[str] = []

    def approve(prompt, timeout_s):
        prompts.append(prompt)
        return True

    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=approve,
        perform=performed(ran), verify_vlm=False,
    )
    assert result.ok
    assert result.approval_required and result.approved
    assert ran == ["z1"]
    assert "%" in prompts[0], "the human must be told the number"


def test_g4_refusal_stands_down_with_zero_sends(rig, monkeypatch):
    see(monkeypatch, detection(confidence=0.5))
    arm = FakeArm()
    result = gates.run_gated_pick(
        arm, "red block", clarify=never_called, approve=lambda p, t: False,
        perform=never_called, verify_vlm=False,
    )
    assert result.stopped_at == gates.G4_CONFIDENCE
    assert result.approved is False
    assert not result.moved
    assert arm.sent == []


def test_g4_silence_stands_down_with_zero_sends(rig, monkeypatch):
    """THE gate. No answer within the deadline means the arm does not move."""
    see(monkeypatch, detection(confidence=0.5))
    arm = FakeArm()

    def never_answers(prompt, timeout_s):
        time.sleep(30)  # far past the deadline
        return True  # ...and would have said yes, too late

    started = time.perf_counter()
    result = gates.run_gated_pick(
        arm, "red block", clarify=never_called, approve=never_answers,
        perform=never_called, verify_vlm=False, approval_timeout_s=0.3,
    )
    elapsed = time.perf_counter() - started

    assert result.stopped_at == gates.G4_CONFIDENCE
    assert result.timed_out
    assert result.approved is False
    assert not result.moved
    assert arm.sent == [], "a timed-out approval must command nothing"
    assert elapsed < 5, "the deadline must be enforced by gates.py, not waited out"


def test_g4_a_late_yes_cannot_resurrect_the_pick(rig, monkeypatch):
    """The callable answers after the deadline. The pick is already discarded."""
    see(monkeypatch, detection(confidence=0.5))
    arm = FakeArm()
    ran: list[str] = []

    def slow_yes(prompt, timeout_s):
        time.sleep(0.6)
        return True

    result = gates.run_gated_pick(
        arm, "red block", clarify=never_called, approve=slow_yes,
        perform=performed(ran), verify_vlm=False, approval_timeout_s=0.2,
    )
    time.sleep(0.8)  # let the late answer land
    assert result.timed_out
    assert ran == [], "nothing may run after a stand-down"
    assert arm.sent == []


def test_g4_a_raising_callable_stands_down(rig, monkeypatch):
    """An injected callable that explodes must not become an approval."""
    see(monkeypatch, detection(confidence=0.5))

    def boom(prompt, timeout_s):
        raise RuntimeError("voice handler died")

    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=boom,
        perform=never_called, verify_vlm=False, approval_timeout_s=0.5,
    )
    assert result.stopped_at == gates.G4_CONFIDENCE
    assert not result.moved


@pytest.mark.parametrize("truthy", [None, 0, "", False])
def test_g4_only_a_real_yes_approves(rig, monkeypatch, truthy):
    """Fail-closed: anything falsy is a refusal, not an approval."""
    see(monkeypatch, detection(confidence=0.5))
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=lambda p, t: truthy,
        perform=never_called, verify_vlm=False,
    )
    assert not result.ok
    assert result.stopped_at == gates.G4_CONFIDENCE


# --- G5 verify -----------------------------------------------------------


def test_g5_failure_is_reported_honestly(rig, monkeypatch):
    """The macro ran; the object is not held. That is not a success."""
    see(monkeypatch, detection())
    verifies(monkeypatch, held=False)
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=performed([]), verify_vlm=False,
    )
    assert not result.ok
    assert result.moved, "it did move — refusing to admit that would be a lie"
    assert result.verified is False
    assert result.stopped_at == gates.G5_VERIFY
    # Place mode (the default): failure is "couldn't move it", not "didn't grab it".
    assert "couldn't move" in result.reason


def test_g5_wording_follows_the_pick_mode(rig, monkeypatch):
    """A pick-and-place macro that worked must not announce a failed grasp."""
    see(monkeypatch, detection())
    verifies(monkeypatch, held=True)

    monkeypatch.setattr(config, "PICK_MODE", "place")
    placed = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=performed([]), verify_vlm=False,
    )
    assert "tray" in placed.speak()

    monkeypatch.setattr(config, "PICK_MODE", "hold")
    holding = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=performed([]), verify_vlm=False,
    )
    assert "got it" in holding.speak()


def test_g5_success(rig, monkeypatch):
    see(monkeypatch, detection())
    verifies(monkeypatch, held=True)
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=performed([]), verify_vlm=False,
    )
    assert result.ok and result.verified and result.moved
    assert result.stopped_at is None


def test_an_interrupted_macro_is_not_a_success(rig, monkeypatch):
    see(monkeypatch, detection())
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=lambda zone: gates.PerformOutcome(completed=False, detail="kill switch"),
        verify_vlm=False,
    )
    assert not result.ok
    assert result.moved
    assert result.verified is None


# --- the audit trail -----------------------------------------------------


def test_every_run_records_gates_in_order(rig, monkeypatch):
    see(monkeypatch, detection())
    verifies(monkeypatch)
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=never_called,
        perform=performed([]), verify_vlm=False,
    )
    names = [r.gate for r in result.records]
    assert names == [gates.G1_SEEN, gates.G2_AMBIGUOUS, gates.G3_REACHABLE,
                     gates.G4_CONFIDENCE, gates.G5_VERIFY]


def test_a_stopped_run_records_up_to_and_including_the_gate_that_stopped(rig, monkeypatch):
    see(monkeypatch, detection(confidence=0.5))
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=never_called, approve=lambda p, t: False,
        perform=never_called, verify_vlm=False,
    )
    names = [r.gate for r in result.records]
    assert names[-1] == gates.G4_CONFIDENCE
    assert not result.records[-1].passed
    assert gates.G5_VERIFY not in names


def test_result_serialises_for_the_decision_log(rig, monkeypatch):
    """Stage 7 renders exactly this. inf/nan would break the reader."""
    see(monkeypatch, detection(point=BETWEEN))
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=lambda q, o: None,
        approve=never_called, perform=never_called, verify_vlm=False,
    )
    json.dumps(result.as_log(), allow_nan=False)


def test_empty_object_name_is_rejected(rig):
    with pytest.raises(ValueError):
        gates.run_gated_pick(FakeArm(), "  ", clarify=never_called, approve=never_called)


# --- the agent's return-to-model dialogue pattern ------------------------
#
# The voice path must reach the SAME verdicts as the console path. In
# particular the 10-second stand-down has to survive being routed through a
# model that may simply never call back.


def _pending(monkeypatch, object_name="red block", **kw):
    from armani import agent, motion

    worker = agent.MotionWorker(motion.DryRunArm(), motion_enabled=True)
    return agent.PendingPick(worker, object_name, verify_vlm=False, **kw)


# --- G2 resolved by a named spot (the stateless re-call) -----------------


def test_a_named_spot_skips_the_question_entirely(rig, monkeypatch):
    see(monkeypatch, detection(point=BETWEEN))
    verifies(monkeypatch)
    ran: list[str] = []
    result = gates.run_gated_pick(
        FakeArm(), "red block", spot="front-right", clarify=never_called,
        approve=lambda p, t: True, perform=performed(ran), verify_vlm=False,
    )
    assert result.ok, result.reason
    assert result.clarified
    assert ran == ["z2"]


def test_a_spot_that_is_not_a_zone_is_refused(rig, monkeypatch):
    see(monkeypatch, detection(point=BETWEEN))
    ran: list[str] = []
    result = gates.run_gated_pick(
        FakeArm(), "red block", spot="nowhere", clarify=never_called,
        approve=never_called, perform=performed(ran), verify_vlm=False,
    )
    assert not result.ok
    assert result.stopped_at == gates.G2_AMBIGUOUS
    assert not result.moved
    assert ran == []


def test_a_spot_the_object_is_not_on_is_refused(rig, monkeypatch):
    """THE INVARIANT. The spot is text relayed by the model, so naming an empty
    spot must not make the arm grasp at it."""
    see(monkeypatch, detection(point=ON_Z1))   # the block is on front-LEFT
    arm = FakeArm()
    ran: list[str] = []
    result = gates.run_gated_pick(
        arm, "red block", spot="front-right", clarify=never_called,
        approve=never_called, perform=performed(ran), verify_vlm=False,
    )
    assert not result.ok
    assert result.stopped_at == gates.G2_AMBIGUOUS
    assert "can't see a red block on front-right" in result.reason
    assert ran == []
    assert arm.sent == []


def test_a_second_instance_on_the_named_spot_is_found(rig, monkeypatch):
    """Two blocks, vision preferred the left one, the human said right. Looking
    again finds it — otherwise 'pick the right one' fails on a coin flip."""
    see(monkeypatch, detection(point=ON_Z1, candidates=2))
    monkeypatch.setattr(
        gates.eyes, "list_visible",
        lambda names, frame=None: [detection(point=ON_Z1), detection(point=ON_Z2)],
    )
    verifies(monkeypatch)
    ran: list[str] = []
    result = gates.run_gated_pick(
        FakeArm(), "red block", spot="front-right", clarify=never_called,
        approve=lambda p, t: True, perform=performed(ran), verify_vlm=False,
    )
    assert result.ok, result.reason
    assert ran == ["z2"]


def test_an_ambiguous_pick_without_a_spot_asks_and_stops(rig, monkeypatch):
    see(monkeypatch, detection(point=BETWEEN))
    ran: list[str] = []
    result = gates.run_gated_pick(
        FakeArm(), "red block", clarify=None, approve=never_called,
        perform=performed(ran), verify_vlm=False,
    )
    assert result.needs_clarification
    assert not result.ok
    assert not result.moved
    assert ran == []
    assert set(result.clarify_options) == {"front-left", "front-right"}
    assert result.speak() == result.clarify_question


def test_the_question_is_not_reported_as_a_plain_refusal(rig, monkeypatch):
    """The dashboard and the model must be able to tell 'ask them' from 'no'."""
    see(monkeypatch, detection(point=BETWEEN))
    asked = gates.run_gated_pick(
        FakeArm(), "red block", clarify=None, approve=never_called,
        perform=never_called, verify_vlm=False,
    )
    refused = gates.run_gated_pick(
        FakeArm(), "red block", spot="nowhere", clarify=None, approve=never_called,
        perform=never_called, verify_vlm=False,
    )
    assert asked.needs_clarification and not refused.needs_clarification


def _stub_macro(monkeypatch):
    """Make the macro instant, so these tests exercise gates and not motion.

    Patching ``gates.pick`` is enough: it is the same module object PendingPick
    imports, so both paths see the stub.
    """
    from armani import gestures

    macro = gestures.Gesture(
        name="pick:test", episode=0, fps=30, frames=({j: 0.0 for j in config.JOINTS},)
    )
    monkeypatch.setattr(gates.pick, "load_pick", lambda zone: macro)
    monkeypatch.setattr(gates.pick, "play_pick", lambda arm, zone: True)
    monkeypatch.setattr(gates.pick, "read_gripper", lambda arm: 30.0)
    return macro


def test_agent_pick_asks_which_and_leaves_nothing_pending(rig, monkeypatch):
    """The old flow blocked on the model completing an answer_pick round trip.
    It did not reliably complete, so the clarification timed out and stood the
    pick down before the human's answer could count. Now it just asks."""
    see(monkeypatch, detection(point=BETWEEN))
    verifies(monkeypatch)
    _stub_macro(monkeypatch)

    pending = _pending(monkeypatch)
    pending.worker.start()
    pending.start()

    event = pending.next_event(timeout=5)
    assert event["status"] == "need_clarification"
    assert "which one" in event["question"].lower()
    assert set(event["options"]) == {"front-left", "front-right"}

    # Nothing is left running, so a model that never follows up strands nothing.
    time.sleep(0.3)
    assert pending.finished
    assert pending.worker.arm.sends == 0
    pending.worker.shutdown()


def test_agent_pick_with_a_spot_resolves_in_one_call(rig, monkeypatch):
    """The second call carries the human's answer and completes with no round trip."""
    see(monkeypatch, detection(point=BETWEEN))
    verifies(monkeypatch)
    _stub_macro(monkeypatch)

    pending = _pending(monkeypatch, spot="front-right")
    pending.worker.start()
    pending.start()

    event = pending.next_event(timeout=10)
    assert event["status"] in ("started", "done"), event
    pending.worker.shutdown()


def test_agent_pick_stands_down_when_the_model_never_calls_back(rig, monkeypatch):
    """THE invariant, through the voice path: silence still means no motion."""
    see(monkeypatch, detection(confidence=0.5))
    verifies(monkeypatch)
    _stub_macro(monkeypatch)
    monkeypatch.setattr(config, "APPROVAL_TIMEOUT_S", 0.4)

    pending = _pending(monkeypatch)
    pending.worker.start()
    pending.start()

    asked = pending.next_event(timeout=5)
    assert asked["status"] == "needs_approval"
    assert asked["confidence"] > 0, "the model must be handed the real number"

    # The model never calls approve_pick. Nobody says anything.
    final = pending.next_event(timeout=10)
    assert final["status"] == "refused"
    assert final["stopped_at"] == gates.G4_CONFIDENCE
    assert final["moved"] is False
    assert pending.worker.arm.sends == 0, "a stand-down must command nothing"
    pending.worker.shutdown()


def test_agent_a_late_approval_is_rejected(rig, monkeypatch):
    """approve_pick arriving after the stand-down must not start anything."""
    see(monkeypatch, detection(confidence=0.5))
    verifies(monkeypatch)
    _stub_macro(monkeypatch)
    monkeypatch.setattr(config, "APPROVAL_TIMEOUT_S", 0.3)

    pending = _pending(monkeypatch)
    pending.worker.start()
    pending.start()
    assert pending.next_event(timeout=5)["status"] == "needs_approval"

    final = pending.next_event(timeout=10)
    assert final["status"] == "refused"

    # The human finally says yes, far too late.
    time.sleep(0.2)
    assert pending.finished
    pending.reply(True)
    time.sleep(0.3)
    assert pending.worker.arm.sends == 0, "a late yes must not resurrect a discarded pick"
    pending.worker.shutdown()


def test_agent_pick_reports_a_refusal_verbatim(rig, monkeypatch):
    see(monkeypatch, None)
    pending = _pending(monkeypatch)
    pending.worker.start()
    pending.start()
    event = pending.next_event(timeout=5)
    assert event["status"] == "refused"
    assert event["stopped_at"] == gates.G1_SEEN
    assert not event["moved"]
    pending.worker.shutdown()


# --- the word matcher ----------------------------------------------------

OPTIONS = [zones.Zone("z1", "front-left", (0, 0), 0), zones.Zone("z2", "back-right", (0, 0), 1)]


@pytest.mark.parametrize("answer,expected", [
    ("front-left", "z1"),
    ("the front left one please", "z1"),
    ("back-right", "z2"),
    ("BACK RIGHT", "z2"),
    ("the first one", "z1"),
    ("the second", "z2"),
    ("z2", "z2"),
])
def test_answers_that_should_resolve(answer, expected):
    assert gates._match_zone_by_words(answer, OPTIONS).id == expected


@pytest.mark.parametrize("answer", ["", "   ", "um", "that one", "the blue one", "yes"])
def test_answers_that_must_not_resolve(answer):
    """Refusing to understand is correct; guessing is not."""
    assert gates._match_zone_by_words(answer, OPTIONS) is None
