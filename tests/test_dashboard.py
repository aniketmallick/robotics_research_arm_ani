"""The dashboard's state layer and the demo-hardening lines. No server, no camera."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import agent, config, dashboard  # noqa: E402


def write_log(tmp_path: Path, records: list[dict]) -> dashboard.Source:
    target = tmp_path / "decisions.jsonl"
    target.write_text("".join(json.dumps(r) + "\n" for r in records))
    return dashboard.Source(path=target)


def pick_record(**overrides) -> dict:
    record = {
        "ts": 1_800_000_000.0,
        "kind": "gated_pick",
        "ok": True,
        "object": "red block",
        "stopped_at": None,
        "reason": "",
        "confidence": 0.92,
        "vision_confidence": 0.95,
        "zone": "z1",
        "zone_label": "front-left",
        "clarified": False,
        "approval_required": False,
        "approved": None,
        "timed_out": False,
        "moved": True,
        "verified": True,
        "gates": [
            {"gate": "G1_seen", "passed": True, "detail": "found"},
            {"gate": "G2_ambiguous", "passed": True, "detail": "clearly on front-left"},
            {"gate": "G3_reachable", "passed": True, "detail": "macro 0"},
            {"gate": "G4_confidence", "passed": True, "detail": "92%"},
            {"gate": "G5_verify", "passed": True, "detail": "held"},
        ],
    }
    record.update(overrides)
    return record


# --- reading the log -----------------------------------------------------


def test_missing_log_is_empty_not_an_exception(tmp_path):
    assert dashboard.read_records(dashboard.Source(tmp_path / "nope.jsonl")) == []


def test_half_written_lines_are_skipped(tmp_path):
    """The agent appends while we read; a torn last line must not break the screen."""
    target = tmp_path / "decisions.jsonl"
    target.write_text(json.dumps(pick_record()) + "\n" + '{"kind": "gated_pi')
    records = dashboard.read_records(dashboard.Source(target))
    assert len(records) == 1


def test_non_object_lines_are_ignored(tmp_path):
    target = tmp_path / "decisions.jsonl"
    target.write_text('[1,2,3]\n""\n' + json.dumps(pick_record()) + "\n")
    assert len(dashboard.read_records(dashboard.Source(target))) == 1


# --- gates ---------------------------------------------------------------


def test_all_five_gates_are_always_shown(tmp_path):
    state = dashboard.build_state(write_log(tmp_path, [pick_record()]))
    assert [g["gate"] for g in state["gates"]] == [name for name, _ in dashboard.GATE_ORDER]
    assert all(g["state"] == "passed" for g in state["gates"])


def test_unreached_gates_read_as_pending_not_passed(tmp_path):
    """A run that stopped at G1 must not paint G5 green."""
    record = pick_record(
        ok=False, stopped_at="G1_seen", reason="I can't see a red block on the table.",
        verified=None, moved=False,
        gates=[{"gate": "G1_seen", "passed": False, "detail": "not seen"}],
    )
    state = dashboard.build_state(write_log(tmp_path, [record]))
    states = {g["gate"]: g["state"] for g in state["gates"]}
    assert states["G1_seen"] == "stopped"
    assert states["G5_verify"] == "pending"


# --- the confidence number ----------------------------------------------


@pytest.mark.parametrize("stopped_at", ["G1_seen", "G2_ambiguous", "G3_reachable"])
def test_confidence_is_blank_when_it_was_never_computed(tmp_path, stopped_at):
    """0% would read as 'certainly not'; it never got that far."""
    record = pick_record(ok=False, stopped_at=stopped_at, confidence=0.0, verified=None)
    state = dashboard.build_state(write_log(tmp_path, [record]))
    assert state["pick"]["confidence"] is None


def test_confidence_is_shown_when_g4_computed_it(tmp_path):
    record = pick_record(ok=False, stopped_at="G4_confidence", confidence=0.34,
                         approval_required=True, approved=False, verified=None)
    state = dashboard.build_state(write_log(tmp_path, [record]))
    assert state["pick"]["confidence"] == pytest.approx(0.34)


# --- headlines -----------------------------------------------------------


@pytest.mark.parametrize("overrides,expected", [
    ({}, "PICKED IT"),
    ({"ok": False, "stopped_at": "G1_seen"}, "CAN'T SEE IT"),
    ({"ok": False, "stopped_at": "G2_ambiguous"}, "WHICH ONE?"),
    ({"ok": False, "stopped_at": "G3_reachable"}, "NOT TAUGHT THAT SPOT"),
    ({"ok": False, "stopped_at": "G4_confidence"}, "NOT APPROVED"),
    ({"ok": False, "stopped_at": "G4_confidence", "timed_out": True}, "STOOD DOWN — NO ANSWER"),
    ({"ok": False, "stopped_at": "G5_verify", "verified": False}, "MISSED IT — AND SAID SO"),
])
def test_headline_for_each_outcome(tmp_path, overrides, expected):
    state = dashboard.build_state(write_log(tmp_path, [pick_record(**overrides)]))
    assert state["pick"]["headline"] == expected


# --- feed and totals -----------------------------------------------------


def test_feed_marks_refusals_and_failures(tmp_path):
    records = [
        pick_record(),
        pick_record(ok=False, stopped_at="G1_seen", reason="I can't see a red block on the table."),
        {"ts": 1_800_000_001.0, "kind": "eyes_locate", "object": "charger", "found": False},
    ]
    feed = dashboard.build_state(write_log(tmp_path, records))["feed"]
    assert feed[0]["bad"] is True   # newest first: the unseen sighting
    assert any(row["bad"] for row in feed)
    assert any(not row["bad"] for row in feed)


def test_totals_count_the_stand_downs(tmp_path):
    records = [
        pick_record(),
        pick_record(ok=False, stopped_at="G4_confidence", timed_out=True),
        pick_record(ok=False, stopped_at="G2_ambiguous", clarified=True),
    ]
    totals = dashboard.build_state(write_log(tmp_path, records))["totals"]
    assert totals == {"picks": 3, "completed": 1, "stood_down": 1, "clarified": 1, "refused": 2}


def test_unknown_record_kinds_are_left_out_of_the_feed(tmp_path):
    source = write_log(tmp_path, [{"ts": 1.0, "kind": "some_internal_thing", "x": 1}])
    assert dashboard.build_state(source)["feed"] == []


# --- detection overlay ---------------------------------------------------


def test_latest_sighting_is_used_for_the_overlay(tmp_path):
    records = [
        {"ts": 1.0, "kind": "eyes_locate", "object": "charger", "found": True,
         "point": [100, 200], "confidence": 0.8},
        {"ts": 2.0, "kind": "eyes_locate", "object": "red block", "found": True,
         "point": [300, 400], "confidence": 0.9},
    ]
    detection = dashboard.build_state(write_log(tmp_path, records))["detection"]
    assert detection["point"] == [300, 400]
    assert detection["label"] == "red block"


def test_a_failed_sighting_is_not_drawn(tmp_path):
    records = [{"ts": 1.0, "kind": "eyes_locate", "object": "red block", "found": False}]
    assert dashboard.build_state(write_log(tmp_path, records))["detection"] is None


# --- replay --------------------------------------------------------------


def test_replay_walks_through_the_picks(tmp_path):
    records = [pick_record(object="a"), pick_record(object="b"), pick_record(object="c")]
    target = tmp_path / "decisions.jsonl"
    target.write_text("".join(json.dumps(r) + "\n" for r in records))
    source = dashboard.Source(path=target, replay=True, interval_s=0.5)
    seen = {dashboard.current_pick(dashboard.read_records(source), source)["object"]
            for _ in range(1)}
    assert seen <= {"a", "b", "c"}
    assert source.label().startswith("REPLAY")


def test_live_shows_the_most_recent_pick(tmp_path):
    records = [pick_record(object="old"), pick_record(object="newest")]
    source = write_log(tmp_path, records)
    assert dashboard.current_pick(dashboard.read_records(source), source)["object"] == "newest"


# --- the payload the page consumes --------------------------------------


def test_state_is_strict_json(tmp_path):
    """inf or nan would break the page silently."""
    state = dashboard.build_state(write_log(tmp_path, [pick_record(confidence=float("inf"))]))
    json.dumps(state, allow_nan=False)


def test_state_without_any_picks_still_builds(tmp_path):
    state = dashboard.build_state(write_log(tmp_path, []))
    assert state["pick"] is None
    assert state["totals"]["picks"] == 0
    json.dumps(state, allow_nan=False)


def test_threshold_comes_from_config(tmp_path):
    state = dashboard.build_state(write_log(tmp_path, []))
    assert state["threshold"] == config.CONF_APPROVAL


# --- demo hardening ------------------------------------------------------


def test_quips_rotate_rather_than_repeat():
    lines = [agent.quip("moving") for _ in range(len(agent.QUIPS["moving"]))]
    assert len(set(lines)) == len(agent.QUIPS["moving"])


def test_unknown_situation_is_silent_not_an_error():
    assert agent.quip("no-such-situation") == ""


@pytest.mark.parametrize("reason", [
    "my eyes aren't working right now: Gemini quota is exhausted",
    "ClientError: 429 RESOURCE_EXHAUSTED",
    "the pick timed out",
])
def test_infrastructure_failures_get_an_in_character_line(reason):
    assert agent.humanise(reason)


@pytest.mark.parametrize("reason", [
    "I can't see a red block on the table.",
    "No answer in 10 seconds, so I'm standing down.",
    "I'm not sure which red block you mean, so I'm not going to guess.",
    "",
])
def test_honest_refusals_are_left_exactly_as_the_gate_worded_them(reason):
    """A refusal is the product. It must not be dressed up as a technical hiccup."""
    assert agent.humanise(reason) == ""


def test_a_refused_pick_carries_an_excuse_only_when_something_broke():
    from armani import gates

    broken = gates.GatedResult(ok=False, object="red block", stopped_at="G1_seen",
                               reason="my eyes aren't working right now: quota exhausted")
    honest = gates.GatedResult(ok=False, object="red block", stopped_at="G1_seen",
                               reason="I can't see a red block on the table.")
    assert "say" in agent._pick_summary(broken)
    assert "say" not in agent._pick_summary(honest)
