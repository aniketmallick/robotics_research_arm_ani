"""The read-only scene survey behind the look() tool, and the fixed greeting.

No camera, no network, no arm — eyes.list_visible is stubbed throughout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from armani import agent, config, eyes, zones  # noqa: E402

FRAME = (640, 480)

ZONES = zones.ZoneSet(
    zones=(
        zones.Zone("z1", "front-left", (100, 300), 0),
        zones.Zone("z2", "centre", (320, 300), 1),
    ),
    frame_size=FRAME,
    created="test",
)

ON_Z1 = (102, 302)
ON_Z2 = (322, 302)
OFF_THE_SPOTS = (600, 40)


def detection(label, point, frame_size=FRAME):
    return eyes.Detection(
        label=label, point=point, confidence=0.9, frame_size=frame_size, model="test"
    )


@pytest.fixture
def table(monkeypatch):
    """Zones loaded; the caller stubs what vision returns."""
    monkeypatch.setattr(zones, "load_zones", lambda *a, **k: ZONES)
    monkeypatch.setattr(eyes, "capture_frame", lambda *a, **k: object())

    def sees(*detections):
        monkeypatch.setattr(eyes, "list_visible", lambda names, frame=None: list(detections))

    return sees


# --- the JSON shape the model speaks ------------------------------------


def test_returns_each_object_with_its_spot(table):
    table(detection("red block", ON_Z1), detection("charger", ON_Z2))
    result = agent.survey_table()
    assert result == {
        "objects": [
            {"name": "red block", "spot": "front-left"},
            {"name": "charger", "spot": "centre"},
        ],
        "count": 2,
    }


def test_an_object_off_every_spot_has_a_null_spot(table):
    """Still reported — it IS on the table — but with no location claimed."""
    table(detection("marker pen", OFF_THE_SPOTS))
    result = agent.survey_table()
    assert result["objects"] == [{"name": "marker pen", "spot": None}]
    assert result["count"] == 1


def test_nothing_seen_says_so_honestly(table):
    table()
    result = agent.survey_table()
    assert result["objects"] == []
    assert result["count"] == 0
    assert "can't see" in result["note"]


def test_a_vision_failure_is_a_note_not_an_exception(table, monkeypatch):
    """A tool error would surface as a stack trace mid-demo."""

    def boom(names, frame=None):
        raise eyes.EyesError("quota exhausted")

    monkeypatch.setattr(eyes, "list_visible", boom)
    result = agent.survey_table()
    assert result["objects"] == []
    assert "eyes aren't working" in result["note"]


def test_a_camera_failure_is_also_a_note(monkeypatch):
    def boom(*a, **k):
        raise eyes.EyesError("camera busy")

    monkeypatch.setattr(eyes, "capture_frame", boom)
    result = agent.survey_table()
    assert result["objects"] == []
    assert "note" in result


def test_locations_are_not_claimed_across_frame_sizes(table):
    """Pixels only mean something at the size the zones were clicked at."""
    table(detection("red block", ON_Z1, frame_size=(1280, 960)))
    assert agent.survey_table()["objects"] == [{"name": "red block", "spot": None}]


def test_no_zones_defined_still_lists_the_objects(table, monkeypatch):
    monkeypatch.setattr(zones, "load_zones", lambda *a, **k: None)
    table(detection("charger", ON_Z2))
    assert agent.survey_table()["objects"] == [{"name": "charger", "spot": None}]


def test_result_is_json_serialisable(table):
    table(detection("red block", ON_Z1), detection("marker pen", OFF_THE_SPOTS))
    json.dumps(agent.survey_table(), allow_nan=False)


def test_one_vision_call_for_the_whole_catalog(table, monkeypatch):
    """Quota matters: a survey must not cost one request per object."""
    calls: list[list[str]] = []

    def once(names, frame=None):
        calls.append(list(names))
        return []

    monkeypatch.setattr(eyes, "list_visible", once)
    agent.survey_table()
    assert len(calls) == 1
    assert calls[0] == list(config.OBJECT_CATALOG)


# --- look() is read-only -------------------------------------------------


def test_look_is_registered_and_needs_no_motion():
    """Same safety class as get_status: it must work in NO-MOTION mode."""
    from armani import motion

    worker = agent.MotionWorker(motion.DryRunArm(), motion_enabled=False)
    names = {getattr(t, "name", getattr(t, "__name__", "?")) for t in agent.build_tools(worker)}
    assert "look" in names


def test_survey_never_commands_the_arm(table):
    from armani import motion

    table(detection("red block", ON_Z1))
    arm = motion.DryRunArm()
    agent.survey_table()
    assert arm.sends == 0


# --- the fixed greeting --------------------------------------------------


def test_greeting_line_is_a_single_fixed_constant():
    assert config.GREETING_LINE == "Hey! I am Groot!"


@pytest.mark.parametrize("text_mode", [True, False])
def test_the_instruction_demands_the_line_verbatim(text_mode):
    import run_agent

    instruction = run_agent._greeting_instruction(text_mode)
    assert config.GREETING_LINE in instruction
    assert "verbatim" in instruction
    assert "paraphrase" in instruction


def test_the_instruction_matches_how_you_actually_talk_to_it():
    import run_agent

    assert "spacebar" in run_agent._greeting_instruction(text_mode=False)
    assert "type" in run_agent._greeting_instruction(text_mode=True)
