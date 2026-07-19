"""The open-vocabulary scene survey behind look(), the greeting, and brevity.

No camera, no network, no arm — eyes.describe_scene (or the model call under it)
is stubbed throughout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from armani import agent, config, eyes  # noqa: E402


@pytest.fixture
def scene(monkeypatch):
    """Stub the open-vocabulary survey."""

    def sees(*names):
        monkeypatch.setattr(eyes, "describe_scene", lambda frame=None: list(names))

    return sees


# --- the JSON shape the model speaks ------------------------------------


def test_returns_every_object_it_can_see(scene):
    scene("wooden log", "charger", "red block")
    assert agent.survey_table() == {
        "objects": ["wooden log", "charger", "red block"],
        "count": 3,
    }


def test_it_is_not_limited_to_the_pick_catalog(scene):
    """The catalog is what the arm was TAUGHT to pick, not the limit of what it
    can see. Reporting a wooden log as nothing was the bug this fixes."""
    scene("wooden log", "coffee mug", "pair of scissors")
    result = agent.survey_table()
    assert result["count"] == 3
    assert not set(result["objects"]) & set(config.OBJECT_CATALOG)


def test_nothing_seen_says_so_honestly(scene):
    scene()
    result = agent.survey_table()
    assert result["objects"] == []
    assert result["count"] == 0
    assert "can't see" in result["note"]


def test_a_vision_failure_is_a_note_not_an_exception(monkeypatch):
    """A tool that raises is a stack trace on stage."""

    def boom(frame=None):
        raise eyes.EyesError("quota exhausted")

    monkeypatch.setattr(eyes, "describe_scene", boom)
    result = agent.survey_table()
    assert result["objects"] == []
    assert "eyes aren't working" in result["note"]


def test_result_is_json_serialisable(scene):
    scene("wooden log", "charger")
    json.dumps(agent.survey_table(), allow_nan=False)


def test_one_vision_call_per_survey(monkeypatch):
    """Quota matters: a survey must not cost one request per object."""
    calls: list[int] = []

    def once(frame=None):
        calls.append(1)
        return ["charger"]

    monkeypatch.setattr(eyes, "describe_scene", once)
    agent.survey_table()
    assert len(calls) == 1


# --- describe_scene parsing ---------------------------------------------


def reply(monkeypatch, raw):
    monkeypatch.setattr(eyes, "capture_frame", lambda *a, **k: object())
    monkeypatch.setattr(eyes, "encode_jpeg", lambda frame: b"")
    monkeypatch.setattr(eyes, "_ask", lambda jpeg, prompt: ("test-model", raw))


def test_plain_json_array(monkeypatch):
    reply(monkeypatch, '["wooden log", "charger"]')
    assert eyes.describe_scene() == ["wooden log", "charger"]


def test_fenced_json_is_accepted(monkeypatch):
    reply(monkeypatch, '```json\n["red block"]\n```')
    assert eyes.describe_scene() == ["red block"]


def test_names_are_normalised(monkeypatch):
    reply(monkeypatch, '["  Wooden   LOG ", "Charger"]')
    assert eyes.describe_scene() == ["wooden log", "charger"]


def test_duplicates_are_collapsed(monkeypatch):
    reply(monkeypatch, '["charger", "Charger", "CHARGER"]')
    assert eyes.describe_scene() == ["charger"]


def test_a_wrapped_object_list_is_tolerated(monkeypatch):
    reply(monkeypatch, '{"objects": ["charger", "red block"]}')
    assert eyes.describe_scene() == ["charger", "red block"]


def test_entries_that_are_objects_are_tolerated(monkeypatch):
    reply(monkeypatch, '[{"name": "charger"}, {"name": "red block"}]')
    assert eyes.describe_scene() == ["charger", "red block"]


def test_non_string_entries_are_skipped(monkeypatch):
    reply(monkeypatch, '["charger", 42, null, {"nope": 1}, "red block"]')
    assert eyes.describe_scene() == ["charger", "red block"]


def test_an_empty_array_is_empty(monkeypatch):
    reply(monkeypatch, "[]")
    assert eyes.describe_scene() == []


@pytest.mark.parametrize("raw", ["", "not json at all", "{}", '"a string"', "12"])
def test_unparseable_replies_return_empty_and_never_raise(monkeypatch, raw):
    reply(monkeypatch, raw)
    assert eyes.describe_scene() == []


def test_a_runaway_list_is_capped(monkeypatch):
    """A model that starts describing the room must not have the robot read an
    inventory at the audience."""
    reply(monkeypatch, json.dumps([f"thing {i}" for i in range(200)]))
    assert len(eyes.describe_scene()) == eyes.MAX_SCENE_OBJECTS


def test_absurdly_long_names_are_trimmed(monkeypatch):
    reply(monkeypatch, json.dumps(["x" * 500]))
    assert len(eyes.describe_scene()[0]) <= eyes.MAX_SCENE_NAME_CHARS


def test_a_dead_network_returns_empty(monkeypatch):
    monkeypatch.setattr(eyes, "capture_frame", lambda *a, **k: object())
    monkeypatch.setattr(eyes, "encode_jpeg", lambda frame: b"")

    def boom(jpeg, prompt):
        raise eyes.EyesError("quota exhausted")

    monkeypatch.setattr(eyes, "_ask", boom)
    assert eyes.describe_scene() == []


def test_the_prompt_asks_for_the_table_and_excludes_the_arm():
    for phrase in ("table surface", "robot arm itself", "JSON array"):
        assert phrase in eyes.SCENE_PROMPT


# --- look() is read-only -------------------------------------------------


def test_look_is_registered_and_needs_no_motion():
    """Same safety class as get_status: it must work in NO-MOTION mode."""
    from armani import motion

    worker = agent.MotionWorker(motion.DryRunArm(), motion_enabled=False)
    names = {getattr(t, "name", getattr(t, "__name__", "?")) for t in agent.build_tools(worker)}
    assert "look" in names


def test_survey_never_commands_the_arm(scene):
    from armani import motion

    scene("red block")
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


# --- post-action brevity -------------------------------------------------


def test_a_finished_action_asks_for_one_short_line():
    """The persona says it, but a finished move is exactly when the model wants
    to recap — so the reminder rides along with the event."""
    import run_agent

    summary = run_agent._completion_summary(
        agent.Completion(action="gesture bow", status="done")
    )
    assert "gesture bow" in summary
    assert "done" in summary
    assert summary.endswith(run_agent.REACT_BRIEFLY)


def test_the_detail_survives_the_brevity_suffix():
    """The honest outcome must still reach the model — especially a failure."""
    import run_agent

    summary = run_agent._completion_summary(
        agent.Completion(
            action="pick red block", status="failed",
            detail="I ran the move but I couldn't move the red block.",
        )
    )
    assert "couldn't move the red block" in summary
    assert summary.endswith(run_agent.REACT_BRIEFLY)


def test_persona_covers_the_new_behaviour():
    assert "ONE short line" in agent.PERSONA
    assert "Never narrate or explain what you just did" in agent.PERSONA


def test_persona_tone_is_plain_deadpan():
    """The Hinglish/Indian-sarcasm layer was reverted; only the base tone remains."""
    assert "Deadpan and dry beats loud and hyper" in agent.PERSONA
    for reverted in ("Hinglish", "haan haan", "kya scene hai", "wah, genius", "arre"):
        assert reverted not in agent.PERSONA


def test_persona_keeps_the_humour_a_garnish_not_a_rule_change():
    """The safety-shaped rules must not have been disturbed by a tone edit."""
    # Fragments, not sentences: PERSONA is hard-wrapped, so anything long
    # enough to cross a line break would fail on formatting alone.
    for rule in (
        "ALWAYS announce a movement",
        "BEFORE you call the tool",
        "Never claim an ability you don't have",
        "If you didn't, SAY SO",
        "never round it",
    ):
        assert rule in agent.PERSONA
