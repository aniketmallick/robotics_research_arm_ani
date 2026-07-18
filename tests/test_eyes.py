"""Reply parsing and the confidence formula. No camera, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import config, eyes  # noqa: E402

FRAME = (640, 480)


# --- point parsing -------------------------------------------------------


def test_normalised_yx_becomes_pixels():
    """Points arrive as [y, x] normalised 0-1000. Getting this backwards would
    put every detection at its mirror image, which is why it has its own test."""
    (label, point, confidence), = eyes._parse_points(
        '[{"point": [500, 250], "label": "banana", "confidence": 0.8}]', FRAME
    )
    assert label == "banana"
    assert point == (int(round(0.25 * 639)), int(round(0.5 * 479)))
    assert confidence == 0.8


def test_corners_map_to_corners():
    (_, top_left, _), = eyes._parse_points('[{"point": [0, 0], "label": "a"}]', FRAME)
    (_, bottom_right, _), = eyes._parse_points('[{"point": [1000, 1000], "label": "b"}]', FRAME)
    assert top_left == (0, 0)
    assert bottom_right == (639, 479)


def test_fenced_json_is_accepted():
    parsed = eyes._parse_points('```json\n[{"point": [500, 500], "label": "x"}]\n```', FRAME)
    assert len(parsed) == 1


def test_prose_around_json_is_accepted():
    parsed = eyes._parse_points('Sure! [{"point": [100, 100], "label": "x"}] hope that helps', FRAME)
    assert len(parsed) == 1


def test_empty_array_means_not_seen():
    assert eyes._parse_points("[]", FRAME) == []


def test_bad_entries_are_skipped_not_fatal():
    """Nine good points and one broken one should yield nine points."""
    parsed = eyes._parse_points(
        '[{"point": [10, 10], "label": "good"}, {"nope": 1}, '
        '{"point": [1, 2, 3]}, {"point": ["a", "b"]}, {"point": [20, 20], "label": "also good"}]',
        FRAME,
    )
    assert [p[0] for p in parsed] == ["good", "also good"]


def test_out_of_range_points_are_clamped_to_the_frame():
    parsed = eyes._parse_points('[{"point": [1200, -50], "label": "edge"}]', FRAME)
    assert parsed[0][1] == (0, 479)


def test_missing_confidence_is_neutral_not_certain():
    (_, _, confidence), = eyes._parse_points('[{"point": [500, 500], "label": "x"}]', FRAME)
    assert confidence == eyes.DEFAULT_SELF_REPORT


def test_absurd_confidence_is_clamped():
    (_, _, high), = eyes._parse_points('[{"point": [1, 1], "confidence": 99}]', FRAME)
    (_, _, low), = eyes._parse_points('[{"point": [1, 1], "confidence": -5}]', FRAME)
    assert (high, low) == (1.0, 0.0)


def test_non_finite_confidence_falls_back():
    (_, _, confidence), = eyes._parse_points('[{"point": [1, 1], "confidence": 1e999}]', FRAME)
    assert confidence == eyes.DEFAULT_SELF_REPORT


def test_bare_object_is_tolerated():
    parsed = eyes._parse_points('{"point": [500, 500], "label": "solo"}', FRAME)
    assert parsed[0][0] == "solo"


def test_wrapped_in_a_key_is_tolerated():
    parsed = eyes._parse_points('{"points": [{"point": [500, 500], "label": "k"}]}', FRAME)
    assert parsed[0][0] == "k"


@pytest.mark.parametrize("reply", ["", "   ", "no json here", "I could not find it"])
def test_unreadable_replies_raise_value_error(reply):
    with pytest.raises(ValueError):
        eyes._parse_points(reply, FRAME)


# --- confidence ----------------------------------------------------------


def test_agreement_is_full_when_queries_pick_the_same_pixel():
    assert eyes._agreement([(100, 100), (100, 100)]) == 1.0


def test_agreement_falls_off_with_distance():
    close = eyes._agreement([(100, 100), (100 + int(config.EYES_AGREEMENT_PX), 100)])
    far = eyes._agreement([(100, 100), (100 + int(4 * config.EYES_AGREEMENT_PX), 100)])
    assert 0.0 < close < 1.0
    assert far == 0.0


def test_a_single_query_cannot_earn_the_agreement_term():
    """One opinion is not a consensus; it must not score as one."""
    assert eyes._agreement([(100, 100)]) == eyes.NO_AGREEMENT_EVIDENCE
    assert eyes.NO_AGREEMENT_EVIDENCE < 1.0


def test_agreement_uses_the_worst_pair_not_the_average():
    """Two queries agreeing does not excuse a third pointing somewhere else."""
    assert eyes._agreement([(100, 100), (100, 100), (600, 400)]) == 0.0


def test_a_missed_query_drags_the_score_down_hard():
    both = eyes.score_detection(1.0, 1.0, 0.9)
    one = eyes.score_detection(0.5, 1.0, 0.9)
    assert one == pytest.approx(both * 0.5)


def test_disagreement_lowers_the_score():
    assert eyes.score_detection(1.0, 0.0, 0.9) < eyes.score_detection(1.0, 1.0, 0.9)


def test_self_report_alone_cannot_produce_a_high_score():
    """A model insisting it is certain, while the queries disagree, must not
    score above the approval threshold on its own say-so."""
    assert eyes.score_detection(1.0, 0.0, 1.0) <= 0.5


def test_score_is_bounded():
    for found in (0.0, 0.5, 1.0):
        for agreement in (0.0, 0.5, 1.0):
            for self_report in (0.0, 0.5, 1.0):
                assert 0.0 <= eyes.score_detection(found, agreement, self_report) <= 1.0


# --- prompts -------------------------------------------------------------


def test_prompt_variants_are_actually_different():
    """The agreement signal is meaningless if both queries ask the same thing."""
    rendered = {eyes.render_prompt(p, "cup") for p in eyes.PROMPT_VARIANTS}
    assert len(rendered) == len(eyes.PROMPT_VARIANTS) >= 2


def test_prompts_state_the_point_format():
    for template in eyes.PROMPT_VARIANTS:
        prompt = eyes.render_prompt(template, "cup")
        assert "[y, x]" in prompt
        assert "0-1000" in prompt
        assert "cup" in prompt
        assert eyes.OBJECTS_PLACEHOLDER not in prompt


def test_rendering_survives_the_literal_json_braces():
    """str.format would raise KeyError on the {"point": ...} example in the
    prompt. This is the regression guard for that bug."""
    for template in eyes.PROMPT_VARIANTS:
        assert '{"point"' in eyes.render_prompt(template, "cup")


def test_locate_rejects_an_empty_name():
    with pytest.raises(ValueError):
        eyes.locate("   ")
