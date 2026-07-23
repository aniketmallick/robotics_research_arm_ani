"""S1 fix: polygon boundary margin + kill-switch teardown/honesty. No hardware.

Three fixes, all unit-testable on a laptop:
  1. point_in_polygon(margin_m) dilates the allowed region so a target on a hull
     vertex (or nudged epsilon-outside by detection jitter) is accepted, while a
     clearly-outside point is still refused (safety rule 3 holds).
  2. safety.release_kill_switch() stops the pynput ESC listener before exit.
  3. safety.warn_kill_switch_untrusted() loudly warns when ESC is dead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import calibrate, config, safety  # noqa: E402

# A 40x40 cm square (metres). Strict ray-casting rejects the top vertex (0.4, 0.4)
# under its half-open convention — exactly the "rejects its own boundary" bug.
SQUARE = ((0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4))


# --- 1. workspace margin -------------------------------------------------
def test_margin_admits_a_hull_vertex_that_strict_rejects():
    assert not calibrate.point_in_polygon(0.4, 0.4, SQUARE)               # strict: boundary rejected
    assert calibrate.point_in_polygon(0.4, 0.4, SQUARE, margin_m=0.015)   # margin admits the vertex


def test_margin_admits_a_point_just_outside_within_margin():
    # 10 mm past the right edge, mid-height.
    assert not calibrate.point_in_polygon(0.410, 0.2, SQUARE)
    assert calibrate.point_in_polygon(0.410, 0.2, SQUARE, margin_m=0.015)


def test_margin_still_refuses_a_clearly_outside_point():
    # 10 cm out — far beyond any sane margin — stays refused.
    assert not calibrate.point_in_polygon(0.5, 0.2, SQUARE, margin_m=0.015)
    assert not calibrate.point_in_polygon(0.6, 0.6, SQUARE, margin_m=0.015)


def test_margin_just_beyond_is_refused():
    # 20 mm out with a 15 mm margin -> refused (the margin is not a blank cheque).
    assert not calibrate.point_in_polygon(0.420, 0.2, SQUARE, margin_m=0.015)


def test_margin_zero_is_strict_unchanged():
    assert calibrate.point_in_polygon(0.2, 0.2, SQUARE, margin_m=0.0)        # inside
    assert not calibrate.point_in_polygon(0.4, 0.4, SQUARE, margin_m=0.0)    # boundary still rejected
    assert not calibrate.point_in_polygon(0.410, 0.2, SQUARE, margin_m=0.0)  # just-outside still rejected


def test_margin_fails_closed_on_bad_input():
    assert not calibrate.point_in_polygon(float("nan"), 0.2, SQUARE, margin_m=0.015)
    assert not calibrate.point_in_polygon(float("inf"), 0.2, SQUARE, margin_m=0.015)
    assert not calibrate.point_in_polygon(0.2, 0.2, (), margin_m=0.015)                    # empty
    assert not calibrate.point_in_polygon(0.1, 0.1, ((0.0, 0.0), (0.4, 0.4)), margin_m=0.015)  # degenerate len<3
    # A non-finite MARGIN must not flip the check fail-OPEN: a clearly-outside point
    # stays refused even with margin_m = inf/nan (the np.isfinite(margin_m) guard).
    assert not calibrate.point_in_polygon(0.5, 0.2, SQUARE, margin_m=float("inf"))
    assert not calibrate.point_in_polygon(0.5, 0.2, SQUARE, margin_m=float("nan"))
    # A zero-area (collinear) polygon with len>=3 still fails closed WITH a margin.
    collinear = ((0.0, 0.0), (0.2, 0.2), (0.4, 0.4))
    assert not calibrate.point_in_polygon(0.2, 0.21, collinear, margin_m=0.015)


def test_negative_margin_does_not_erode():
    # A negative margin must NOT shrink the region. Use a point only 5 mm inside the
    # right edge: strict admits it, and an (incorrect) inward erode of 0.05 would reject it.
    assert calibrate.point_in_polygon(0.395, 0.2, SQUARE, margin_m=-0.05)
    assert calibrate.point_in_polygon(0.395, 0.2, SQUARE)  # identical to strict


def test_margin_admits_and_refuses_diagonally_past_a_corner():
    # Exercises the point-to-segment vertex projection (t clamped to an endpoint): a
    # point diagonally past the (0.4, 0.4) corner is sqrt(2)*d from it.
    assert calibrate.point_in_polygon(0.41, 0.41, SQUARE, margin_m=0.015)      # dist ~0.0141 <= 0.015
    assert not calibrate.point_in_polygon(0.42, 0.42, SQUARE, margin_m=0.015)  # dist ~0.0283 >  0.015


def test_config_polygon_margin_default_is_sane():
    assert isinstance(config.POLYGON_MARGIN_M, float)
    # Capped so an env override can never silently dilate past the real table.
    assert 0.0 <= config.POLYGON_MARGIN_M <= config.POLYGON_MARGIN_MAX_M
    assert config.POLYGON_MARGIN_MAX_M <= 0.05  # a table dilation, not a barn door


# --- 2. kill-switch teardown --------------------------------------------
def test_release_kill_switch_without_listener_is_noop():
    safety._esc_listener = None
    safety.release_kill_switch()  # must not raise
    assert safety._esc_listener is None


def test_release_kill_switch_stops_and_clears_the_listener():
    stopped = {"v": False}

    class _FakeListener:
        def stop(self):
            stopped["v"] = True

    safety._esc_listener = _FakeListener()
    safety.release_kill_switch()
    assert stopped["v"] is True
    assert safety._esc_listener is None


def test_release_kill_switch_swallows_stop_errors():
    class _Boom:
        def stop(self):
            raise RuntimeError("teardown race")

    safety._esc_listener = _Boom()
    safety.release_kill_switch()  # must not propagate
    assert safety._esc_listener is None


# --- 3. kill-switch honesty (trusted check) ------------------------------
def test_esc_listener_trusted_returns_bool_or_none():
    value = safety.esc_listener_trusted()
    assert value is None or isinstance(value, bool)


def test_warn_kill_switch_untrusted_warns_only_when_false(monkeypatch, capsys):
    monkeypatch.setattr(safety, "esc_listener_trusted", lambda: False)
    assert safety.warn_kill_switch_untrusted() is True
    out = capsys.readouterr().out
    assert "ESC" in out and "Ctrl-C" in out  # names the dead key and the live fallback

    monkeypatch.setattr(safety, "esc_listener_trusted", lambda: True)
    assert safety.warn_kill_switch_untrusted() is False
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(safety, "esc_listener_trusted", lambda: None)  # unknown -> don't cry wolf
    assert safety.warn_kill_switch_untrusted() is False
