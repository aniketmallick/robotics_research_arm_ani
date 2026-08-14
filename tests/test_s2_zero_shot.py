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
    assert stats.n_steps == int(run_zero_shot.MAX_EPISODE_SECONDS * 8)  # clamped to the hard cap


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
    assert "(policy)" in clamp.clamp_source("policy")
    assert "(recorded)" in clamp.clamp_source("recorded")


# --- clamp profile (S3 fine-tuned eval) ---------------------------------
def test_clamp_profile_default_is_policy():
    assert clamp.resolve_clamp_profile(None, is_base=True) == "policy"
    assert clamp.resolve_clamp_profile(None, is_base=False) == "policy"


def test_recorded_profile_refused_on_base_model():
    """The closed S2 baseline (0/2) can never be re-measured under a wider envelope."""
    with pytest.raises(SystemExit) as exc:
        clamp.resolve_clamp_profile("recorded", is_base=True)
    assert "base model" in str(exc.value)


def test_recorded_profile_fails_closed_without_armani(monkeypatch):
    """No fallback table exists for `recorded` — an embedded guess would drift from the
    operator's real calibration (safety rule 2). Simulates the stripped env by flipping
    the import flag; both the resolver and the clamp itself must refuse, not approximate.

    Non-zero exit: SystemExit carrying a message exits 1, never 0.
    """
    monkeypatch.setattr(clamp, "_HAVE_ARMANI", False)
    with pytest.raises(SystemExit) as exc:
        clamp.resolve_clamp_profile("recorded", is_base=False)
    assert "armani.safety" in str(exc.value)
    assert exc.value.code != 0

    with pytest.raises(SystemExit):
        clamp.policy_clamp(dict(REST), profile="recorded")
    # ...while the policy profile still works from the embedded fallback.
    assert clamp.policy_clamp(dict(REST), profile="policy").bit is False


def test_main_exits_nonzero_when_recorded_requested_without_armani(monkeypatch):
    """End-to-end: the refusal fires before any camera/arm/model is touched."""
    monkeypatch.setattr(clamp, "_HAVE_ARMANI", False)
    with pytest.raises(SystemExit) as exc:
        run_zero_shot.main(
            ["--clamp-profile", "recorded", "--policy-path", "/nonexistent/ckpt",
             "--no-arm", "--synthetic-frame"]
        )
    assert exc.value.code != 0
    assert "armani.safety" in str(exc.value)


def test_recorded_profile_allows_what_policy_clips():
    """The whole point of FIX 1: teleop-derived targets outside +-60 survive `recorded`.

    -100 deg shoulder_lift is inside the training data (min -111.2) and inside the
    recorded envelope (physical -111 + 2 deg margin), but the policy profile clips it
    to -60 and the arm never reaches the table.
    """
    config = pytest.importorskip("armani.config")
    lo, _hi = config.LIMIT_PROFILES["recorded"]["shoulder_lift"]
    action = dict(REST, shoulder_lift=-100.0)

    assert clamp.policy_clamp(action, profile="policy").clamped["shoulder_lift"] == -60.0
    recorded = clamp.policy_clamp(action, profile="recorded")
    assert recorded.clamped["shoulder_lift"] == -100.0
    assert recorded.bit is False
    # ...and it is still an envelope, not "unclamped": past it, recorded bites too.
    beyond = clamp.policy_clamp(dict(REST, shoulder_lift=-999.0), profile="recorded")
    assert beyond.clamped["shoulder_lift"] == lo
    assert beyond.bit is True


def test_clamp_profile_recorded_in_records_and_stats():
    """The profile is part of the audit trail: every step record and the summary."""
    records: list[dict] = []
    now, sleep = fake_clock()
    pytest.importorskip("armani.safety")
    stats = run_zero_shot.run_episode(
        arm=FakeArm(), read_frame=frame_fn, infer_fn=const_infer(REST),
        task="t", live=False, hz=10.0, seconds=999.0, sink=records.append,
        clamp_profile="recorded", now=now, sleep=sleep, stop=stop_after(2),
    )
    assert stats.clamp_profile == "recorded"
    assert "recorded" in stats.clamp_source
    assert all(r["clamp_profile"] == "recorded" for r in records)


def test_step_records_default_to_policy_profile():
    records: list[dict] = []
    now, sleep = fake_clock()
    stats = run_zero_shot.run_episode(
        arm=FakeArm(), read_frame=frame_fn, infer_fn=const_infer(REST),
        task="t", live=False, hz=10.0, seconds=999.0, sink=records.append,
        now=now, sleep=sleep, stop=stop_after(1),
    )
    assert stats.clamp_profile == "policy"
    assert records[0]["clamp_profile"] == "policy"


def test_banner_names_the_model_and_the_envelope(capsys):
    """A fine-tuned eval must not print itself as the zero-shot baseline, and the
    envelope in the send path has to be on screen before anything moves."""
    run_zero_shot._banner("t", "mps", False, "recorded", "/models/ckpt", is_base=False)
    tuned = capsys.readouterr().out
    assert "SPIKE S3 — FINE-TUNED" in tuned and "/models/ckpt" in tuned
    assert "zero-shot" not in tuned
    assert "(recorded)" in tuned

    run_zero_shot._banner("t", "mps", False, "policy", smolvla_io.MODEL_ID, is_base=True)
    base = capsys.readouterr().out
    assert "SPIKE S2 — zero-shot" in base and "(policy)" in base
    assert "RECORDED envelope" not in base


def test_live_confirmation_describes_the_actual_experiment():
    """Safety rule 1 is INFORMED consent. The prompt must name the model, say the arm is
    meant to touch the table, flag a wider-than-policy envelope, and state the real cap —
    a fine-tuned 90 s recorded-envelope run must never be described as untuned/erratic."""
    tuned = run_zero_shot._live_confirmation("/models/ckpt", False, "recorded", 90.0, 30.0)
    assert "/models/ckpt" in tuned and "FINE-TUNED" in tuned
    assert "REACHES FOR THE TABLE" in tuned and "table contact is intended" in tuned
    assert "WIDER than policy" in tuned
    assert "90s" in tuned and "30 Hz" in tuned
    assert "untuned" not in tuned and "erratic" not in tuned

    base = run_zero_shot._live_confirmation(smolvla_io.MODEL_ID, True, "policy", 20.0, 10.0)
    assert "ZERO-SHOT" in base and "untuned" in base
    assert "WIDER than policy" not in base

    # An over-cap request is confirmed at the cap that will actually be enforced.
    assert f"{run_zero_shot.MAX_EPISODE_SECONDS:g}s" in run_zero_shot._live_confirmation(
        "/models/ckpt", False, "policy", 999.0, 30.0
    )


def test_report_warns_when_the_window_closed_before_the_trajectory_finished(capsys):
    """A truncated episode is not a policy failure. The operator must see that before
    scoring, or the run reads as a clean 0 — the false zero the cap was raised to avoid."""
    stats = run_zero_shot.EpisodeStats(n_steps=400, elapsed_s=30.0, clamp_profile="recorded")
    run_zero_shot._report(stats, Path("/x"), "recorded", run_zero_shot.DEMO_WAYPOINTS)
    short = capsys.readouterr().out
    assert "[warn]" in short and "before scoring" in short
    assert "13.3 Hz achieved" in short  # 400 steps / 30 s, so the real rate is visible

    stats = run_zero_shot.EpisodeStats(n_steps=658, elapsed_s=30.0, clamp_profile="recorded")
    run_zero_shot._report(stats, Path("/x"), "recorded", run_zero_shot.DEMO_WAYPOINTS)
    assert "[warn]" not in capsys.readouterr().out  # a full trajectory fits

    # The untuned base has no trajectory length to fall short of — never warn there.
    run_zero_shot._report(run_zero_shot.EpisodeStats(n_steps=110, elapsed_s=13.1), Path("/x"), "policy", None)
    assert "[warn]" not in capsys.readouterr().out


def test_resolve_revision_reads_local_download_metadata(tmp_path):
    """A local checkpoint fetched with `hf download --local-dir` keeps its source commit."""
    sha = "85eb875eb7e58595d383102ae089c78d7e25db49"
    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "model.safetensors.metadata").write_text(f"{sha}\netag\n1786475664.278785\n")
    assert smolvla_io.resolve_revision(str(tmp_path)) == sha


def test_resolve_revision_is_blank_rather_than_guessed(tmp_path):
    """Unknown must read as unknown: a fabricated sha is worse than an empty cell."""
    assert smolvla_io.resolve_revision(str(tmp_path)) == ""  # plain dir, no metadata
    assert smolvla_io.resolve_revision("no-such-org/no-such-model") == ""


def test_parser_clamp_profile_default_and_choices():
    parser = run_zero_shot.build_parser()
    assert parser.parse_args([]).clamp_profile is None  # -> resolves to "policy"
    assert parser.parse_args(["--clamp-profile", "recorded"]).clamp_profile == "recorded"
    with pytest.raises(SystemExit):
        parser.parse_args(["--clamp-profile", "physical"])


# --- smolvla_io pure logic (no model) -----------------------------------
def _spec(action_dim=6, state_dim=6, fill_image_keys=None):
    cams = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")
    return smolvla_io.PolicySpec(
        model_ref=smolvla_io.MODEL_ID,
        image_keys=cams,
        fill_image_keys=cams if fill_image_keys is None else fill_image_keys,
        state_dim=state_dim, action_dim=action_dim, chunk_size=50, device="cpu",
        stats_source="pretrain:so100", stats_dataset="so100",
        routed_features=("observation.state", "action"), action_unnorm="action unnormalize MEAN_STD: ...",
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


class _FakeStatsStep:
    """Mimics a lerobot normalizer step with per-dataset-prefixed stats."""

    def __init__(self):
        self.features = {"action": object(), "observation.state": object()}
        self._tensor_stats = {
            "so100.buffer.action": {"mean": 1, "std": 2},
            "so100.buffer.observation.state": {"mean": 3, "std": 4},
            "so100-red.buffer.action": {"mean": 9, "std": 9},
        }
        self.stats = {"so100.buffer.action": {"mean": [1], "std": [2]}}


class _FakePipe:
    def __init__(self, step):
        self.steps = [step]


def test_route_dataset_stats_aliases_prefixed_keys():
    step = _FakeStatsStep()
    routed = smolvla_io.route_dataset_stats(_FakePipe(step), "so100")
    assert set(routed) == {"action", "observation.state"}
    # the bare key now points at the chosen dataset's tensor stats
    assert step._tensor_stats["action"] is step._tensor_stats["so100.buffer.action"]
    assert step._tensor_stats["observation.state"] is step._tensor_stats["so100.buffer.observation.state"]
    assert step.stats["action"] is step.stats["so100.buffer.action"]  # numpy mirror aliased too


def test_route_dataset_stats_missing_dataset_routes_nothing():
    step = _FakeStatsStep()
    assert smolvla_io.route_dataset_stats(_FakePipe(step), "does-not-exist") == []
    assert "action" not in step._tensor_stats  # nothing aliased


# --- which stats reach the motors (S3 fine-tuned checkpoint) -------------
class _StatsStep:
    """Minimal stand-in for a lerobot normalizer step with the given tensor stats."""

    def __init__(self, tensor_stats: dict):
        self.features = {"action": object(), "observation.state": object()}
        self._tensor_stats = dict(tensor_stats)
        self.stats = {}


def _mean_std(mean, std):
    return {"mean": mean, "std": std}


CKPT = "/models/smolvla_pick_red_v1"


def test_missing_stats_reports_unusable_features():
    step = _StatsStep({"action": _mean_std(1, 2), "observation.state": {"mean": 3}})
    pipe = _FakePipe(step)
    assert smolvla_io.missing_stats(pipe, ("action",)) == ()
    assert smolvla_io.missing_stats(pipe, ("observation.state",)) == ("observation.state",)  # no std
    assert smolvla_io.missing_stats(pipe, ("nope",)) == ("nope",)
    # A pipeline with no stats step at all can normalize nothing.
    assert smolvla_io.missing_stats(_FakePipe(None), ("action",)) == ("action",)


def test_resolve_stats_base_routes_the_pretraining_dataset():
    """S2 baseline behaviour, unchanged: the base's per-embodiment stats get aliased."""
    pre = _FakePipe(_StatsStep({"so100.buffer.observation.state": _mean_std(3, 4)}))
    post = _FakePipe(_StatsStep({"so100.buffer.action": _mean_std(1, 2)}))

    source, dataset, routed = smolvla_io.resolve_stats(pre, post, smolvla_io.MODEL_ID, "so100")

    assert source == "pretrain:so100"
    assert dataset == "so100"
    assert set(routed) == {"observation.state", "action"}


def test_resolve_stats_finetuned_uses_its_own_and_never_routes():
    """FIX 3: a fine-tuned checkpoint's own MEAN_STD stats, never generic so100 ones.

    The fake carries BOTH the checkpoint's bare keys and leftover so100-prefixed keys.
    Routing would be a no-op here anyway (bare keys win), but resolve_stats must not
    even attempt it — and must report the checkpoint, not the dataset, as the source.
    """
    own = _mean_std(14.125, 27.06)  # armani_pick_red_v1's real shoulder_pan mean/std
    pre = _FakePipe(_StatsStep({"observation.state": own, "so100.buffer.observation.state": _mean_std(0, 1)}))
    post = _FakePipe(_StatsStep({"action": own, "so100.buffer.action": _mean_std(0, 1)}))

    source, dataset, routed = smolvla_io.resolve_stats(pre, post, CKPT, "so100")

    assert source == f"checkpoint:{CKPT}"
    assert (dataset, routed) == ("", ())
    assert post.steps[0]._tensor_stats["action"] is own  # untouched by any aliasing


def test_resolve_stats_finetuned_without_own_stats_refuses_instead_of_routing():
    """The load-bearing one: no bare stats -> REFUSE. Never silently substitute so100.

    Denormalizing against the wrong scale commands the arm to wrong positions with no
    error at all, so this has to fail closed before the policy is ever stepped.
    """
    pre = _FakePipe(_StatsStep({"so100.buffer.observation.state": _mean_std(3, 4)}))
    post = _FakePipe(_StatsStep({"so100.buffer.action": _mean_std(1, 2)}))

    with pytest.raises(SystemExit) as exc:
        smolvla_io.resolve_stats(pre, post, CKPT, "so100")

    message = str(exc.value)
    assert CKPT in message and "REFUSING" in message
    assert "preprocessor:observation.state" in message and "postprocessor:action" in message
    assert "action" not in post.steps[0]._tensor_stats  # nothing was routed in


def test_action_stats_line_names_the_numbers_and_flags_absence():
    line = smolvla_io.action_stats_line(_FakePipe(_StatsStep({"action": _mean_std([14.125], [27.06])})))
    assert "MEAN_STD" in line and "14.12" in line and "27.06" in line
    assert "ABSENT" in smolvla_io.action_stats_line(_FakePipe(_StatsStep({})))


def test_spec_summary_names_the_stats_source():
    base = _spec()
    assert "stats=pretrain:so100 (routed [" in base.summary()

    tuned = smolvla_io.PolicySpec(
        model_ref=CKPT, image_keys=("observation.images.camera1",),
        fill_image_keys=("observation.images.camera1",), state_dim=6, action_dim=6,
        chunk_size=50, device="mps", stats_source=f"checkpoint:{CKPT}", stats_dataset="",
        routed_features=(), action_unnorm="action unnormalize MEAN_STD: mean=[...] std=[...]",
    )
    summary = tuned.summary()
    assert f"stats=checkpoint:{CKPT}" in summary and "no pretrain routing" in summary
    assert "so100" not in summary


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


def test_append_trial_row_refuses_a_stale_header(tmp_path):
    """A pre-clamp_profile file must not be appended to: DictWriter would write our
    column order under their header and shift every column of the results table."""
    csv_path = tmp_path / "trials.csv"
    csv_path.write_text("timestamp,episode_tag,task,device,live,n_steps,clamp_bit_rate,score,note\n")
    with pytest.raises(ValueError):
        run_zero_shot.append_trial_row(csv_path, {"episode_tag": "e1", "clamp_profile": "recorded"})


def test_trial_row_records_profile_and_model_identity(tmp_path):
    """A score is only quotable if the row says which envelope AND which weights made it."""
    csv_path = tmp_path / "trials.csv"
    run_zero_shot.append_trial_row(csv_path, {
        "episode_tag": "e1", "clamp_profile": "recorded", "score": 3,
        "model_ref": "/models/ckpt", "model_revision": "85eb875eb7e58595d383102ae089c78d7e25db49",
    })
    header, row = csv_path.read_text().strip().splitlines()
    cols = dict(zip(header.split(","), row.split(",")))
    assert cols["clamp_profile"] == "recorded"
    assert cols["model_ref"] == "/models/ckpt"
    assert cols["model_revision"] == "85eb875eb7e58595d383102ae089c78d7e25db49"


def test_shipped_trials_csv_matches_the_header_and_names_its_model():
    """The real results file must stay appendable and self-describing — the closed S2
    rows carry the base model, so they can never be read as fine-tuned scores."""
    import csv as _csv

    with run_zero_shot.TRIALS_CSV.open(newline="") as handle:
        rows = list(_csv.DictReader(handle))
    assert tuple(rows[0]) == run_zero_shot.TRIALS_HEADER  # append_trial_row would refuse otherwise
    for row in rows:
        assert row["model_ref"] == smolvla_io.MODEL_ID
        assert row["clamp_profile"] == "policy"


def test_episode_cap_covers_a_600_waypoint_demo():
    """The demos are 30 fps, ~600 waypoints; this loop runs one waypoint per loop STEP at
    the --hz pace (10 Hz default), so ~60 s. The cap has to clear that or the pick is cut
    off before the grasp. Defaults themselves stay at the S2 values.

    Measured, not assumed: 813 waypoints in 90.1 s. The loop is pace-bound, not
    inference-bound — median step cost ~9 ms, with a ~400 ms re-plan every 50th step
    (n_action_steps: 50) — so the shortfall from 10 Hz is the re-plan stalls alone.
    """
    assert run_zero_shot.MAX_EPISODE_SECONDS >= 600 / run_zero_shot.DEFAULT_HZ
    assert run_zero_shot.DEFAULT_HZ == 10.0
    assert run_zero_shot.DEFAULT_SECONDS == 20.0


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
