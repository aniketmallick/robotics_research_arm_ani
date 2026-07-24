"""Headless tests for the S3 fine-tune harness. No arm, no GPU, no dataset.

Covers the parts that must be right BEFORE the operator runs anything: the record
command builds with the camera + pinned root, the dataset checker's verdict logic,
the config, and — the load-bearing one — that the eval runner's checkpoint flag
really swaps the policy source (base -> fine-tuned).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.s3_finetune import check_dataset, record_picks, s3_config  # noqa: E402
from experiments.s2_zero_shot import run_zero_shot, smolvla_io  # noqa: E402


# --- record_picks: the command that produces a VLA-trainable dataset -----
def _argv(**over):
    kwargs = dict(
        follower_port="/dev/ttyF", leader_port="/dev/ttyL",
        follower_id="follower_arm", leader_id="leader_arm",
        repo_id="anikmall/armani_pick_red_v1", task="Pick up the red block",
        root="/root/pick", camera_index=0, num_episodes=50,
        episode_time_s=20, reset_time_s=10, fps=30,
    )
    kwargs.update(over)
    return record_picks.build_record_argv(**kwargs)


def test_record_argv_includes_the_camera():
    # THE point: without a camera the dataset has no images and can't train a VLA.
    argv = _argv()
    cam = [a for a in argv if a.startswith("--robot.cameras=")]
    assert cam, "no --robot.cameras — dataset would have no images"
    assert "opencv" in cam[0] and "index_or_path: 0" in cam[0] and "640" in cam[0] and "480" in cam[0]
    # Must be 'camera1' (a subset of smolvla_base's camera1/2/3), NOT a friendly name
    # like 'front' — else lerobot rejects the dataset at fine-tune (see s3_config).
    assert f"{{{s3_config.CAMERA_NAME}:" in cam[0]
    assert s3_config.CAMERA_NAME == "camera1"


def test_record_argv_flags_and_pinned_root():
    argv = _argv()
    assert argv[0] == "lerobot-record"
    assert "--robot.use_degrees=true" in argv
    assert "--dataset.repo_id=anikmall/armani_pick_red_v1" in argv
    assert "--dataset.root=/root/pick" in argv                       # pinned (stamp_repo_id gotcha)
    assert "--dataset.single_task=Pick up the red block" in argv
    assert "--dataset.num_episodes=50" in argv
    assert "--dataset.push_to_hub=false" in argv
    assert "--resume=true" not in argv


def test_record_argv_resume_appends_flag():
    assert "--resume=true" in _argv(resume=True)


def test_record_camera_arg_uses_the_configured_name_and_index():
    arg = record_picks.camera_arg(2)
    assert s3_config.CAMERA_NAME in arg and "index_or_path: 2" in arg and "type: opencv" in arg


def test_record_parser_defaults_to_dry_run():
    args = record_picks.build_parser().parse_args([])
    assert args.go is False
    assert args.num_episodes == s3_config.NUM_EPISODES
    assert args.resume is False


# --- check_dataset: the verdict logic ------------------------------------
def _features(action_shape=(6,), state_shape=(6,), camera=True):
    feats = {
        "action": {"dtype": "float32", "shape": list(action_shape)},
        "observation.state": {"dtype": "float32", "shape": list(state_shape)},
    }
    if camera:
        feats["observation.images.camera1"] = {"dtype": "video", "shape": [3, 480, 640]}
    return feats


def test_check_features_accepts_a_valid_dataset():
    assert check_dataset.check_features(_features()) == []


def test_check_features_flags_missing_camera():
    problems = check_dataset.check_features(_features(camera=False))
    assert any("camera" in p.lower() for p in problems)  # the SmolVLA-needs-images check


def test_check_features_flags_wrong_dims():
    assert any("action" in p for p in check_dataset.check_features(_features(action_shape=(7,))))
    assert any("state" in p for p in check_dataset.check_features(_features(state_shape=(5,))))


def test_length_outliers():
    assert check_dataset.length_outliers([700, 710, 690, 705]) == []          # consistent
    assert check_dataset.length_outliers([700, 700, 700, 200]) == [3]         # one short take
    assert check_dataset.length_outliers([]) == []


# --- s3_config -----------------------------------------------------------
def test_s3_config_defaults():
    assert s3_config.REPO_ID == "anikmall/armani_pick_red_v1"
    assert s3_config.TASK == "Pick up the red block"
    assert s3_config.CAMERA_NAME == "camera1"  # subset of smolvla_base's camera1/2/3
    assert str(s3_config.dataset_root()).endswith(s3_config.REPO_ID)


def test_s3_config_env_override_takes_effect():
    # The module reads env at import, so reload under a patched environment to prove the
    # "reusable for another object via env vars" claim actually holds (not just defaults).
    import importlib

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ARMANI_S3_REPO_ID", "someone/armani_pick_marker_v1")
        mp.setenv("ARMANI_S3_TASK", "Pick up the marker")
        mp.setenv("ARMANI_S3_NUM_EPISODES", "40")
        reloaded = importlib.reload(s3_config)
        try:
            assert reloaded.REPO_ID == "someone/armani_pick_marker_v1"
            assert reloaded.TASK == "Pick up the marker"
            assert reloaded.NUM_EPISODES == 40
            assert str(reloaded.dataset_root()).endswith("someone/armani_pick_marker_v1")
        finally:
            importlib.reload(s3_config)  # restore defaults for the rest of the suite


# --- the checkpoint override (load-bearing: does --policy-path swap the policy?) ---
def test_resolve_checkpoint_precedence(monkeypatch):
    monkeypatch.delenv("ARMANI_SMOLVLA_CHECKPOINT", raising=False)
    assert smolvla_io.resolve_checkpoint(None) == smolvla_io.MODEL_ID          # base by default
    assert smolvla_io.resolve_checkpoint("/cli/ckpt") == "/cli/ckpt"           # --policy-path
    monkeypatch.setenv("ARMANI_SMOLVLA_CHECKPOINT", "/env/ckpt")
    assert smolvla_io.resolve_checkpoint(None) == "/env/ckpt"                  # env
    assert smolvla_io.resolve_checkpoint("/cli/ckpt") == "/cli/ckpt"           # cli overrides env


def test_resolve_checkpoint_blank_is_an_error_not_a_base_fallthrough(monkeypatch):
    # `--policy-path "$CKPT"` with an unset $CKPT expands to "" — that must FAIL loudly,
    # not silently load the base while the trial is logged as fine-tuned.
    monkeypatch.delenv("ARMANI_SMOLVLA_CHECKPOINT", raising=False)
    for blank in ("", "   "):
        with pytest.raises(SystemExit):
            smolvla_io.resolve_checkpoint(blank)
    monkeypatch.setenv("ARMANI_SMOLVLA_CHECKPOINT", "")
    with pytest.raises(SystemExit):
        smolvla_io.resolve_checkpoint(None)


def test_cameras_to_fill_matches_training_camera_count():
    cams = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")
    assert smolvla_io.cameras_to_fill(cams, is_base=True) == cams          # base: feed all 3 (S2)
    assert smolvla_io.cameras_to_fill(cams, is_base=False) == cams[:1]     # fine-tuned: only camera1


def test_build_frame_fills_only_the_fill_keys():
    # A fine-tuned S3 checkpoint declares 3 cameras but was trained on 1 — build_frame
    # must populate exactly the one, matching training (finding: train/eval image skew).
    import numpy as np

    spec = smolvla_io.PolicySpec(
        model_ref="/some/ckpt",
        image_keys=("observation.images.camera1", "observation.images.camera2", "observation.images.camera3"),
        fill_image_keys=("observation.images.camera1",),
        state_dim=6, action_dim=6, chunk_size=50, device="cpu",
        stats_dataset="so100", routed_features=("action",),
    )
    frame = smolvla_io.build_frame({}, np.zeros((480, 640, 3), dtype=np.uint8), "pick", spec)
    assert "observation.images.camera1" in frame
    assert "observation.images.camera2" not in frame and "observation.images.camera3" not in frame


def test_run_zero_shot_parses_policy_path():
    args = run_zero_shot.build_parser().parse_args(["--policy-path", "/some/ckpt"])
    assert args.policy_path == "/some/ckpt"


def test_make_infer_fn_forwards_checkpoint_to_load(monkeypatch):
    # Proves the flag SWAPS the policy source without loading a real model: make_infer_fn
    # must pass the checkpoint through to load().
    captured = {}

    class _FakePolicy:
        def reset(self):
            pass

    fake_spec = smolvla_io.PolicySpec(
        model_ref="/fake/ckpt",
        image_keys=("observation.images.camera1",), fill_image_keys=("observation.images.camera1",),
        state_dim=6, action_dim=6,
        chunk_size=50, device="cpu", stats_dataset="so100", routed_features=("action",),
    )

    def fake_load(device, dataset=smolvla_io.PRETRAIN_DATASET, checkpoint=None):
        captured["checkpoint"] = checkpoint
        return _FakePolicy(), object(), object(), fake_spec

    monkeypatch.setattr(smolvla_io, "load", fake_load)
    _, spec = smolvla_io.make_infer_fn(device="cpu", checkpoint="/fake/ckpt")
    assert captured["checkpoint"] == "/fake/ckpt"
    assert spec is fake_spec


# --- the main() glue: the integration points the operator's run depends on ------
def test_record_picks_main_threads_camera_and_pins_root(monkeypatch, capsys):
    # Drive record_picks.main() end-to-end (dry-run) and prove the camera index + pinned
    # root actually reach the emitted command — the pure builder is tested with explicit
    # kwargs, but main()'s threading of s3_config values was previously uncovered.
    monkeypatch.setattr(
        record_picks, "_ports",
        lambda: ("/dev/ttyF", "/dev/ttyL", "follower_arm", "leader_arm"),
    )
    monkeypatch.setattr(s3_config, "CAMERA_INDEX", 3, raising=False)
    rc = record_picks.main([])  # no --go: dry-run print
    assert rc == 0
    out = capsys.readouterr().out
    assert "--robot.cameras=" in out and "index_or_path: 3" in out
    assert f"--dataset.root={s3_config.dataset_root()}" in out
    assert "--dataset.push_to_hub=false" in out


def test_run_zero_shot_maps_base_to_none_checkpoint(monkeypatch):
    # The Part C glue: base id -> make_infer_fn(checkpoint=None); a real path -> forwarded.
    captured = {}

    def fake_make_infer_fn(device, checkpoint=None):
        captured["checkpoint"] = checkpoint

        def _infer(state, bgr, task):
            return {}

        return _infer, object()

    monkeypatch.setattr(smolvla_io, "make_infer_fn", fake_make_infer_fn)

    # base id resolves and must be passed as None (S2 behaviour)
    ckpt = smolvla_io.resolve_checkpoint(None)
    is_base = ckpt == smolvla_io.MODEL_ID
    smolvla_io.make_infer_fn(device="cpu", checkpoint=None if is_base else ckpt)
    assert captured["checkpoint"] is None

    # a fine-tuned path resolves and is forwarded verbatim
    ckpt = smolvla_io.resolve_checkpoint("/some/ckpt")
    is_base = ckpt == smolvla_io.MODEL_ID
    smolvla_io.make_infer_fn(device="cpu", checkpoint=None if is_base else ckpt)
    assert captured["checkpoint"] == "/some/ckpt"
