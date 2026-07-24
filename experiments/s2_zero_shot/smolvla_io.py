"""Load lerobot/smolvla_base and turn one (state, frame, task) into a joint action.

This is the ONLY module that touches the ML stack. It is deliberately thin: it
discovers the checkpoint's real feature spec (it is NOT hard-coded — the base
model dictates it), builds the observation the pipeline expects, runs
preprocess -> select_action -> postprocess, and maps the raw action vector onto
our six SO-101 joints.

Two honesty notes baked in, because this is a zero-shot / out-of-distribution
measurement, not a working policy:

* **Cameras vs one C920.** smolvla_base declares camera1/2/3. Zero-shot (base) we
  feed the one C920 to all three keys (flagged, not hidden). A fine-tuned S3
  checkpoint was trained on a single-camera dataset (only the first key present
  each frame), so we feed exactly that one key — feeding 3 copies to a
  1-camera-trained policy would be a train/serve mismatch.
* **Positional joint map.** The base model outputs a 6-vector in ITS training
  arm's convention. We map it positionally onto our JOINTS order and unnormalize
  with the checkpoint's OWN action stats — so the numbers land in the base
  model's space, not our SO-101 calibration. That mismatch is the baseline the
  spike measures; the policy clamp downstream is what keeps it safe.

Image format follows lerobot's own observation_processor: HWC uint8 BGR frame ->
RGB -> CHW float32 in [0, 1] (the pipeline's IDENTITY visual norm passes it
through; the model resizes + scales to [-1, 1] internally).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

log = logging.getLogger("s2.smolvla_io")

MODEL_ID = "lerobot/smolvla_base"

# smolvla_base is MULTI-EMBODIMENT: its normalize/unnormalize stats are keyed per
# PRETRAINING dataset ("<dataset>.buffer.<feature>"), never the bare feature name
# ("action", "observation.state") our runner uses. Unrouted, every lookup misses
# and normalization is a SILENT no-op (line "key not in _tensor_stats -> return
# tensor" in normalize_processor.py) — the model then sees raw state and emits raw
# normalized actions that we'd hand to the motors as if they were joint targets.
# Zero-shot inference therefore has to PICK one pretraining dataset's stats. so100
# is the closest embodiment to our SO-101; the choice is inherently arbitrary and
# is part of the out-of-distribution story (the resulting convention is the base
# arm's, not ours). Override via ARMANI_SMOLVLA_STATS_DATASET.
import os as _os

PRETRAIN_DATASET = _os.getenv("ARMANI_SMOLVLA_STATS_DATASET", "so100")

# Our canonical joint order (matches armani.config.JOINTS). The action vector is
# mapped onto these positionally. Kept as a local constant so this module has no
# hard import dependency on armani beyond what the clamp already pulls in.
JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class PolicySpec:
    """What the loaded checkpoint actually expects/produces — discovered, not assumed."""

    model_ref: str  # what actually loaded: the base id or a local fine-tuned dir
    image_keys: tuple[str, ...]  # every camera key the checkpoint DECLARES
    fill_image_keys: tuple[str, ...]  # keys we populate from our one C920 (see load)
    state_dim: int
    action_dim: int
    chunk_size: int
    device: str
    stats_dataset: str  # which pretraining dataset's stats we routed to (OOD choice)
    routed_features: tuple[str, ...]  # features whose stats were successfully routed

    def summary(self) -> str:
        fill, declared = list(self.fill_image_keys), list(self.image_keys)
        cams = f"cameras filled {fill}" + (f" of declared {declared}" if declared != fill else "")
        return (
            f"{self.model_ref} on {self.device}: state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, chunk={self.chunk_size}, "
            f"{cams}, stats={self.stats_dataset} (routed {list(self.routed_features)})"
        )


def stats_step(pipeline: Any) -> Any | None:
    """Return a pipeline's stats-bearing (normalizer / unnormalizer) step, or None."""
    for step in getattr(pipeline, "steps", None) or []:
        if isinstance(getattr(step, "stats", None), dict):
            return step
    return None


def route_dataset_stats(pipeline: Any, dataset: str) -> list[str]:
    """Alias ``<dataset>.buffer.<feature>`` stats onto the bare ``<feature>`` key the
    transform actually looks up. Returns the feature keys successfully routed.

    Without this the multi-embodiment checkpoint's stats never match and
    normalization silently no-ops (see PRETRAIN_DATASET). Aliasing both the tensor
    stats (used by the transform) and the numpy mirror keeps the step consistent.
    """
    step = stats_step(pipeline)
    if step is None:
        return []
    tensor_stats = getattr(step, "_tensor_stats", None)
    if tensor_stats is None:
        return []
    numpy_stats = getattr(step, "stats", {}) or {}
    routed: list[str] = []
    for feature in list(getattr(step, "features", {}) or {}):
        source = f"{dataset}.buffer.{feature}"
        if feature not in tensor_stats and source in tensor_stats:
            tensor_stats[feature] = tensor_stats[source]
            if source in numpy_stats:
                numpy_stats[feature] = numpy_stats[source]
            routed.append(feature)
    return routed


def resolve_checkpoint(cli_path: str | None = None) -> str:
    """The checkpoint to load: an explicit path, else ARMANI_SMOLVLA_CHECKPOINT, else the
    base model. A LOCAL fine-tuned checkpoint dir (Spike S3) swaps the policy; everything
    downstream — clamp, kill switch, caps, scoring — is unchanged. Pure/testable.

    A provided-but-BLANK source is an error, not a silent fall-through to the base: the
    eval docs pass ``--policy-path "$CKPT"``, so an unset ``$CKPT`` expands to ``""`` and
    would otherwise run the base model while the trial is still logged as fine-tuned —
    corrupting the S3-vs-baseline comparison with no error raised.
    """
    if cli_path is not None and not cli_path.strip():
        raise SystemExit(
            "--policy-path was set but empty (is $CKPT unset?). Point it at a fine-tuned "
            "checkpoint dir, or omit it entirely to run the base model."
        )
    if cli_path:
        return cli_path
    env = _os.getenv("ARMANI_SMOLVLA_CHECKPOINT")
    if env is not None and not env.strip():
        raise SystemExit(
            "ARMANI_SMOLVLA_CHECKPOINT is set but empty. Point it at a fine-tuned checkpoint "
            "dir, or unset it to run the base model."
        )
    return env or MODEL_ID


def cameras_to_fill(image_keys: tuple[str, ...], is_base: bool) -> tuple[str, ...]:
    """Which declared camera keys to populate from our single C920. Pure/testable.

    Base (zero-shot): all of them — feed the one frame to camera1/2/3 (the S2 OOD
    choice, kept for baseline comparability). Fine-tuned S3 checkpoint: exactly the
    first key — it was trained on our one-camera dataset (only camera1 present each
    frame, empty_cameras=0 pads nothing), so feeding 3 copies would be a train/serve
    mismatch.
    """
    return image_keys if is_base else image_keys[:1]


def load(
    device: str, dataset: str = PRETRAIN_DATASET, checkpoint: str | None = None
) -> tuple[Any, Any, Any, PolicySpec]:
    """Load the policy + pre/post processors on ``device``. Returns (policy, pre, post, spec).

    ``checkpoint`` is a LOCAL fine-tuned checkpoint directory (Spike S3, via
    ``--policy-path`` / ``ARMANI_SMOLVLA_CHECKPOINT``) or None to load the base model
    (``MODEL_ID``, S2 behaviour). Only the weights + stats change; the eval path is identical.

    The checkpoint bakes ``device="cuda"`` into its processor config; we override
    the device_processor step at build time exactly as lerobot's own eval script
    does, otherwise the build asserts CUDA and crashes on this Mac. We then route
    ``dataset``'s per-embodiment stats onto the bare feature keys so normalization
    actually runs (see PRETRAIN_DATASET) — a clean no-op for a fine-tuned checkpoint,
    whose stats are already keyed by the bare feature.
    """
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    model_ref = checkpoint or MODEL_ID
    policy = SmolVLAPolicy.from_pretrained(model_ref)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    cfg = policy.config

    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=model_ref,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    routed = route_dataset_stats(preprocessor, dataset) + route_dataset_stats(postprocessor, dataset)
    # Base model: stats are keyed per pretraining dataset, so routing is required. A
    # fine-tuned checkpoint keys them by the bare "action" already, so routing is a clean
    # no-op. Warn only when the ACTION unnormalize stats are STILL absent after both — a
    # genuinely broken load, not the fine-tuned case (which would otherwise false-alarm).
    post_step = stats_step(postprocessor)
    if not (post_step is not None and "action" in (getattr(post_step, "_tensor_stats", None) or {})):
        log.error(
            "action unnormalize stats absent (checkpoint=%r, dataset=%r) — outputs would be raw "
            "normalized values. Base model: check ARMANI_SMOLVLA_STATS_DATASET; fine-tuned "
            "checkpoint: its training stats should be keyed 'action'.", model_ref, dataset,
        )

    image_keys = tuple(cfg.image_features)
    fill_image_keys = cameras_to_fill(image_keys, is_base=model_ref == MODEL_ID)
    state_dim = int(cfg.input_features["observation.state"].shape[0])
    action_dim = int(cfg.action_feature.shape[0])
    spec = PolicySpec(
        model_ref=model_ref,
        image_keys=image_keys,
        fill_image_keys=fill_image_keys,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=int(cfg.chunk_size),
        device=device,
        stats_dataset=dataset,
        routed_features=tuple(routed),
    )
    if state_dim != len(JOINT_ORDER):
        log.warning(
            "checkpoint state_dim=%d != our %d joints; state will be padded/truncated (OOD)",
            state_dim, len(JOINT_ORDER),
        )
    if action_dim != len(JOINT_ORDER):
        log.warning(
            "checkpoint action_dim=%d != our %d joints; mapping the first %d positionally (OOD)",
            action_dim, len(JOINT_ORDER), min(action_dim, len(JOINT_ORDER)),
        )
    return policy, preprocessor, postprocessor, spec


def frame_to_chw_float(bgr: np.ndarray):
    """C920 BGR HxWx3 uint8 -> torch RGB CHW float32 in [0, 1] (model input format)."""
    import torch

    rgb = np.ascontiguousarray(bgr[:, :, ::-1])  # BGR -> RGB, matches training colour order
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
    return tensor


def state_vector(state_by_joint: dict[str, float], state_dim: int):
    """Our joint readings -> a state tensor of length ``state_dim`` (JOINTS order).

    Padded with zeros / truncated if the checkpoint's state dim differs from our
    6 joints — an OOD case that is logged at load time.
    """
    import torch

    values = [float(state_by_joint.get(joint, 0.0)) for joint in JOINT_ORDER]
    if len(values) < state_dim:
        values = values + [0.0] * (state_dim - len(values))
    elif len(values) > state_dim:
        values = values[:state_dim]
    return torch.tensor(values, dtype=torch.float32)


def build_frame(state_by_joint: dict[str, float], bgr: np.ndarray, task: str, spec: PolicySpec) -> dict:
    """Assemble the pipeline input: one state tensor, the frame under every camera key, the task."""
    image = frame_to_chw_float(bgr)
    frame: dict = {"observation.state": state_vector(state_by_joint, spec.state_dim), "task": task}
    for key in spec.fill_image_keys:
        # Populate only the keys the loaded policy actually expects present (all three
        # for the base, one for a fine-tuned S3 checkpoint). Clone per key so no pipeline
        # step can alias one camera's tensor into another's.
        frame[key] = image.clone()
    return frame


def infer(policy: Any, preprocessor: Any, postprocessor: Any, frame: dict) -> np.ndarray:
    """Run preprocess -> select_action -> postprocess. Returns the raw action vector (numpy)."""
    import torch

    with torch.no_grad():
        batch = preprocessor(frame)
        action = policy.select_action(batch)
        out = postprocessor(action)
    array = out.detach().to("cpu").numpy() if hasattr(out, "detach") else np.asarray(out)
    return array.ravel()


def action_to_joints(vector: np.ndarray, spec: PolicySpec) -> dict[str, float]:
    """Map the raw action vector positionally onto our joint names (OOD assumption)."""
    n = min(len(vector), len(JOINT_ORDER), spec.action_dim)
    return {JOINT_ORDER[i]: float(vector[i]) for i in range(n)}


def make_infer_fn(
    device: str, dataset: str = PRETRAIN_DATASET, checkpoint: str | None = None
) -> tuple[Callable[[dict[str, float], np.ndarray, str], dict[str, float]], PolicySpec]:
    """Load once, return (infer_fn, spec). ``infer_fn(state, frame_bgr, task) -> raw joint action``.

    ``checkpoint`` (Spike S3) loads a local fine-tuned checkpoint instead of the base.
    The policy's internal action queue (temporal chunking) persists across calls,
    so it is reset exactly once here at the start of the episode.
    """
    policy, preprocessor, postprocessor, spec = load(device, dataset, checkpoint)
    policy.reset()

    def infer_fn(state_by_joint: dict[str, float], bgr: np.ndarray, task: str) -> dict[str, float]:
        frame = build_frame(state_by_joint, bgr, task, spec)
        vector = infer(policy, preprocessor, postprocessor, frame)
        return action_to_joints(vector, spec)

    return infer_fn, spec
