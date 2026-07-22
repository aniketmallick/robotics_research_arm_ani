"""Load lerobot/smolvla_base and turn one (state, frame, task) into a joint action.

This is the ONLY module that touches the ML stack. It is deliberately thin: it
discovers the checkpoint's real feature spec (it is NOT hard-coded — the base
model dictates it), builds the observation the pipeline expects, runs
preprocess -> select_action -> postprocess, and maps the raw action vector onto
our six SO-101 joints.

Two honesty notes baked in, because this is a zero-shot / out-of-distribution
measurement, not a working policy:

* **Three cameras, one C920.** smolvla_base expects camera1/2/3. We have one
  camera, so the same frame is fed to all three keys. Flagged, not hidden.
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

    image_keys: tuple[str, ...]
    state_dim: int
    action_dim: int
    chunk_size: int
    device: str

    def summary(self) -> str:
        return (
            f"smolvla_base on {self.device}: state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, chunk={self.chunk_size}, "
            f"cameras={list(self.image_keys)}"
        )


def load(device: str) -> tuple[Any, Any, Any, PolicySpec]:
    """Load the policy + pre/post processors on ``device``. Returns (policy, pre, post, spec).

    The checkpoint bakes ``device="cuda"`` into its processor config; we override
    the device_processor step at build time exactly as lerobot's own eval script
    does, otherwise the build asserts CUDA and crashes on this Mac.
    """
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    cfg = policy.config

    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    image_keys = tuple(cfg.image_features)
    state_dim = int(cfg.input_features["observation.state"].shape[0])
    action_dim = int(cfg.action_feature.shape[0])
    spec = PolicySpec(
        image_keys=image_keys,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=int(cfg.chunk_size),
        device=device,
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
    for key in spec.image_keys:
        # Same C920 frame to every expected camera (one-camera OOD). Clone per key
        # so no pipeline step can alias one camera's tensor into another's.
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


def make_infer_fn(device: str) -> tuple[Callable[[dict[str, float], np.ndarray, str], dict[str, float]], PolicySpec]:
    """Load once, return (infer_fn, spec). ``infer_fn(state, frame_bgr, task) -> raw joint action``.

    The policy's internal action queue (temporal chunking) persists across calls,
    so it is reset exactly once here at the start of the episode.
    """
    policy, preprocessor, postprocessor, spec = load(device)
    policy.reset()

    def infer_fn(state_by_joint: dict[str, float], bgr: np.ndarray, task: str) -> dict[str, float]:
        frame = build_frame(state_by_joint, bgr, task, spec)
        vector = infer(policy, preprocessor, postprocessor, frame)
        return action_to_joints(vector, spec)

    return infer_fn, spec
