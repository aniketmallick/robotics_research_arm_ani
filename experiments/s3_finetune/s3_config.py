"""Spike S3 config — the ONE fine-tune task's dataset + recording parameters.

Kept pure (no `armani` import) so the harness and its tests stay hardware-free.
Reusable for a different object later by overriding the env vars — e.g.
`ARMANI_S3_REPO_ID=anikmall/armani_pick_marker_v1 ARMANI_S3_TASK="Pick up the marker"`.
"""

from __future__ import annotations

import os
from pathlib import Path

# The dataset and the instruction string. The instruction is ALSO fixed across every
# demo (SOP) — the model is learning one command, not a vocabulary.
REPO_ID = os.getenv("ARMANI_S3_REPO_ID", "anikmall/armani_pick_red_v1")
TASK = os.getenv("ARMANI_S3_TASK", "Pick up the red block")

# ONE camera named 'camera1' -> dataset key `observation.images.camera1`. SmolVLA is
# a vision policy: WITHOUT this the dataset has no images and cannot fine-tune a VLA
# (the existing taught-zone datasets are state+action only — that is why S3 records a
# fresh, camera-bearing dataset rather than reusing them). The name MUST be 'camera1'
# (not a friendly name like 'front'): smolvla_base declares input_features camera1/2/3,
# and fine-tuning FROM --policy.path keeps that declaration; lerobot only accepts the
# dataset if our camera set is a subset of the policy's, so 'camera1' passes while
# 'front' fails validation (factory.validate_visual_features_consistency).
CAMERA_NAME = "camera1"
CAMERA_INDEX = int(os.getenv("ARMANI_CAMERA_INDEX", "0"))
CAMERA_W, CAMERA_H, CAMERA_FPS = 640, 480, 30  # C920 at the S2/demo resolution

NUM_EPISODES = int(os.getenv("ARMANI_S3_NUM_EPISODES", "50"))  # ~50 CLEAN demos (SOP)
# CORRECTED 2026-08-24 (Option E measurement, ego2so101 research channel). The two lines
# below are WRONG about the delivered dataset and are kept, struck through, rather than deleted:
#
#   > "A PICK - no place: rest -> approach -> grasp -> lift -> stop. Shorter than the 30 s
#   >  pick-AND-place recording; keep every demo about this long (SOP: similar lengths)."
#
# MEASURED from all 50 episodes of anikmall/armani_pick_red_v1 (data/chunk-000/file-000.parquet):
# every episode contains grasp -> sustained closed hold -> RELEASE. 50/50, invariant to the
# detector's only free parameter (sustain 5/10/20 frames). Gripper stays closed a median 157
# frames (5.2 s) while shoulder_pan swings a median 59.4 deg, and the release lands at
# shoulder_pan 62.73 deg +/- 1.44 (sd, n=50) - a consistent place spot. Confirmed visually on
# episodes 0, 23 and 49: approach, grasp, lateral transport, place, retract.
#
# THE DATASET IS A PICK **AND PLACE**. The task string ("Pick up the red block") also does not
# describe the demonstrated behaviour; it is left alone because it is baked into the trained
# checkpoint and the frozen eval, and is logged as an R-A instance instead.
# See Robo_Research_Data/ego2so101/deliverables/OPTION_E_MEASUREMENT.md.
EPISODE_TIME_S = int(os.getenv("ARMANI_S3_EPISODE_TIME_S", "20"))
RESET_TIME_S = int(os.getenv("ARMANI_S3_RESET_TIME_S", "10"))
FPS = 30

# NOTE: the eval checkpoint knob lives in smolvla_io.resolve_checkpoint (reads
# ARMANI_SMOLVLA_CHECKPOINT / --policy-path at call time). It is intentionally NOT
# duplicated here — one source of truth for which policy the eval runner loads.

_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))


def dataset_root() -> Path:
    """Local dataset directory, PINNED to the repo id.

    lerobot-record timestamp-stamps the repo id internally (``stamp_repo_id()`` runs
    unconditionally on the non-resume path), but passing ``--dataset.root`` explicitly
    keeps the on-disk directory at THIS predictable path regardless — the gotcha
    documented in docs/recording_picks.md — so the checker and the Hub push can find it.
    """
    return _LEROBOT_HOME / REPO_ID
