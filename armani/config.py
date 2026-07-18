"""Central configuration for ARM-ANI.

Units matter here and are easy to get wrong, so they are stated explicitly:

- The five body joints are commanded in DEGREES. lerobot 0.5.2's ``SOFollower``
  defaults to ``use_degrees=True``, which puts the body motors in
  ``MotorNormMode.DEGREES``. This project pins that default explicitly
  (``USE_DEGREES``) so a future lerobot default flip cannot silently change the
  meaning of every number in this file.
- The gripper is ALWAYS ``MotorNormMode.RANGE_0_100`` in ``SOFollower``,
  regardless of ``use_degrees``. It is a percentage (0 = closed, 100 = open),
  NOT degrees. Clamping it with a degree limit would be a unit error.

Action dicts use lerobot's feature keys: ``"<joint>.pos"`` e.g. ``"shoulder_pan.pos"``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


# --- Paths ---------------------------------------------------------------
LOG_DIR = REPO_ROOT / "logs"
DECISION_LOG = LOG_DIR / "decisions.jsonl"
TEST_OUT_DIR = REPO_ROOT / "tests" / "out"

# --- Dry run -------------------------------------------------------------
# When true, nothing is sent to the motors; intended actions are printed.
DRY_RUN = _env_flag("ARMANI_DRY_RUN", default=False)

# --- Robot ---------------------------------------------------------------
# Calibration ids as they already exist on this machine. lerobot resolves
# calibration to ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json
# (the directory is the class ``name``, which is "so_follower" in 0.5.2).
# These MUST match the existing files — we never recreate calibration.
FOLLOWER_ID = os.getenv("ARMANI_FOLLOWER_ID", "follower_arm")
LEADER_ID = os.getenv("ARMANI_LEADER_ID", "leader_arm")

# Ports are per-boot on macOS (/dev/tty.usbmodem*) and must be discovered.
# None means "ask the operator" rather than "guess".
FOLLOWER_PORT = os.getenv("ARMANI_FOLLOWER_PORT") or None
LEADER_PORT = os.getenv("ARMANI_LEADER_PORT") or None
SERIAL_PORT_GLOB = "/dev/tty.usbmodem*"

# Pin the normalisation mode rather than inheriting a library default.
USE_DEGREES = True

JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_JOINT = "gripper"

# --- Joint limits --------------------------------------------------------
# Policy limits from CLAUDE.md safety rule 2: base +/-90 deg, other joints
# +/-60 deg. Gripper is a 0-100 percentage and is excluded from that rule.
#
# PHYSICAL_LIMITS below were computed from the follower calibration file
# (deg = (ticks - (range_min+range_max)/2) * 360 / 4095) and are recorded so
# that a future widening of the policy limits can be checked against reality
# instead of guessed. Every policy limit is currently strictly inside its
# physical limit, so the policy is the binding constraint. This is verified at
# import time by _assert_limits_within_physical().
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-60.0, 60.0),
    "elbow_flex": (-60.0, 60.0),
    "wrist_flex": (-60.0, 60.0),
    "wrist_roll": (-60.0, 60.0),
    "gripper": (0.0, 100.0),  # percent, not degrees
}

PHYSICAL_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-114.9, 114.9),
    "shoulder_lift": (-111.0, 111.0),
    "elbow_flex": (-97.7, 97.7),
    "wrist_flex": (-96.3, 96.3),
    "wrist_roll": (-180.0, 180.0),
    "gripper": (0.0, 100.0),
}


def _assert_limits_within_physical() -> None:
    """Fail loudly at import if a policy limit exceeds the calibrated range."""
    for joint, (lo, hi) in JOINT_LIMITS.items():
        plo, phi = PHYSICAL_LIMITS[joint]
        if lo < plo or hi > phi:
            raise ValueError(
                f"JOINT_LIMITS[{joint}] = ({lo}, {hi}) exceeds the calibrated "
                f"physical range ({plo}, {phi}). Refusing to load an unsafe limit table."
            )


_assert_limits_within_physical()

# --- Motion --------------------------------------------------------------
CONTROL_HZ = 25  # interpolation rate; CLAUDE.md requires 20-30 Hz
MAX_JOINT_SPEED = 45.0  # deg/s for body joints, percent/s for the gripper
HOME_DURATION_S = 3.0  # deliberately slow; used by home(slow=True)

# How far outside a joint limit a MEASURED position may sit before we refuse to
# move at all. Covers encoder noise, not a genuinely out-of-envelope arm.
ENVELOPE_TOLERANCE = 1.0

# Defence in depth: lerobot's own per-step cap (config.max_relative_target),
# in the same units as the action. Derived from the interpolator's own per-step
# budget rather than picked by hand, so it is guaranteed to sit just above what
# interp_move can legitimately emit and to actually catch a runaway step. A
# hand-picked constant here was 4x looser than MAX_JOINT_SPEED and so could
# never have fired.
MAX_RELATIVE_TARGET = 2.0 * MAX_JOINT_SPEED / CONTROL_HZ

# PLACEHOLDER home pose. All-zero degrees is the middle of each joint's
# calibrated travel because calibration used set_half_turn_homings(). This has
# NOT been verified on hardware yet and must be refined by the operator in
# stage 2 before it is trusted as a resting pose.
HOME_POSE: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}

# --- Camera --------------------------------------------------------------
CAMERA_INDEX = _env_int("ARMANI_CAMERA_INDEX")  # None -> prompt the operator
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_MAX_PROBE_INDEX = 4  # how many OpenCV indices the smoke test scans

# --- Audio ---------------------------------------------------------------
MIC_SAMPLE_RATE = 16_000
MIC_CHANNELS = 1
MIC_TEST_SECONDS = 2.0

# --- Workspace -----------------------------------------------------------
# PLACEHOLDER table polygon in robot XY (metres), replaced in stage 4 by the
# homography calibration. Deliberately left empty: an empty polygon makes
# gates.py fail closed rather than silently approving an uncalibrated reach.
TABLE_POLYGON: tuple[tuple[float, float], ...] = ()

# --- Trust gate thresholds ----------------------------------------------
CONF_APPROVAL = 0.60
APPROVAL_TIMEOUT_S = 10

# --- Models --------------------------------------------------------------
REALTIME_MODEL = "gpt-realtime-2.1"
GEMINI_MODELS: tuple[str, ...] = (
    "gemini-robotics-er-1.6-preview",
    "gemini-robotics-er-1.5-preview",
    "gemini-flash-latest",
)
ANTHROPIC_MODEL = "claude-sonnet-4-5"

# --- API keys ------------------------------------------------------------
# Values are read on demand and never printed. Only presence is ever reported.
API_KEY_VARS: tuple[str, ...] = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY")


def api_key(name: str) -> str | None:
    """Return an API key by env var name, or None when unset/blank."""
    return (os.getenv(name) or "").strip() or None
