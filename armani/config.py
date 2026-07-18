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

import json
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

# lerobot defaults this to True, which means EVERY disconnect() releases torque
# and the arm drops from wherever it is — including immediately after
# capture_home has carefully parked it. We hold instead: a script ending should
# not move the arm. The operator releases it deliberately, via the kill switch's
# [t] option or by powering the arm down.
DISABLE_TORQUE_ON_DISCONNECT = False

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
    # Pure rotation: collision-free, and the calibrated range is +-180, so +-150
    # still leaves margin. Reviewer ruling after the stage-1 review.
    "wrist_roll": (-150.0, 150.0),
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

# How far BEYOND a physical limit a measured position may read before we treat
# it as an encoder/calibration fault and refuse to move. Reads at a mechanical
# stop legitimately overshoot the calibrated range: on 2026-07-18 the parked arm
# read shoulder_lift = -111.69 against a calibrated -111.0, because lerobot's
# DEGREES conversion does not clamp reads to the calibration range.
PHYSICAL_TOLERANCE = 2.0

# Margin pulled in from the physical limit for the "recorded" profile.
RECORDED_MARGIN = 2.0


def _shrink(limits: dict[str, tuple[float, float]], margin: float) -> dict[str, tuple[float, float]]:
    """Pull every limit in by `margin`. The gripper is a percentage, not an
    angle, and needs its full 0-100 travel to open and close, so it is left alone."""
    return {
        joint: (lo, hi) if joint == GRIPPER_JOINT else (lo + margin, hi - margin)
        for joint, (lo, hi) in limits.items()
    }


# Targets are clamped against one of these, chosen by where the target came from:
#
#   policy    LLM- and IK-originated targets. Deliberately conservative.
#   recorded  Gesture replay and return-to-entry recovery. These targets are
#             MEASURED reality, so the conservative policy envelope must not
#             clip them — an entry pose the arm was actually in is by definition
#             reachable, and clipping it would move the arm somewhere it never was.
#   physical  Hard backstop applied at the send boundary only. Protects the
#             servos; nothing should ever reach it.
#   backstop  The send boundary. PHYSICAL widened by the same tolerance the
#             start-pose check allows, because those two must agree: a joint
#             parked at its mechanical stop reads slightly PAST the calibrated
#             range (shoulder_lift -111.69 vs -111.0), and the first steps of
#             any move out of that pose are necessarily still past it. Clamping
#             those to the exact physical limit would fight a legal recovery
#             and log an error on every step of it.
LIMIT_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "policy": JOINT_LIMITS,
    "recorded": _shrink(PHYSICAL_LIMITS, RECORDED_MARGIN),
    "physical": PHYSICAL_LIMITS,
    "backstop": _shrink(PHYSICAL_LIMITS, -PHYSICAL_TOLERANCE),
}
DEFAULT_PROFILE = "policy"

# Joints closer than this to their entry value are treated as "did not move",
# so error recovery does not drag untouched joints around (safety rule 4's
# "zero-length move if nothing moved yet").
MOVED_EPSILON = 0.5


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

# Slow, controlled return to the pose a motion began at (safety rule 4).
RECOVERY_DURATION_S = 3.0

# lerobot's own per-send cap (config.max_relative_target), same units as the
# action. It must be sized for the LOOSEST legitimate motion in the system,
# which is gesture replay — not for interp_move.
#
# Measured on real teleop at 30 fps: frame-to-frame deltas reach 6.16 (gripper),
# 4.04 (shoulder_lift), 3.08 (elbow_flex). A cap derived from MAX_JOINT_SPEED
# (1.8 per step) would silently clip every one of those and distort the replay
# while reporting success. So this is sized off recorded reality, with headroom.
#
# It is a gross-error backstop, NOT the speed policy: interp_move independently
# holds interpolated motion to MAX_JOINT_SPEED / CONTROL_HZ = 1.8 per step.
MAX_FRAME_DELTA = 8.0
MAX_RELATIVE_TARGET = MAX_FRAME_DELTA

# Home pose. The placeholder below (all-zero degrees, the middle of each joint's
# calibrated travel) has never been verified on hardware, so HOME_VERIFIED stays
# False until scripts/capture_home.py records a pose the operator physically set.
# Safety rule 4 forbids auto-driving to an unverified home.
DATA_DIR = REPO_ROOT / "armani" / "data"
HOME_POSE_PATH = DATA_DIR / "home_pose.json"

_PLACEHOLDER_HOME: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}


def _load_home() -> tuple[dict[str, float], bool]:
    """Return (pose, verified). A missing or malformed file is not fatal — it
    just leaves home unverified, which blocks auto-homing rather than the app."""
    if not HOME_POSE_PATH.is_file():
        return dict(_PLACEHOLDER_HOME), False
    try:
        payload = json.loads(HOME_POSE_PATH.read_text())
        # AttributeError included: a "pose" that is a list or string has no
        # .items(), and an unverified home must never be a crash at import time.
        pose = {str(j): float(v) for j, v in payload["pose"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"WARNING: ignoring unreadable {HOME_POSE_PATH}: {exc}")
        return dict(_PLACEHOLDER_HOME), False

    missing = [j for j in JOINTS if j not in pose]
    if missing:
        print(f"WARNING: {HOME_POSE_PATH} is missing joint(s) {missing}; treating home as unverified")
        return dict(_PLACEHOLDER_HOME), False
    return pose, bool(payload.get("verified", False))


HOME_POSE, HOME_VERIFIED = _load_home()

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
# The table polygon lives in the homography file, because it is a product of the
# same calibration: it is only meaningful for the camera/table geometry that
# calibration measured. An empty polygon makes the workspace check fail closed
# rather than silently approving an uncalibrated reach (safety rule 3).
HOMOGRAPHY_PATH = DATA_DIR / "homography.json"

# Height of the TABLE SURFACE in robot-base coordinates (metres). Zero means the
# base sits on the same plane as the table, which is the usual SO-101 mounting.
# The operator measures this during calibration; a robot on a riser makes it
# negative, which moves the whole table into an easier part of the envelope
# (see docs/env_report.md, "top-down reach").
TABLE_HEIGHT_M = float(os.getenv("ARMANI_TABLE_HEIGHT_M") or 0.0)

# How far above the table the gripper hovers. Stage 4 stops here: nothing in the
# codebase may command a Z below this until stage 5 adds the descent.
HOVER_HEIGHT_M = float(os.getenv("ARMANI_HOVER_HEIGHT_M") or 0.10)


def hover_z() -> float:
    """Robot-frame Z of the hover plane. The stage-4 floor for every IK target."""
    return TABLE_HEIGHT_M + HOVER_HEIGHT_M


def _load_table_polygon() -> tuple[tuple[float, float], ...]:
    """Read the calibrated table polygon out of the homography file.

    Missing or malformed calibration is not an import-time crash — it just
    leaves the polygon empty, which fails closed at the workspace check.
    """
    if not HOMOGRAPHY_PATH.is_file():
        return ()
    try:
        payload = json.loads(HOMOGRAPHY_PATH.read_text())
        polygon = tuple(
            (float(x), float(y)) for x, y in payload.get("table_polygon", ())
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"WARNING: ignoring unreadable {HOMOGRAPHY_PATH}: {exc}")
        return ()
    if 0 < len(polygon) < 3:
        print(f"WARNING: {HOMOGRAPHY_PATH} table_polygon has {len(polygon)} points; need 3+. Ignoring.")
        return ()
    return polygon


TABLE_POLYGON: tuple[tuple[float, float], ...] = _load_table_polygon()

# --- Kinematics (stage 4) -------------------------------------------------
# Kinematics-only copy of so101_new_calib.urdf from TheRobotStudio/SO-ARM100
# (Apache-2.0). <visual>/<collision> are stripped because placo refuses to load
# a URDF whose mesh files are absent, and IK needs only the kinematic chain.
# Provenance and the exact strip command are in docs/env_report.md.
URDF_PATH = DATA_DIR / "so101_kin.urdf"
IK_TARGET_FRAME = "gripper_frame_link"

# The five body joints, in URDF order. The gripper is NOT part of the IK chain.
IK_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# lerobot's RobotKinematics.inverse_kinematics is ONE soft-QP step, not an
# iterate-to-convergence solve, and it returns no success flag. We iterate it
# ourselves and then VERIFY by forward kinematics; these bound that loop.
IK_MAX_ITERATIONS = 150
IK_POSITION_TOLERANCE_M = 0.010  # a solution further than this from the ask is a failure

# The placo frame task is SOFT: position and orientation trade against each
# other, and even a small orientation weight costs centimetres of position
# (measured: weight 0.05 gives a ~60 mm position error at hover height).
# solve_top_down walks this ladder and keeps the most vertical solution that
# still lands inside IK_POSITION_TOLERANCE_M. Descending order; the final 0.0
# is position-only and always converges when the point is reachable at all.
IK_ORIENTATION_WEIGHTS: tuple[float, ...] = (0.05, 0.02, 0.008, 0.003, 0.001, 0.0)

# How far the gripper's approach axis may lean off straight-down at hover.
# Measured fact, not a preference: inside the policy envelope a near-vertical
# approach only exists at z <= 0.0 m, and at z = +0.10 m the best achievable
# tilt is about 35 deg. Hover is a staging pose where position is what matters,
# so we allow the lean and report it. See docs/env_report.md.
HOVER_MAX_TILT_DEG = 40.0

# --- Eyes (stage 4) -------------------------------------------------------
# Gemini pointing returns no calibrated confidence. Ours is built from what we
# can actually observe — see eyes.score_detection for the exact formula.
EYES_SAMPLES = 2  # independent pointing queries per locate(); 2 = dual-query agreement
# Agreement between independent queries falls off linearly with how far apart
# they point: full marks at 0 px, half marks at EYES_AGREEMENT_PX, and zero at
# twice it. 40 px at 640x480 is roughly the width of a small demo object, so two
# queries landing on the same object still score well.
EYES_AGREEMENT_PX = 40.0
EYES_SELF_REPORT_WEIGHT = 0.5  # rest of the weight goes to inter-sample agreement
EYES_CONF_THRESHOLD = 0.50  # below this, stage 6 will ask for approval
EYES_MAX_OUTPUT_TOKENS = 512
# Google's robotics-ER docs use temperature 1.0 for pointing. Kept at their
# recommended value rather than forced to 0: the two prompt variants are what
# make the queries independent, and sampling variance the model genuinely has is
# something the agreement term should SEE rather than something we should hide.
EYES_TEMPERATURE = 1.0

# --- Camera calibration (stage 4) ----------------------------------------
# ChArUco board the operator lays flat on the table. Defaults match the board
# generated by scripts/calibrate_camera.py --print-board; override if using a
# pre-printed board and MEASURE the squares with a ruler — a wrong square length
# scales the entire map linearly.
CHARUCO_SQUARES_X = _env_int("ARMANI_CHARUCO_SQUARES_X") or 5
CHARUCO_SQUARES_Y = _env_int("ARMANI_CHARUCO_SQUARES_Y") or 7
CHARUCO_SQUARE_M = float(os.getenv("ARMANI_CHARUCO_SQUARE_M") or 0.030)
CHARUCO_MARKER_M = float(os.getenv("ARMANI_CHARUCO_MARKER_M") or 0.022)
CHARUCO_DICT = os.getenv("ARMANI_CHARUCO_DICT", "DICT_4X4_50")

# findHomography needs 4 correspondences; 6 is the fallback method's ask and
# gives the reprojection check something to actually detect.
CALIB_MIN_POINTS = 6

# The table polygon is the convex hull of the calibrated points pulled in by
# this much (metres). A homography extrapolates confidently and wrongly outside
# the region it was fitted on, so the arm is only allowed where calibration
# actually measured, minus a margin.
TABLE_MARGIN_M = float(os.getenv("ARMANI_TABLE_MARGIN_M") or 0.02)

# A bad map is worse than no map: refuse to save above this mean reprojection
# error. Measured in pixels, by mapping the calibrated robot points back through
# the inverse homography and comparing against the pixels they came from.
CALIB_MAX_REPROJECTION_PX = float(os.getenv("ARMANI_CALIB_MAX_REPROJ_PX") or 15.0)

# --- Object catalog ------------------------------------------------------
# The five demo objects. `grasp_height_m` is how far above the table surface the
# gripper should close on that object — DEFINED NOW, USED IN STAGE 5. Stage 4
# never descends, so these values are not exercised yet.
OBJECT_CATALOG: dict[str, dict[str, float]] = {
    "red block": {"grasp_height_m": 0.015},
    "banana": {"grasp_height_m": 0.015},
    "marker pen": {"grasp_height_m": 0.010},
    "blue cup": {"grasp_height_m": 0.040},
    "toy car": {"grasp_height_m": 0.020},
}

# --- Gestures ------------------------------------------------------------
# One local dataset, one episode per gesture, in this exact order. The operator
# records it with lerobot-record; see docs/recording_gestures.md.
GESTURE_DATASET_REPO_ID = os.getenv("ARMANI_GESTURE_REPO_ID", "anikmall/armani_gestures")

_LEROBOT_HOME = Path(
    os.getenv("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot")
)
def _resolve_gesture_root() -> Path:
    """Where the recorded dataset actually landed.

    lerobot-record calls DatasetConfig.stamp_repo_id() at creation, which appends
    _YYYYMMDD_HHMMSS to the repo id — so `--dataset.repo_id=x/armani_gestures`
    really writes `x/armani_gestures_20260718_193045`. The runbook pins
    --dataset.root to avoid that, but if the operator omits it we would otherwise
    look in a directory that never gets created and SKIP forever with no clue why.
    So: exact path if it exists, else the newest timestamped sibling.

    Only `root` matters to LeRobotDataset; repo_id is just a label (verified).
    """
    explicit = os.getenv("ARMANI_GESTURE_ROOT")
    if explicit:
        return Path(explicit)

    exact = _LEROBOT_HOME / GESTURE_DATASET_REPO_ID
    if (exact / "meta" / "info.json").is_file():
        return exact

    stamped = sorted(
        candidate
        for candidate in exact.parent.glob(f"{exact.name}_*")
        if (candidate / "meta" / "info.json").is_file()
    )
    return stamped[-1] if stamped else exact


GESTURE_DATASET_ROOT = _resolve_gesture_root()

GESTURES: dict[str, int] = {
    "bow": 0,
    "wave": 1,
    "dance": 2,
    "nod_yes": 3,
    "shake_no": 4,
    "look_around": 5,
    "celebrate": 6,
    "sad_droop": 7,
}
GESTURE_RECORD_FPS = 30
GESTURE_EPISODE_TIME_S = 10
GESTURE_PREPOSITION_S = 2.0  # slow move onto the episode's first frame

# --- Improvise (safety rule 8) -------------------------------------------
IMPROVISE_MAX_KEYFRAMES = 8
IMPROVISE_MIN_SECONDS = 0.3
IMPROVISE_MAX_SECONDS = 5.0
# Per-keyframe limits alone allow 8 x 5 = 40s of LLM-authored motion, and
# interp_move's speed stretch makes that a floor, not a ceiling. Cap the whole
# plan as well. See QUESTIONS FOR REVIEWER: CLAUDE.md rule 8 says "max 5s per
# move", which may have meant the whole move rather than each keyframe.
IMPROVISE_MAX_TOTAL_SECONDS = 15.0
IMPROVISE_MAX_RETRIES = 1
IMPROVISE_MAX_TOKENS = 1024

# --- Trust gate thresholds ----------------------------------------------
CONF_APPROVAL = 0.60
APPROVAL_TIMEOUT_S = 10

# --- Voice agent (stage 3) ----------------------------------------------
# Realtime output voice. See RealtimeVoice options in the OpenAI Agents SDK;
# "ash" is the SDK default. Kept in config so the persona's sound is not buried
# in code.
REALTIME_VOICE = os.getenv("ARMANI_REALTIME_VOICE", "ash")

# Push-to-talk: turn detection is OFF (see agent.py); the operator holds this key
# to stream mic audio and releases to commit + request a response. pynput names
# the spacebar "space"; kept configurable for a different PTT key if needed.
PTT_KEY = os.getenv("ARMANI_PTT_KEY", "space")

# Optional sounddevice device overrides. None -> the system default device.
# Accepts either an integer index or a substring of the device name (sounddevice
# resolves both). The demo mic is a wired headset, NOT the C920 or AirPods.
AUDIO_INPUT_DEVICE = os.getenv("ARMANI_AUDIO_INPUT_DEVICE") or None
AUDIO_OUTPUT_DEVICE = os.getenv("ARMANI_AUDIO_OUTPUT_DEVICE") or None

# The realtime API speaks PCM16 mono at 24 kHz (see the SDK's audio_formats.py,
# which pins AudioPCM rate=24000). Both capture and playback must match it.
AUDIO_SAMPLE_RATE = 24_000
AUDIO_CHANNELS = 1
# Frames per mic read. ~20 ms at 24 kHz keeps push-to-talk latency low without
# flooding the socket.
AUDIO_BLOCKSIZE = 480

# Device names that mean the operator forgot to plug in / select the headset.
# Printed as a reminder at session start (CLAUDE.md demo-mic note).
AUDIO_DEVICE_WARN_SUBSTRINGS: tuple[str, ...] = ("AirPods", "MacBook Pro Microphone")

# Keep replies snappy and cheap. Inclusive of tool-call tokens (SDK semantics).
AGENT_MAX_OUTPUT_TOKENS = _env_int("ARMANI_AGENT_MAX_TOKENS") or 1024

# Estimated durations reported to the model when a long action is enqueued, so
# it can talk for roughly the right length while the arm moves. Gestures carry
# their own measured length; this is the fallback for improvised moves.
AGENT_IMPROVISE_ETA_S = 6.0

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
