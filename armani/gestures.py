"""Replay recorded teleop episodes as named gesture macros.

One local dataset (`config.GESTURE_DATASET_REPO_ID`), one episode per gesture,
recorded by the operator with `lerobot-record` — see docs/recording_gestures.md.

Frames are streamed at the recorded fps rather than interpolated: a recording is
ground truth that a human already performed safely, and re-timing it would
destroy the character of the gesture. They are still clamped, with the
`recorded` profile (safety rule 2), because teleop legitimately exceeds the
conservative policy envelope — measured: real episodes start at
shoulder_lift = -107.9, which policy would clip by ~48 degrees.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from armani import config, safety
from armani.logutil import get_logger, log_event

log = get_logger("gestures")

Action = dict[str, float]


@dataclass(frozen=True)
class Gesture:
    """One loaded episode, ready to stream."""

    name: str
    episode: int
    fps: int
    frames: tuple[Action, ...]

    @property
    def seconds(self) -> float:
        return len(self.frames) / self.fps if self.fps else 0.0

    @property
    def first(self) -> Action:
        return self.frames[0]

    @property
    def last(self) -> Action:
        return self.frames[-1]


def list_gestures() -> list[str]:
    """Gesture names, in configured episode order. Used by the voice agent."""
    return sorted(config.GESTURES, key=lambda name: config.GESTURES[name])


def dataset_available() -> bool:
    """True when the recorded dataset exists locally. No network access."""
    return (config.GESTURE_DATASET_ROOT / "meta" / "info.json").is_file()


def _missing_dataset_message() -> str:
    return (
        f"gesture dataset not found at {config.GESTURE_DATASET_ROOT}. "
        "The operator records it once — see docs/recording_gestures.md."
    )


def episode_count() -> int:
    """How many episodes the local dataset holds, read straight from its metadata."""
    try:
        info = json.loads((config.GESTURE_DATASET_ROOT / "meta" / "info.json").read_text())
        return int(info.get("total_episodes", 0))
    except (OSError, ValueError, TypeError):
        return 0


def load_gesture(name: str) -> Gesture:
    """Load one episode's action stream. Local only, no video decoding."""
    if name not in config.GESTURES:
        raise KeyError(f"unknown gesture {name!r}; known: {', '.join(list_gestures())}")
    if not dataset_available():
        raise FileNotFoundError(_missing_dataset_message())

    # Imported lazily so that importing this module (e.g. to call
    # list_gestures) does not drag in torch and the whole lerobot stack.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episode = config.GESTURES[name]

    # Check the count ourselves: asking LeRobotDataset for a missing episode
    # raises ValueError('Instruction "train" corresponds to no data!'), which
    # tells an operator who recorded 3 of 8 gestures nothing at all.
    total = episode_count()
    if episode >= total:
        missing = [g for g in list_gestures() if config.GESTURES[g] >= total]
        raise ValueError(
            f"gesture {name!r} is episode {episode}, but the dataset only has {total} "
            f"episode(s). Not yet recorded: {', '.join(missing)}. "
            "Record them in order — see docs/recording_gestures.md."
        )

    dataset = LeRobotDataset(
        config.GESTURE_DATASET_REPO_ID,
        root=config.GESTURE_DATASET_ROOT,
        episodes=[episode],
    )
    if dataset.num_frames == 0:
        raise ValueError(f"gesture {name!r} (episode {episode}) has no frames — re-record it")

    joints = [n.removesuffix(".pos") for n in dataset.meta.features["action"]["names"]]
    unknown = [j for j in joints if j not in config.JOINT_LIMITS]
    if unknown:
        raise ValueError(f"dataset has unknown joint(s) {unknown}; recorded with a different robot?")

    # select_columns avoids touching any video the dataset may carry.
    rows = dataset.hf_dataset.select_columns("action")["action"]
    frames = tuple({j: float(v) for j, v in zip(joints, row)} for row in rows)

    fps = int(dataset.fps)
    if fps <= 0:
        # _stream divides by this; a malformed info.json would be a ZeroDivisionError
        # mid-playback rather than a clear message at load time.
        raise ValueError(f"gesture {name!r} reports fps={dataset.fps!r}; the dataset metadata is broken")

    gesture = Gesture(name=name, episode=episode, fps=fps, frames=frames)
    _check_units(gesture)
    _check_playable(gesture)
    return gesture


def _check_units(gesture: Gesture) -> None:
    """Warn if the recording might not be in degrees.

    A dataset recorded with `use_degrees=false` stores normalised -100..100
    values. Replayed as degrees they are silently, dangerously wrong: -100
    "normalised" is full travel, but -100 degrees is a real and quite different
    angle. Nothing in the dataset metadata records which was used, so the only
    signal is the value range — a degrees recording of this arm reaches past
    100 on shoulder_lift, a normalised one never can.
    """
    body = [j for j in config.JOINTS if j != config.GRIPPER_JOINT]
    peak = max(
        (abs(frame[j]) for frame in gesture.frames for j in body if j in frame),
        default=0.0,
    )
    if peak <= 100.0:
        log.warning(
            "%s: no joint exceeds 100.0 (peak %.1f), so degrees and normalised -100..100 "
            "recordings are indistinguishable here. Confirm it was recorded with "
            "--robot.use_degrees=true, or the replay will be wrong.",
            gesture.name,
            peak,
        )


def _check_playable(gesture: Gesture) -> None:
    """Refuse a recording whose frame-to-frame jumps exceed the send-time cap.

    lerobot clips any single send to `max_relative_target`. A recording with
    larger jumps (a dropped frame, a bad take) would still "play", but the arm
    would silently lag behind and the gesture would come out distorted. Better
    to say so than to perform something nobody recorded.

    Necessary but not sufficient: lerobot compares the goal against the arm's
    PRESENT position, not against the previous frame. If the follower falls
    behind — a stiff joint, a slow bus — the present-to-goal gap can exceed the
    cap even when every frame-to-frame delta is small. That shows up as a
    gesture that lags rather than one that lurches, which is the safer failure.
    """
    worst_joint, worst_delta = "", 0.0
    for before, after in zip(gesture.frames, gesture.frames[1:]):
        for joint, value in after.items():
            delta = abs(value - before[joint])
            if delta > worst_delta:
                worst_joint, worst_delta = joint, delta

    if worst_delta > config.MAX_FRAME_DELTA:
        raise ValueError(
            f"gesture {gesture.name!r} jumps {worst_delta:.1f} on {worst_joint} between frames, "
            f"above the {config.MAX_FRAME_DELTA:g} send cap. lerobot would clip it and the replay "
            "would not match the recording. Re-record this episode more slowly."
        )
    log.debug("%s: largest frame delta %.2f on %s", gesture.name, worst_delta, worst_joint)


def frame_clamp_deviation(gesture: Gesture) -> dict[str, float]:
    """Largest amount the `recorded` profile would alter any frame, per joint.

    Expected to be small (the profile is the physical range minus 2 degrees) but
    non-zero for a recording that pushed against a mechanical stop.
    """
    worst: dict[str, float] = {}
    for frame in gesture.frames:
        # log_clamps=False or this measurement becomes the very spam it exists
        # to replace: one entry per frame, per joint, per gesture.
        clamped = safety.clamp_action(frame, profile="recorded", log_clamps=False)
        for joint, value in frame.items():
            worst[joint] = max(worst.get(joint, 0.0), abs(clamped[joint] - value))
    return worst


def play_gesture(arm, name: str, return_home: bool = True) -> None:
    """Pre-position, stream the recording, then return home if it is verified.

    Motion is wrapped in SafeMotion, so any failure walks the arm back to where
    the gesture began instead of leaving it mid-pose.
    """
    gesture = load_gesture(name)
    log.info("%s: %d frames @ %d fps (%.1fs)", name, len(gesture.frames), gesture.fps, gesture.seconds)

    # One summary event instead of a clamp entry per frame. Streaming clamps
    # silently precisely because this line already recorded the whole story.
    deviation = {j: round(d, 2) for j, d in frame_clamp_deviation(gesture).items() if d > 0.0}
    if deviation:
        log.info("%s: 'recorded' clamp trims %s", name, deviation)
    log_event(
        "gesture_start",
        name=name,
        episode=gesture.episode,
        frames=len(gesture.frames),
        clamp_deviation=deviation,
    )

    from armani import motion

    entry = arm.read_positions()
    with safety.SafeMotion(arm, description=f"gesture {name}"):
        # Onto the first frame slowly and under interpolation: this is the only
        # part of the move that is not a recording, so it gets the full treatment.
        motion.goto(arm, dict(gesture.first), config.GESTURE_PREPOSITION_S, profile="recorded")
        if safety.stop_requested():
            log_event("gesture_aborted", name=name, phase="preposition")
            return

        if not _stream(arm, gesture, entry):
            return

        if return_home and config.HOME_VERIFIED:
            motion.home(arm, slow=True)
        elif return_home:
            log.warning("home not verified; leaving the arm at the gesture's last frame")

    log_event("gesture_done", name=name)


def _stream(arm, gesture: Gesture, entry: Action) -> bool:
    """Send every frame at the recorded rate. False if the kill switch fired."""
    period = 1.0 / gesture.fps
    deadline = time.perf_counter()
    for index, frame in enumerate(gesture.frames):
        if safety.stop_requested():
            log.warning("stop requested at frame %d/%d", index, len(gesture.frames))
            log_event("gesture_aborted", name=gesture.name, phase="stream", frame=index)
            safety.handle_freeze(arm, entry)
            return False
        # log_clamps=False: a take resting on a stop clamps every single frame,
        # and play_gesture already logged one summary of exactly how much.
        arm.send(safety.clamp_action(frame, profile="recorded", log_clamps=False))
        deadline += period
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            deadline = time.perf_counter()  # fell behind; do not accumulate debt
    return True
