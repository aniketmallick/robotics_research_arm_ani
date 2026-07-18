"""Pick a named object by replaying the taught macro for the zone it sits on.

The whole demo pick path, in one sentence: Gemini says WHICH marked spot holds
the thing you asked for, and the arm replays the grasp a human already
demonstrated at that spot.

Every step that could be wrong refuses instead of guessing, and refusing costs
nothing because the arm has not moved yet:

* object not seen               -> no motion, ``seen=False``
* not on any marked spot        -> no motion, the zone match says how far off
* between two spots             -> no motion, ``ambiguous=True`` + both candidates
* that zone has no macro yet    -> no motion, says which episode is missing

Those four are exactly the inputs stage 6's trust gates need, which is why
``PickResult`` carries them as fields rather than as a single error string.
This module deliberately does NOT talk to the voice agent and is not a tool yet
(CLAUDE.md rule 6: gates live in our code, and stage 6 wires them).

Motion is not reimplemented here. A pick macro is a recorded teleop episode
exactly like a gesture, so it loads and streams through ``gestures.py`` — same
``recorded`` clamp profile, same kill-switch checks, same SafeMotion recovery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from armani import config, eyes, gestures, zones
from armani.logutil import get_logger, log_event

log = get_logger("pick")

RUNBOOK = "docs/recording_picks.md"


@dataclass(frozen=True)
class VerifyResult:
    """Did we actually end up holding it? (Partial G5 — completed in stage 6.)"""

    gripper_percent: float | None = None
    # Weak secondary signal only: jaws that closed on nothing sit near 0.
    held_guess: bool | None = None
    frame_path: str | None = None
    reason: str = ""

    def as_log(self) -> dict[str, object]:
        return {
            "gripper_percent": (
                None if self.gripper_percent is None else round(self.gripper_percent, 1)
            ),
            "held_guess": self.held_guess,
            "verify_frame": self.frame_path,
            "verify_reason": self.reason,
        }


@dataclass(frozen=True)
class PickResult:
    """What happened, in the fields stage 6's gates read."""

    ok: bool
    reason: str = ""
    object: str = ""
    seen: bool = False
    confidence: float = 0.0
    zone: str | None = None
    zone_label: str | None = None
    ambiguous: bool = False
    candidate_zones: tuple[str, ...] = ()
    assignment_margin: float = 0.0
    distance_px: float = 0.0
    moved: bool = False
    verify: VerifyResult = field(default_factory=VerifyResult)

    def __bool__(self) -> bool:
        return self.ok

    @property
    def held_guess(self) -> bool | None:
        return self.verify.held_guess

    def as_log(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "object": self.object,
            "seen": self.seen,
            "confidence": round(self.confidence, 3),
            "zone": self.zone,
            "zone_label": self.zone_label,
            "ambiguous": self.ambiguous,
            "candidate_zones": list(self.candidate_zones),
            # A lone zone has no runner-up, so the margin is infinite. json.dumps
            # writes that as the literal `Infinity`, which is not valid JSON and
            # would break every reader of the decision log.
            "assignment_margin_px": (
                round(self.assignment_margin, 1)
                if math.isfinite(self.assignment_margin)
                else None
            ),
            "distance_px": round(self.distance_px, 1) if math.isfinite(self.distance_px) else None,
            "moved": self.moved,
            **self.verify.as_log(),
        }


# --- Macros --------------------------------------------------------------


def macro_available(zone: zones.Zone) -> bool:
    """True when this zone's pick episode has actually been recorded."""
    return zone.pick_episode < gestures.episode_count(config.PICK_DATASET_ROOT)


def load_pick(zone: zones.Zone) -> gestures.Gesture:
    """Load a zone's recorded pick macro through the gesture engine."""
    return gestures.load_episode(
        f"pick:{zone.label}",
        config.PICK_DATASET_REPO_ID,
        config.PICK_DATASET_ROOT,
        zone.pick_episode,
        RUNBOOK,
    )


def play_pick(arm, zone: zones.Zone) -> bool:
    """Replay a zone's pick macro. False if the kill switch stopped it.

    ``return_home=False`` is load-bearing, not a default carried over: home
    commands every joint INCLUDING the gripper, so homing after a successful
    grasp would open the jaws and drop the object on the table. The macro is
    recorded to end back near home while still holding.
    """
    return gestures.play_macro(arm, load_pick(zone), return_home=False, kind="pick")


# --- Verification (partial G5) -------------------------------------------


def verify_held(arm, object_name: str, save_frame: bool = True) -> VerifyResult:
    """After the lift: are we actually holding it?

    Two signals, one of which is not implemented yet:

    1. Gripper closure — cheap and available now. The gripper is 0-100 percent
       with 0 fully closed, so jaws that shut on nothing read near zero while
       jaws stopped by an object rest higher. This is WEAK: the threshold has to
       be calibrated against the recorded close target, and a very thin object
       is indistinguishable from air.
    2. The VLM check — TODO(stage 6): ask Gemini "is the <object> in the gripper,
       and is it gone from its spot?" over the re-captured frame. That is the
       real G5 verification; the frame is captured here so the hook is ready and
       so there is a picture of the outcome in the log either way.
    """
    reasons: list[str] = []

    gripper_percent: float | None = None
    held_guess: bool | None = None
    try:
        pose = arm.read_positions()
        gripper_percent = float(pose[config.GRIPPER_JOINT])
    except Exception as exc:
        # Verification reports, it never raises: the arm is holding an object
        # and a failed read must not become an exception mid-grasp.
        reasons.append(f"could not read the gripper: {exc}")
    else:
        held_guess = gripper_percent > config.GRIPPER_EMPTY_MAX_PERCENT
        reasons.append(
            f"gripper at {gripper_percent:.1f}% "
            f"({'something between the jaws' if held_guess else 'closed on nothing'})"
        )

    frame_path: str | None = None
    if save_frame:
        try:
            frame = eyes.capture_frame()
            import cv2

            config.TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = config.TEST_OUT_DIR / "pick_verify.jpg"
            if cv2.imwrite(str(path), frame):
                frame_path = str(path)
        except Exception as exc:
            # Verification is a report, not a control path: never let it raise
            # while the arm is holding an object.
            reasons.append(f"could not capture a verification frame: {exc}")

    # TODO(stage 6, gate G5): run the Gemini check over the captured frame —
    # "is the <object> held in the gripper, and gone from its marked spot?" —
    # and let it override held_guess. Deliberately not called here: stage 5 does
    # not wire vision into a gate, and a half-wired gate is worse than none.
    reasons.append(f"VLM confirmation of {object_name!r} not implemented (stage 6)")

    return VerifyResult(
        gripper_percent=gripper_percent,
        held_guess=held_guess,
        frame_path=frame_path,
        reason="; ".join(reasons),
    )


# --- The pick ------------------------------------------------------------


def pick_object(arm, object_name: str, frame=None, verify: bool = True) -> PickResult:
    """Find the named object, decide which marked spot it is on, replay that pick.

    Returns a falsy PickResult WITHOUT moving whenever the decision is not clean.
    """
    if not object_name or not object_name.strip():
        raise ValueError("object_name is empty")
    object_name = object_name.strip()

    zone_set = zones.load_zones()
    if zone_set is None:
        return _refuse(object_name, "no zones defined — run: python scripts/define_zones.py")

    # --- see it ---
    try:
        detection = eyes.locate(object_name, frame=frame)
    except eyes.EyesError as exc:
        return _refuse(object_name, f"vision unavailable: {exc}")

    if detection is None:
        return _refuse(object_name, f"I cannot see a {object_name} on the table")

    # --- place it on a spot ---
    match = zones.assign_zone(detection, zone_set)
    candidates = tuple(
        z.id for z in (match.zone, match.runner_up) if z is not None
    )
    base = PickResult(
        ok=False,
        object=object_name,
        seen=True,
        confidence=detection.confidence,
        zone=None if match.zone is None else match.zone.id,
        zone_label=None if match.zone is None else match.zone.label,
        ambiguous=match.ambiguous,
        candidate_zones=candidates if match.ambiguous else (),
        assignment_margin=0.0 if match.zone is None else match.margin_px,
        distance_px=0.0 if match.zone is None else match.distance_px,
    )

    if match.zone is None:
        return _log(_replace(base, reason=match.reason or "could not assign it to a marked spot"))
    if match.ambiguous:
        return _log(
            _replace(
                base,
                reason=(
                    f"the {object_name} is between two spots — {match.reason}. "
                    "Which one do you mean?"
                ),
            )
        )

    # --- do we know how to pick from there? ---
    if not macro_available(match.zone):
        recorded = gestures.episode_count(config.PICK_DATASET_ROOT)
        return _log(
            _replace(
                base,
                reason=(
                    f"no pick macro for {match.zone.label!r}: it needs episode "
                    f"{match.zone.pick_episode} but only {recorded} are recorded. See {RUNBOOK}."
                ),
            )
        )

    try:
        macro = load_pick(match.zone)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return _log(_replace(base, reason=f"pick macro would not load: {exc}"))

    log.info(
        "picking %r from %s (%.0f px away, %.0f px clear of the runner-up, confidence %.2f)",
        object_name, match.zone.label, match.distance_px, match.margin_px, detection.confidence,
    )
    # as_log() already carries `object`; passing it again as a keyword is a
    # TypeError for duplicate arguments — and it would fire here, on the happy
    # path, one line before the arm moves.
    log_event("pick_start", **base.as_log())

    # --- move ---
    completed = gestures.play_macro(arm, macro, return_home=False, kind="pick")
    if not completed:
        return _log(
            _replace(
                base,
                reason="the pick was stopped by the kill switch before it finished",
                moved=True,
            )
        )

    outcome = verify_held(arm, object_name) if verify else VerifyResult()
    return _log(_replace(base, ok=True, moved=True, verify=outcome, reason=""))


def _replace(result: PickResult, **changes) -> PickResult:
    return replace(result, **changes)


def _refuse(object_name: str, reason: str) -> PickResult:
    result = PickResult(ok=False, reason=reason, object=object_name)
    return _log(result)


def _log(result: PickResult) -> PickResult:
    log_event("pick_result", **result.as_log())
    if not result.ok:
        log.warning("not picking: %s", result.reason)
    return result
