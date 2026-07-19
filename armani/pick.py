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
    """Did we actually end up holding it? Trust gate G5.

    Two independent signals. The VLM is the real check and wins; the gripper
    reading is a cheap tie-breaker for when the model will not commit.
    """

    gripper_percent: float | None = None
    # Weak secondary signal only: jaws that closed on nothing sit near 0.
    held_guess: bool | None = None
    # The VLM's verdict, and the fused answer the robot actually announces.
    vlm: eyes.HeldCheck | None = None
    held: bool | None = None
    frame_path: str | None = None
    reason: str = ""

    def as_log(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "gripper_percent": (
                None if self.gripper_percent is None else round(self.gripper_percent, 1)
            ),
            "held_guess": self.held_guess,
            "held": self.held,
            "verify_frame": self.frame_path,
            "verify_reason": self.reason,
        }
        if self.vlm is not None:
            payload.update(self.vlm.as_log())
        return payload


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


def read_gripper(arm) -> float | None:
    """The gripper's position, or None if it could not be read.

    Separate from verify_held because this is the only part that touches the
    serial bus. In the voice agent it runs on the motion worker thread, which
    owns the bus, while the VLM half runs elsewhere — reading the bus from two
    threads is how you get a corrupted frame mid-grasp.
    """
    try:
        return float(arm.read_positions()[config.GRIPPER_JOINT])
    except Exception as exc:
        # Verification reports, it never raises: the arm may be holding an
        # object and a failed read must not become an exception mid-grasp.
        log.warning("could not read the gripper: %s", exc)
        return None


def verify_held(
    object_name: str,
    gripper_percent: float | None,
    frame=None,
    use_vlm: bool = True,
) -> VerifyResult:
    """Trust gate G5: did the arm actually end up holding the object?

    Two independent signals, deliberately weighted:

    1. **The VLM** (`eyes.confirm_held`) — the real check. It can see that the
       object is in the jaws AND gone from its spot, which is the thing we
       actually care about.
    2. **Gripper closure** — cheap and weak. The gripper is 0-100 percent with 0
       fully closed, so jaws that shut on nothing read near zero. A thin object
       is indistinguishable from air on this signal alone, which is exactly why
       it does not get the final say.

    The VLM wins when it commits with enough confidence. Closure is the
    tie-breaker for when it will not. If neither can tell, ``held`` is None and
    the robot says so rather than claiming success.

    Takes no ``arm``: everything here is a camera and a network call.
    """
    reasons: list[str] = []

    held_guess: bool | None = None
    if gripper_percent is None:
        reasons.append("gripper position unavailable")
    else:
        held_guess = gripper_percent > config.GRIPPER_EMPTY_MAX_PERCENT
        reasons.append(
            f"gripper at {gripper_percent:.1f}% "
            f"({'something between the jaws' if held_guess else 'closed on nothing'})"
        )

    frame_path: str | None = None
    if frame is None and use_vlm:
        try:
            frame = eyes.capture_frame()
        except Exception as exc:
            reasons.append(f"could not capture a verification frame: {exc}")
    if frame is not None:
        frame_path = _save_verification_frame(frame, reasons)

    vlm: eyes.HeldCheck | None = None
    if use_vlm:
        vlm = eyes.confirm_held(object_name, frame=frame)
        reasons.append(f"vision says held={vlm.held} ({vlm.confidence:.2f}): {vlm.reason}")

    # Fuse. The VLM only overrules closure when it actually committed to an
    # answer AND was confident enough to be worth believing.
    if vlm is not None and vlm.held is not None and vlm.confidence >= config.G5_MIN_CONFIDENCE:
        held = vlm.held
        reasons.append("verdict from vision")
    elif held_guess is not None:
        held = held_guess
        reasons.append("verdict from the gripper reading (vision would not commit)")
    else:
        held = None
        reasons.append("no usable signal — cannot tell")

    return VerifyResult(
        gripper_percent=gripper_percent,
        held_guess=held_guess,
        vlm=vlm,
        held=held,
        frame_path=frame_path,
        reason="; ".join(reasons),
    )


def _save_verification_frame(frame, reasons: list[str]) -> str | None:
    """Keep a picture of the outcome, so a disputed verdict can be reviewed."""
    try:
        import cv2

        config.TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TEST_OUT_DIR / "pick_verify.jpg"
        return str(path) if cv2.imwrite(str(path), frame) else None
    except Exception as exc:
        reasons.append(f"could not save the verification frame: {exc}")
        return None


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

    outcome = (
        verify_held(object_name, read_gripper(arm)) if verify else VerifyResult()
    )
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
