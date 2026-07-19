"""Trust gates: the ordered, Python-enforced pipeline around a taught-zone pick.

CLAUDE.md safety rule 6: **gates live here, never in the LLM prompt.** The voice
model's only jobs are to SPEAK the question a gate produces and to RELAY the
human's answer back in. It never decides whether a gate passes, and there is no
code path in which the model's output alone moves the arm. Prompts are style;
this file is law.

The pipeline, in order — each one can only stop the pick, never start it:

    G1 seen         eyes.locate found the named object at all
    G2 ambiguous    two candidates for the name, or an object sitting between
                    two marked spots -> ask the human WHICH, re-resolve in
                    Python, and only then continue
    G3 reachable    a taught macro actually exists for the chosen zone
    G4 confidence   state the number; below CONF_APPROVAL demand spoken approval
                    within APPROVAL_TIMEOUT_S or STAND DOWN without moving
    G5 verify       after the macro, check the grasp with the VLM and announce
                    the outcome honestly, including failure

``clarify`` and ``approve`` are INJECTED so the same pipeline serves the console
smoke test and the voice agent unchanged. That injection is also why the
approval deadline is enforced *here*, with our own clock, rather than being
delegated to the callable: a hung or dishonest callable must still stand down.

Fail-closed is the default everywhere. Unseen, unresolved, unapproved, timed
out, or unverified all mean the arm does not act — and says why.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from armani import config, eyes, pick, zones
from armani.logutil import get_logger, log_event

log = get_logger("gates")

# Gate names, used in the decision log and in GatedResult.stopped_at. Stable
# strings: stage 7's dashboard renders exactly this log.
G1_SEEN = "G1_seen"
G2_AMBIGUOUS = "G2_ambiguous"
G3_REACHABLE = "G3_reachable"
G4_CONFIDENCE = "G4_confidence"
G5_VERIFY = "G5_verify"

# Returned by an injected callable that did not answer in time.
TIMED_OUT = object()

# How the caller performs the macro once every gate before G5 has passed.
# Injected so the voice agent can hand the motion to its worker thread instead
# of running it inline. Returns (started, detail).
Perform = Callable[[zones.Zone], "PerformOutcome"]
Clarify = Callable[[str, list[str]], str | None]
Approve = Callable[[str, float], bool]


@dataclass(frozen=True)
class PerformOutcome:
    """What happened when the macro actually ran.

    ``moved`` is separate from ``completed`` because "the worker refused to take
    the job" and "the arm moved and was interrupted" are different facts, and
    reporting the first as movement would be a lie in the audit trail.
    """

    completed: bool
    detail: str = ""
    gripper_percent: float | None = None
    moved: bool = True


@dataclass(frozen=True)
class GateRecord:
    """One gate's verdict, for the audit trail."""

    gate: str
    passed: bool
    detail: str = ""
    data: dict = field(default_factory=dict)

    def as_log(self) -> dict[str, object]:
        return {"gate": self.gate, "passed": self.passed, "detail": self.detail, **self.data}


@dataclass(frozen=True)
class GatedResult:
    """The whole run: what happened, which gate stopped it, and why.

    This IS the artefact — stage 7's dashboard and the judges' audit trail both
    read it, so every field is filled in even on the paths that refuse.
    """

    ok: bool
    object: str
    stopped_at: str | None = None  # gate name, or None when the pick completed
    reason: str = ""
    confidence: float = 0.0
    vision_confidence: float = 0.0
    zone: str | None = None
    zone_label: str | None = None
    clarified: bool = False
    approval_required: bool = False
    approved: bool | None = None
    timed_out: bool = False
    moved: bool = False
    verified: bool | None = None
    records: tuple[GateRecord, ...] = ()

    def __bool__(self) -> bool:
        return self.ok

    def as_log(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "object": self.object,
            "stopped_at": self.stopped_at,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "vision_confidence": round(self.vision_confidence, 3),
            "zone": self.zone,
            "zone_label": self.zone_label,
            "clarified": self.clarified,
            "approval_required": self.approval_required,
            "approved": self.approved,
            "timed_out": self.timed_out,
            "moved": self.moved,
            "verified": self.verified,
            "gates": [r.as_log() for r in self.records],
        }

    def speak(self) -> str:
        """One line the robot can say. Style is the persona's; facts are ours."""
        if self.ok:
            done = (
                "and it's in the tray" if config.PICK_MODE == "place" else "and I've got it"
            )
            return f"{self.object}, {self.confidence:.0%} — {done}."
        return self.reason


def confidence_for(vision_confidence: float, margin_px: float) -> float:
    """THE confidence number, defined once (config documents the formula).

    Vision confidence says "that is a red block". It says nothing about WHICH
    marked spot the block is on, so the assignment margin tempers it: an object
    barely inside one zone is a shakier pick than the same object dead centre.
    The floor keeps a tight assignment from zeroing an otherwise certain
    detection — being between two spots is a reason to hesitate, not a reason to
    disbelieve your eyes.
    """
    # `not (x >= 0)` also catches NaN, where every comparison is False. An
    # INFINITE margin is legitimate — a lone zone has no runner-up to be
    # confused with — and clamps to full clarity below.
    if not (margin_px >= 0):
        margin_px = 0.0
    clarity = min(max(margin_px / config.CONF_CLEAR_MARGIN_PX, 0.0), 1.0)
    floor = config.CONF_ASSIGNMENT_FLOOR
    return round(min(max(vision_confidence * (floor + (1.0 - floor) * clarity), 0.0), 1.0), 3)


def _ask_with_deadline(fn: Callable, args: tuple, timeout_s: float):
    """Run an injected callable with OUR deadline. Returns TIMED_OUT if it misses.

    The whole point of G4 is that silence means no. Delegating the countdown to
    the callable would mean a hung voice handler, a wedged console read or a
    model that simply never calls back could leave a pick pending forever and
    then fire it late. So the deadline is enforced here.

    The worker is a daemon thread: if the callable never returns, it is
    abandoned rather than joined, and the process can still exit.
    """
    box: list = [TIMED_OUT]

    def _run() -> None:
        try:
            box[0] = fn(*args)
        except Exception as exc:  # an injected callable must not break the gate
            log.error("injected callable raised: %s", exc)
            box[0] = None

    worker = threading.Thread(target=_run, name="gate-input", daemon=True)
    worker.start()
    worker.join(timeout_s)
    return box[0]


def run_gated_pick(
    arm,
    object_name: str,
    *,
    clarify: Clarify,
    approve: Approve,
    verify_vlm: bool = True,
    perform: Perform | None = None,
    frame=None,
    approval_timeout_s: float | None = None,
    on_confidence: Callable[[float], None] | None = None,
) -> GatedResult:
    """Run the five gates around a taught-zone pick.

    ``clarify(question, options) -> str | None`` and
    ``approve(prompt, timeout_s) -> bool`` are injected; the console smoke test
    supplies terminal versions and the agent supplies voice ones. Neither is
    trusted with the deadline.

    ``perform(zone) -> PerformOutcome`` runs the macro. The default runs it
    inline; the agent passes one that enqueues it on the motion worker.
    """
    if not object_name or not object_name.strip():
        raise ValueError("object_name is empty")
    object_name = object_name.strip()
    if approval_timeout_s is None:
        approval_timeout_s = config.APPROVAL_TIMEOUT_S
    if perform is None:
        perform = _default_perform(arm)

    records: list[GateRecord] = []

    def stop(gate: str, reason: str, **fields) -> GatedResult:
        records.append(GateRecord(gate, passed=False, detail=reason, data=fields))
        result = GatedResult(
            ok=False, object=object_name, stopped_at=gate, reason=reason,
            records=tuple(records), **_carry(fields),
        )
        log_event("gated_pick", **result.as_log())
        log.info("STOPPED at %s: %s", gate, reason)
        return result

    zone_set = zones.load_zones()
    if zone_set is None:
        return stop(G3_REACHABLE, "I don't have any taught spots yet — nobody has shown me where to pick.")

    # ---------------- G1: seen ----------------
    try:
        # ONE frame for the whole run. If G2 needs a second look to find where
        # the other candidates are, it must reason about the same image G1 did —
        # re-capturing would let the scene change underneath the gate.
        if frame is None:
            frame = eyes.capture_frame()
        detection = eyes.locate(object_name, frame=frame)
    except eyes.EyesError as exc:
        return stop(G1_SEEN, f"my eyes aren't working right now: {exc}")

    if detection is None:
        return stop(G1_SEEN, f"I can't see a {object_name} on the table.")
    records.append(
        GateRecord(G1_SEEN, True, f"found at {detection.point}", {
            "vision_confidence": round(detection.confidence, 3),
            "candidates": detection.candidates,
        })
    )

    # ---------------- G2: ambiguous ----------------
    match = zones.assign_zone(detection, zone_set)
    clarified = False

    if match.zone is None:
        return stop(G2_AMBIGUOUS, f"I can see the {object_name}, but it isn't on one of my marked spots.",
                    vision_confidence=round(detection.confidence, 3))

    if detection.candidates > 1 or match.ambiguous:
        resolved = _resolve_ambiguity(
            object_name, detection, match, zone_set, clarify, records,
            frame=frame, timeout_s=approval_timeout_s,
        )
        if resolved is None:
            return stop(
                G2_AMBIGUOUS,
                f"I'm not sure which {object_name} you mean, so I'm not going to guess.",
                vision_confidence=round(detection.confidence, 3),
            )
        match, clarified = resolved, True
    else:
        records.append(
            GateRecord(G2_AMBIGUOUS, True, f"clearly on {match.zone.label}", match.as_log())
        )

    zone = match.zone
    assert zone is not None  # G2 either resolved to a zone or stopped above

    # ---------------- G3: reachable ----------------
    if not pick.macro_available(zone):
        return stop(
            G3_REACHABLE,
            f"I don't have a taught pick for {zone.label} — nobody has shown me that spot.",
            zone=zone.id, zone_label=zone.label,
            vision_confidence=round(detection.confidence, 3),
        )
    records.append(GateRecord(G3_REACHABLE, True, f"macro {zone.pick_episode} exists",
                              {"zone": zone.id, "episode": zone.pick_episode}))

    # ---------------- G4: confidence + approval + timeout ----------------
    confidence = confidence_for(detection.confidence, match.margin_px)
    needs_approval = confidence < config.CONF_APPROVAL
    if on_confidence is not None:
        # Lets a caller report the number alongside its own question (the voice
        # agent puts it in the JSON the model reads) without having to scrape it
        # back out of the prompt string.
        try:
            on_confidence(confidence)
        except Exception as exc:
            log.warning("on_confidence observer raised: %s", exc)
    approved: bool | None = None
    timed_out = False

    if needs_approval:
        prompt = (
            f"I think that's the {object_name} on {zone.label}, but I'm only "
            f"{confidence:.0%} sure. Want me to go for it?"
        )
        answer = _ask_with_deadline(approve, (prompt, approval_timeout_s), approval_timeout_s)
        if answer is TIMED_OUT:
            timed_out = True
            approved = False
            return stop(
                G4_CONFIDENCE,
                f"No answer in {approval_timeout_s:.0f} seconds, so I'm standing down.",
                confidence=confidence, zone=zone.id, zone_label=zone.label,
                vision_confidence=round(detection.confidence, 3),
                approval_required=True, approved=False, timed_out=True, clarified=clarified,
            )
        approved = bool(answer)
        if not approved:
            return stop(
                G4_CONFIDENCE, "Understood — not touching it.",
                confidence=confidence, zone=zone.id, zone_label=zone.label,
                vision_confidence=round(detection.confidence, 3),
                approval_required=True, approved=False, clarified=clarified,
            )

    records.append(
        GateRecord(G4_CONFIDENCE, True,
                   f"{confidence:.0%}" + (" (approved)" if needs_approval else " (above threshold)"),
                   {"confidence": confidence, "threshold": config.CONF_APPROVAL,
                    "approval_required": needs_approval, "approved": approved})
    )

    # ---------------- move ----------------
    log.info("all gates passed for %r on %s at %.0f%%", object_name, zone.label, confidence * 100)
    outcome = perform(zone)
    if not outcome.completed:
        records.append(GateRecord(G5_VERIFY, False, outcome.detail or "the pick did not finish"))
        result = GatedResult(
            ok=False, object=object_name, stopped_at=G5_VERIFY,
            reason=outcome.detail or "the pick didn't finish.",
            confidence=confidence, vision_confidence=detection.confidence,
            zone=zone.id, zone_label=zone.label, clarified=clarified,
            approval_required=needs_approval, approved=approved, timed_out=timed_out,
            moved=outcome.moved, verified=None, records=tuple(records),
        )
        log_event("gated_pick", **result.as_log())
        return result

    # ---------------- G5: verify ----------------
    verification = pick.verify_held(object_name, outcome.gripper_percent, use_vlm=verify_vlm)
    held = verification.held
    records.append(
        GateRecord(G5_VERIFY, bool(held), verification.reason, verification.as_log())
    )

    result = GatedResult(
        ok=bool(held),
        object=object_name,
        stopped_at=None if held else G5_VERIFY,
        reason="" if held else _missed(object_name),
        confidence=confidence,
        vision_confidence=detection.confidence,
        zone=zone.id,
        zone_label=zone.label,
        clarified=clarified,
        approval_required=needs_approval,
        approved=approved,
        timed_out=timed_out,
        moved=True,
        verified=held,
        records=tuple(records),
    )
    log_event("gated_pick", **result.as_log())
    return result


def _missed(object_name: str) -> str:
    """How to admit a failed action, in whichever mode we are running."""
    if config.PICK_MODE == "place":
        return f"I ran the move but I couldn't move the {object_name}."
    return f"I ran the pick but I don't think I got the {object_name}."


def _carry(fields: dict) -> dict:
    """Pull the GatedResult fields out of a stop()'s extra data."""
    known = (
        "confidence", "vision_confidence", "zone", "zone_label", "clarified",
        "approval_required", "approved", "timed_out",
    )
    return {key: fields[key] for key in known if key in fields}


def _resolve_ambiguity(
    object_name: str,
    detection: eyes.Detection,
    match: zones.ZoneMatch,
    zone_set: zones.ZoneSet,
    clarify: Clarify,
    records: list[GateRecord],
    *,
    frame=None,
    timeout_s: float = config.APPROVAL_TIMEOUT_S,
) -> zones.ZoneMatch | None:
    """Ask WHICH one, then resolve the answer to a zone IN PYTHON.

    The model relays the human's words; the matching from those words to a zone
    happens here. A model that replies "the left one" cannot talk us into a zone
    the human did not pick, and a reply that matches nothing is a stand-down
    rather than a guess.
    """
    if detection.candidates > 1:
        # Several things matched the name. locate() only reports HOW MANY, so
        # ask where they all are — otherwise we would offer the human a choice
        # between the winner and its nearest spot, which is a different question
        # from the one the scene is actually posing.
        options = _zones_holding(object_name, zone_set, frame)
        if len(options) < 2:
            # They are all on one spot (or we could not place them), so naming
            # spots cannot separate them. Fall back to the geometric candidates.
            options = [z for z in (match.zone, match.runner_up) if z is not None]
    else:
        options = [z for z in (match.zone, match.runner_up) if z is not None]

    if len(options) < 2:
        records.append(GateRecord(G2_AMBIGUOUS, False,
                                  "ambiguous but there is no second spot to offer",
                                  {"candidates": detection.candidates}))
        return None

    labels = [z.label for z in options]
    question = (
        f"I can see more than one thing that could be the {object_name}. "
        f"Which spot — {' or '.join(labels)}?"
        if detection.candidates > 1
        else f"That {object_name} is sitting between {' and '.join(labels)}. Which one?"
    )
    log_event("gate_clarify_asked", object=object_name, question=question, options=labels)

    answer = _ask_with_deadline(clarify, (question, labels), timeout_s)
    if answer is TIMED_OUT or not answer or not str(answer).strip():
        records.append(GateRecord(G2_AMBIGUOUS, False, "no answer to the clarification",
                                  {"question": question, "options": labels}))
        return None

    chosen = _match_zone_by_words(str(answer), options)
    log_event("gate_clarify_answered", answer=str(answer),
              resolved=None if chosen is None else chosen.id)
    if chosen is None:
        records.append(GateRecord(G2_AMBIGUOUS, False, f"could not resolve {answer!r} to a spot",
                                  {"question": question, "options": labels, "answer": str(answer)}))
        return None

    records.append(
        GateRecord(G2_AMBIGUOUS, True, f"human chose {chosen.label}",
                   {"question": question, "options": labels, "answer": str(answer),
                    "resolved": chosen.id})
    )
    # Re-resolve to a settled match on the chosen zone. The margin is no longer
    # what decides it — a human just did — so it is reported as clear.
    #
    # DELIBERATE CONSEQUENCE: an infinite margin means G4 sees full assignment
    # clarity, so a clarified pick often clears CONF_APPROVAL without a second
    # question. That is the intent — the human has already engaged with this
    # exact pick and named the spot, and asking "which one?" and then "are you
    # sure?" back to back is nagging, not safety. Vision confidence still applies
    # in full, so a pick that is unsure about WHAT it sees is still gated.
    return zones.ZoneMatch(
        zone=chosen,
        distance_px=match.distance_px if chosen is match.zone else match.runner_up_distance_px,
        runner_up=None,
        runner_up_distance_px=float("inf"),
        reason=f"resolved by the operator: {answer!r}",
    )


def _zones_holding(object_name: str, zone_set: zones.ZoneSet, frame) -> list[zones.Zone]:
    """Every marked spot that currently holds something matching the name.

    One extra vision call, made only on the ambiguous path, so the clarifying
    question names the spots the objects are actually on.
    """
    try:
        detections = eyes.list_visible([object_name], frame=frame)
    except eyes.EyesError as exc:
        log.warning("could not list candidates: %s", exc)
        return []

    found: list[zones.Zone] = []
    for detection in detections:
        placed = zones.assign_pixel(detection.point, zone_set)
        if placed.zone is not None and placed.zone not in found:
            found.append(placed.zone)
    return found


def _match_zone_by_words(answer: str, options: list[zones.Zone]) -> zones.Zone | None:
    """Map a spoken answer onto one of the offered zones, or None.

    Deliberately strict and dumb. It matches on the words that are actually in
    the zone labels plus a few obvious synonyms, and refuses anything ambiguous.
    Being unable to understand "um, that one" is the correct outcome: it becomes
    a stand-down, not a coin flip.
    """
    text = answer.strip().lower()
    if not text:
        return None

    scored: list[tuple[int, zones.Zone]] = []
    for zone in options:
        words = [w for w in zone.label.lower().replace("-", " ").split() if w]
        score = sum(1 for word in words if word in text)
        # "the first one" / "the second one" when the labels share no words.
        if zone is options[0] and any(w in text for w in ("first", "former")):
            score += 1
        if zone is options[-1] and any(w in text for w in ("second", "latter", "last")):
            score += 1
        if zone.id.lower() in text:
            score += 2
        scored.append((score, zone))

    best = max(score for score, _ in scored)
    if best == 0:
        return None
    winners = [zone for score, zone in scored if score == best]
    if len(winners) != 1:
        return None  # "left or right?" answered "left right" — refuse, don't guess
    return winners[0]


def _default_perform(arm) -> Perform:
    """Run the macro inline and read the gripper straight afterwards."""

    def perform(zone: zones.Zone) -> PerformOutcome:
        completed = pick.play_pick(arm, zone)
        return PerformOutcome(
            completed=completed,
            detail="" if completed else "the pick was stopped before it finished",
            gripper_percent=pick.read_gripper(arm),
        )

    return perform
