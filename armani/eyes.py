"""Perception. Gemini points at named objects in a C920 frame.

This module NEVER moves the arm and imports nothing from motion, kinematics or
grasp. It answers exactly one question — "where in this image is the thing you
named?" — and is honest about how sure it is.

On the confidence number
------------------------
Gemini's pointing API does not return a calibrated confidence, so we do not
pretend to have one. What ``locate`` reports is built only from things we can
actually observe:

* whether a point came back at all, across several independent queries
  (``found_rate``),
* how closely those independent queries agree on WHERE the object is
  (``agreement``) — this is the dual-query agreement signal from CLAUDE.md,
* the model's own stated confidence (``self_report``), which is a self-report
  and is treated as such: it is capped at half the score.

Independence is the whole point of the agreement term, so the queries use
differently worded prompts rather than the same prompt twice. Two identical
prompts would agree with themselves and the number would mean nothing.

The IK reachability margin — the third signal CLAUDE.md asks for — is folded in
by ``grasp.hover_over``, not here, because this module must not know about the
arm. The MCQA logprob check is not implemented; see the stage-4 report.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
from dataclasses import dataclass

from armani import config
from armani.logutil import get_logger, log_event

log = get_logger("eyes")

# What we assume when the model does not state a confidence. Neutral on purpose:
# silence is neither evidence for nor against.
DEFAULT_SELF_REPORT = 0.5
# With a single query there is no second opinion to agree with, so the agreement
# term cannot be earned. Neutral, never 1.0 — that would be a free high score.
NO_AGREEMENT_EVIDENCE = 0.5


class EyesError(RuntimeError):
    """The camera or the vision model could not be used at all.

    Distinct from "the object is not in the frame", which is a normal answer and
    comes back as None.
    """


@dataclass(frozen=True)
class Detection:
    """Where one named object is, and how much of that we actually believe."""

    label: str
    point: tuple[int, int]  # pixel (x, y) in the captured frame
    confidence: float  # 0..1, see the module docstring
    frame_size: tuple[int, int]  # (width, height) the point is valid for
    model: str  # which Gemini model answered
    raw: str = ""  # the model's reply verbatim, for the decision log
    samples: int = 1  # how many independent queries were asked
    found_rate: float = 1.0  # fraction of those that saw it
    agreement: float = 1.0  # 1.0 = the queries picked the same pixel
    self_report: float = 0.5  # what the model said about itself
    # Most points any single query returned for this one named object. More than
    # one means the model saw several things matching the name — the raw material
    # for stage 6's "which one did you mean?" gate. Recorded rather than silently
    # discarded, because dropping it is how an ambiguous scene looks certain.
    candidates: int = 1

    def as_log(self) -> dict[str, object]:
        """Compact form for logs/decisions.jsonl (raw reply truncated)."""
        return {
            "label": self.label,
            "point": list(self.point),
            "confidence": round(self.confidence, 3),
            "model": self.model,
            "samples": self.samples,
            "found_rate": round(self.found_rate, 3),
            "agreement": round(self.agreement, 3),
            "self_report": round(self.self_report, 3),
            "candidates": self.candidates,
            "raw": self.raw[:400],
        }


# --- Camera --------------------------------------------------------------
# Same path smoke_03 proved: AVFoundation backend, 640x480, explicit index.

# The C920 needs a few frames before auto-exposure and auto-focus settle. The
# first frame off a cold open is routinely dark or blurred, and a blurred frame
# is exactly the input that makes a vision model point at the wrong object.
CAMERA_WARMUP_FRAMES = 5


def capture_frame(index: int | None = None):
    """Grab one settled frame from the C920. Returns a BGR numpy array.

    Raises EyesError with an actionable message rather than returning None: a
    missing camera is an operator problem, not a perception result.
    """
    import cv2

    if index is None:
        index = config.CAMERA_INDEX
    if index is None:
        raise EyesError(
            "ARMANI_CAMERA_INDEX is not set and no index was given. "
            "Run tests/smoke_03_camera.py to find the C920's index, then put it in .env."
        )

    capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        if not capture.isOpened():
            raise EyesError(
                f"camera index {index} would not open. Check the C920 is plugged in, that no "
                "other app (Zoom, Photo Booth, OBS) holds it, and that macOS Camera access is "
                "granted to your terminal (System Settings > Privacy & Security > Camera)."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        frame = None
        for _ in range(CAMERA_WARMUP_FRAMES):
            read_ok, candidate = capture.read()
            if read_ok and candidate is not None:
                frame = candidate
        if frame is None:
            raise EyesError(f"camera index {index} opened but delivered no frame")
    finally:
        capture.release()

    _publish_frame(frame)

    height, width = frame.shape[:2]
    if (width, height) != (config.CAMERA_WIDTH, config.CAMERA_HEIGHT):
        # Not fatal — points are normalised — but the homography was computed at
        # one specific frame size and silently changing it would shift every
        # mapped coordinate.
        log.warning(
            "camera delivered %dx%d, expected %dx%d; the homography is only valid "
            "at the size it was calibrated at",
            width, height, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
        )
    return frame


def _publish_frame(frame) -> None:
    """Drop the latest frame where the dashboard can pick it up.

    Best-effort and silent by design: this exists so a screen can show what the
    robot saw, and a full disk or a missing directory must never take down
    perception. The dashboard reads this file instead of opening the camera
    itself, so the two never fight over the C920 mid-demo.
    """
    try:
        import cv2

        config.LAST_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(config.LAST_FRAME_PATH), frame)
    except Exception as exc:
        log.debug("could not publish the frame for the dashboard: %s", exc)


def encode_jpeg(frame) -> bytes:
    import cv2

    encoded, buffer = cv2.imencode(".jpg", frame)
    if not encoded:
        raise EyesError("could not JPEG-encode the frame")
    return buffer.tobytes()


# --- Prompting -----------------------------------------------------------
# Gemini robotics-ER returns points normalised 0-1000 as [y, x] (CLAUDE.md).
# Two differently worded prompts, so that asking twice is a real second opinion
# rather than the same question echoed back.

# NOTE on "confidence": Gemini's pointing API does NOT return a score. The
# documented reply is only [{"point": [y, x], "label": ...}]. We ask for a
# confidence anyway, but that makes it a model SELF-REPORT invented on request,
# not a calibrated probability from the API — which is exactly why it is capped
# at half the final score and defaults to neutral when absent.
_POINT_FORMAT = (
    'Reply with JSON only: [{"point": [y, x], "label": "<name>", "confidence": <0..1>}]. '
    "Points are [y, x] normalised to 0-1000. "
    '"confidence" is how sure you are that this is really the object. '
    "If the object is not visible, reply with an empty array []."
)

PROMPT_VARIANTS: tuple[str, ...] = (
    "Point to the {objects} in the image. " + _POINT_FORMAT,
    "Locate the {objects}. Give the single point at the centre of each one, "
    "on the part a robot gripper should grasp. " + _POINT_FORMAT,
)

# Spike S1: the camera views the table at an angle, so a point on an object's
# visual CENTRE sits above the table plane and the homography maps it to a spot
# BEHIND the object (parallax). Asking for the point where the object meets the
# table removes most of that bias. Selected by locate(contact_point=True); the
# default behaviour is unchanged.
CONTACT_PROMPT_VARIANTS: tuple[str, ...] = (
    "Point to where the {objects} TOUCHES THE TABLE — the spot on the table "
    "surface at the base of the object, where it meets the table, NOT its top "
    "or its visual centre. " + _POINT_FORMAT,
    "Find the {objects}. Give the single point on the TABLE SURFACE directly "
    "under the object, at its contact point with the table (the base, not the "
    "top). " + _POINT_FORMAT,
)

# Deliberately NOT str.format: these prompts contain literal JSON braces, and
# format() reads {"point"} as a replacement field and raises KeyError. That bug
# broke every call to locate() until a unit test caught it.
OBJECTS_PLACEHOLDER = "{objects}"


def render_prompt(template: str, objects: str) -> str:
    return template.replace(OBJECTS_PLACEHOLDER, objects)


def _describe(targets: list[str]) -> str:
    if len(targets) == 1:
        return targets[0]
    return ", ".join(targets[:-1]) + f" and {targets[-1]}"


def _extract_json(raw: str) -> object:
    """Pull a JSON array out of a reply that may be fenced or padded with prose.

    Same spirit as improvise._json_candidates, kept local and much smaller:
    this one only ever expects a list of point objects, and eyes must not
    depend on the gesture stack.
    """
    text = raw.strip()
    if not text:
        raise ValueError("empty reply")

    candidates: list[str] = []
    if "```" in text:
        for index, block in enumerate(text.split("```")):
            if index % 2 == 1:  # inside a fence
                head, newline, rest = block.partition("\n")
                candidates.append(rest if newline and head.strip().isalpha() else block)
    candidates.append(text)
    # Widest bracketed span, which survives "Here you go: [{...}]".
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped or stripped[0] not in "[{":
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            last_error = exc
    detail = f" ({last_error})" if last_error is not None else ""
    raise ValueError(f"no JSON array in the reply{detail}; began: {raw[:160]!r}")


def _parse_points(raw: str, frame_size: tuple[int, int]) -> list[tuple[str, tuple[int, int], float]]:
    """Reply -> [(label, (x_px, y_px), self_confidence)]. Bad entries are skipped.

    Defensive by design: a model that returns nine good points and one malformed
    one should give us nine points, not an exception.
    """
    payload = _extract_json(raw)
    if isinstance(payload, dict):
        for key in ("points", "detections", "objects"):
            if key in payload:
                payload = payload[key]
                break
        else:
            payload = [payload]  # a single bare {"point": ...} object
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON array, got {type(payload).__name__}")

    width, height = frame_size
    results: list[tuple[str, tuple[int, int], float]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        point = item.get("point")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            y_norm, x_norm = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x_norm) and math.isfinite(y_norm)):
            continue
        # Normalised 0-1000 -> pixels. Clamped rather than rejected: a model
        # that says 1002 means "at the edge", and the homography check
        # downstream is what decides whether the spot is usable.
        x_px = int(round(min(max(x_norm, 0.0), 1000.0) / 1000.0 * (width - 1)))
        y_px = int(round(min(max(y_norm, 0.0), 1000.0) / 1000.0 * (height - 1)))

        raw_conf = item.get("confidence")
        try:
            self_conf = float(raw_conf)
        except (TypeError, ValueError):
            self_conf = DEFAULT_SELF_REPORT
        if not math.isfinite(self_conf):
            self_conf = DEFAULT_SELF_REPORT
        self_conf = min(max(self_conf, 0.0), 1.0)

        label = str(item.get("label") or "").strip()
        results.append((label, (x_px, y_px), self_conf))
    return results


# --- Model calls ---------------------------------------------------------

_client_lock = threading.Lock()
_cached_client = None


def _client():
    """One genai client for the process. Building it per query is pure latency."""
    global _cached_client
    with _client_lock:
        if _cached_client is not None:
            return _cached_client
        key = config.api_key("GOOGLE_API_KEY")
        if key is None:
            raise EyesError("GOOGLE_API_KEY is not set (see .env.example)")
        try:
            from google import genai
        except ImportError as exc:
            raise EyesError(f"google-genai is not installed: {exc}") from exc
        _cached_client = genai.Client(api_key=key)
        return _cached_client


def _ask(frame_jpeg: bytes, prompt: str) -> tuple[str, str]:
    """One pointing query. Returns (model_name, raw_reply).

    Walks config.GEMINI_MODELS in order so a preview model being retired
    degrades to the next one instead of taking the demo down.

    Deliberately does NOT set response_mime_type/response_schema: these are
    preview robotics models tuned for a specific pointing reply, and forcing a
    structured-output mode they may not support would turn a working model into
    a hard failure. The prompt asks for JSON and _extract_json copes with prose
    and fences around it — the same bargain improvise.py makes with Claude.
    """
    from google.genai import types

    client = _client()
    image = types.Part.from_bytes(data=frame_jpeg, mime_type="image/jpeg")

    base = {
        "temperature": config.EYES_TEMPERATURE,
        "max_output_tokens": config.EYES_MAX_OUTPUT_TOKENS,
    }
    # Google's robotics docs recommend disabling thinking for pointing: it is a
    # perception call and the budget is latency paid on every query. Not every
    # model in the fallback list necessarily accepts ThinkingConfig, so a model
    # that rejects it is retried plainly rather than being written off.
    attempts = (
        {**base, "thinking_config": types.ThinkingConfig(thinking_budget=0)},
        base,
    )

    errors: list[str] = []
    quota_hit = False
    for model in config.GEMINI_MODELS:
        for settings in attempts:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(**settings),
                )
            except Exception as exc:
                errors.append(f"{model}: {_short_error(exc)}")
                log.warning("gemini model %s failed: %s", model, _short_error(exc))
                if _is_quota_error(exc):
                    # Retrying the same model with different settings cannot fix
                    # a spent quota, and burning attempts against a rate limit is
                    # how a demo turns a slow API into no API at all.
                    quota_hit = True
                    break
                if not _looks_like_bad_settings(exc):
                    break  # the model is unhappy about something else; move on
                continue
            text = (response.text or "").strip()
            if not text:
                errors.append(f"{model}: empty reply")
                continue
            return model, text

    if quota_hit:
        raise EyesError(
            "Gemini quota is exhausted. The free tier allows only 20 requests per day "
            "PER MODEL, and one gated pick spends three or four. Enable billing on the "
            "Google API key before the demo — see docs/env_report.md."
        )
    raise EyesError("every Gemini model failed: " + "; ".join(errors))


# Aggregated model errors end up in VerifyResult.reason and from there in the
# decision log. A single 429 from google-genai is ~1.5 kB of JSON, so six of them
# would bury one log line under 9 kB and make the stage-7 dashboard unreadable.
MAX_ERROR_CHARS = 160


def _short_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    if len(text) > MAX_ERROR_CHARS:
        text = text[:MAX_ERROR_CHARS] + "..."
    return f"{type(exc).__name__}: {text}"


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def _looks_like_bad_settings(exc: Exception) -> bool:
    """Is this the model rejecting our config, rather than a real failure?

    Only then is it worth re-asking the same model without ThinkingConfig.
    """
    text = str(exc).lower()
    return any(word in text for word in ("thinking", "unsupported", "invalid argument", "not supported"))


# --- Scoring -------------------------------------------------------------


def score_detection(
    found_rate: float, agreement: float, self_report: float
) -> float:
    """Combine the three observable signals into 0..1.

    ``found_rate`` multiplies rather than averages: a query that could not see
    the object at all is much stronger evidence than a slightly different pixel,
    and it should drag the score down hard.
    """
    weight = config.EYES_SELF_REPORT_WEIGHT
    blended = weight * self_report + (1.0 - weight) * agreement
    return round(min(max(blended * found_rate, 0.0), 1.0), 3)


def _agreement(points: list[tuple[int, int]]) -> float:
    """1.0 when the queries picked the same pixel, 0.0 when far apart.

    Uses the widest disagreement, not the average: with three queries, two
    agreeing does not excuse the third pointing at a different object.
    """
    if len(points) < 2:
        return NO_AGREEMENT_EVIDENCE
    spread = max(
        math.dist(a, b)
        for index, a in enumerate(points)
        for b in points[index + 1 :]
    )
    ceiling = 2.0 * config.EYES_AGREEMENT_PX
    return round(max(0.0, min(1.0, 1.0 - spread / ceiling)), 3)


# --- Public API ----------------------------------------------------------


def locate(
    object_name: str,
    frame=None,
    samples: int | None = None,
    contact_point: bool = False,
) -> Detection | None:
    """Find one named object. None means "not seen", which is a normal answer.

    Asks ``samples`` independently-worded questions and fuses the answers; see
    the module docstring for what the confidence means.

    ``contact_point`` (Spike S1) asks for the point where the object meets the
    TABLE rather than its visual centre, which reduces parallax bias for a
    homography-based reach. It defaults to False — the demo path is unchanged.
    """
    if not object_name or not object_name.strip():
        raise ValueError("object_name is empty")
    object_name = object_name.strip()
    if samples is None:
        samples = config.EYES_SAMPLES
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")

    if frame is None:
        frame = capture_frame()
    height, width = frame.shape[:2]
    frame_size = (width, height)
    jpeg = encode_jpeg(frame)

    variants = CONTACT_PROMPT_VARIANTS if contact_point else PROMPT_VARIANTS
    points: list[tuple[int, int]] = []
    self_reports: list[float] = []
    raws: list[str] = []
    model_used = ""
    most_candidates = 0

    for index in range(samples):
        prompt = render_prompt(variants[index % len(variants)], object_name)
        model_used, raw = _ask(jpeg, prompt)
        raws.append(raw)
        try:
            parsed = _parse_points(raw, frame_size)
        except ValueError as exc:
            # A reply we cannot read is evidence of nothing, so it counts as a
            # miss rather than taking the whole lookup down.
            log.warning("unparseable reply from %s: %s", model_used, exc)
            continue
        if not parsed:
            continue
        most_candidates = max(most_candidates, len(parsed))
        if len(parsed) > 1:
            log.info(
                "%d things matched %r in one query — ambiguous scene (stage 6 gate G2)",
                len(parsed), object_name,
            )
        # One object was asked for; if several came back, the first is the
        # model's own best answer. The count is kept on the Detection so an
        # ambiguous scene cannot silently read as a confident single answer.
        _, point, self_conf = parsed[0]
        points.append(point)
        self_reports.append(self_conf)

    found_rate = len(points) / samples
    if not points:
        log_event("eyes_locate", object=object_name, found=False, samples=samples, model=model_used)
        return None

    # Median per axis: with three or more queries this ignores a single outlier
    # entirely, and with two it is the midpoint.
    point = (
        int(statistics.median(p[0] for p in points)),
        int(statistics.median(p[1] for p in points)),
    )
    agreement = _agreement(points)
    self_report = statistics.fmean(self_reports)

    detection = Detection(
        label=object_name,
        point=point,
        confidence=score_detection(found_rate, agreement, self_report),
        frame_size=frame_size,
        model=model_used,
        raw="\n---\n".join(raws),
        samples=samples,
        found_rate=found_rate,
        agreement=agreement,
        self_report=self_report,
        candidates=max(1, most_candidates),
    )
    log_event("eyes_locate", object=object_name, found=True, **detection.as_log())
    return detection


def list_visible(candidates: list[str], frame=None) -> list[Detection]:
    """Point at several named objects in ONE query.

    Built for the stage-6 "which one did you mean?" disambiguation; nothing
    calls it yet. Because it is a single query there is no second opinion, so
    every confidence here carries the neutral agreement term — these scores are
    not comparable with locate()'s and must not be used to gate motion.
    """
    if not candidates:
        return []
    cleaned = [name.strip() for name in candidates if name and name.strip()]
    if not cleaned:
        raise ValueError("candidates contained no usable names")

    if frame is None:
        frame = capture_frame()
    height, width = frame.shape[:2]
    frame_size = (width, height)

    prompt = render_prompt(PROMPT_VARIANTS[0], _describe(cleaned))
    model_used, raw = _ask(encode_jpeg(frame), prompt)
    try:
        parsed = _parse_points(raw, frame_size)
    except ValueError as exc:
        log.warning("unparseable reply from %s: %s", model_used, exc)
        log_event("eyes_list", candidates=cleaned, found=0, error=str(exc))
        return []

    detections = [
        Detection(
            label=label or cleaned[0],
            point=point,
            confidence=score_detection(1.0, NO_AGREEMENT_EVIDENCE, self_conf),
            frame_size=frame_size,
            model=model_used,
            raw=raw,
            samples=1,
            found_rate=1.0,
            agreement=NO_AGREEMENT_EVIDENCE,
            self_report=self_conf,
        )
        for label, point, self_conf in parsed
    ]
    log_event(
        "eyes_list",
        candidates=cleaned,
        found=len(detections),
        detections=[d.as_log() for d in detections],
    )
    return detections


# --- Open-vocabulary scene survey ----------------------------------------

SCENE_PROMPT = (
    "List the distinct physical objects sitting on the table surface in front of you. "
    "Ignore the background, walls, people, cables, the plant, and the robot arm itself. "
    'Reply as a JSON array of short lowercase object names, e.g. '
    '["wooden log","charger","red block"].'
)

# A scene reply is a handful of short names. A much longer list means the model
# started describing the room, and passing fifty "objects" to the voice agent
# would have it reading an inventory at the audience.
MAX_SCENE_OBJECTS = 12
MAX_SCENE_NAME_CHARS = 40


def describe_scene(frame=None) -> list[str]:
    """Everything on the table, in the model's own words. Open vocabulary.

    Unlike ``locate``/``list_visible``, this is not restricted to
    ``config.OBJECT_CATALOG`` — it answers "what IS this?" rather than "where is
    the thing I named?". That is the right question for "what's on the table?"
    and the wrong one for a pick, which still needs a point.

    Returns names only; there are no coordinates to assign to a zone. Never
    raises: nothing seen, an unparseable reply and a dead network all come back
    as an empty list, because a scene survey failing must not take down a turn.
    """
    try:
        if frame is None:
            frame = capture_frame()
        model, raw = _ask(encode_jpeg(frame), SCENE_PROMPT)
    except Exception as exc:
        log.warning("scene survey failed: %s", exc)
        return []

    try:
        payload = _extract_json(raw)
    except ValueError as exc:
        log.warning("unreadable scene reply from %s: %s", model, exc)
        return []

    # Tolerate {"objects": [...]} as well as a bare array — same latitude the
    # pointing parser gives, for the same reason.
    if isinstance(payload, dict):
        for key in ("objects", "items", "things"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        log.warning("scene reply was %s, not a list", type(payload).__name__)
        return []

    names: list[str] = []
    for entry in payload:
        # A model that returns [{"name": "charger"}] is not wrong enough to bin.
        if isinstance(entry, dict):
            entry = entry.get("name") or entry.get("object") or entry.get("label")
        if not isinstance(entry, str):
            continue
        name = " ".join(entry.split()).strip().lower()[:MAX_SCENE_NAME_CHARS]
        if name and name not in names:
            names.append(name)
        if len(names) >= MAX_SCENE_OBJECTS:
            break

    log_event("eyes_describe_scene", model=model, count=len(names), objects=names)
    return names


# --- Grasp verification (trust gate G5) ----------------------------------


@dataclass(frozen=True)
class HeldCheck:
    """Did the arm actually end up holding the thing? The VLM's opinion."""

    held: bool | None  # None = the model would not commit
    confidence: float
    reason: str
    raw: str = ""
    model: str = ""

    def as_log(self) -> dict[str, object]:
        return {
            "vlm_held": self.held,
            "vlm_confidence": round(self.confidence, 3),
            "vlm_reason": self.reason,
            "vlm_model": self.model,
        }


# Two prompts, because "did it work?" means different things depending on what
# the operator recorded. See config.PICK_MODE.
HELD_PROMPT = (
    "Look at this robot arm and table. The robot just tried to pick up the {object}.\n"
    "Two things decide the answer: is the {object} held in the robot's gripper, and is it "
    "gone from the spot on the table where it was sitting?\n"
    'Reply with JSON only: {"held": true or false, "confidence": 0.0 to 1.0, '
    '"reason": "one short sentence"}.\n'
    "Answer held=false if the gripper is empty or the object is still on the table. "
    "Be strict: if you cannot tell, say so in the reason and give a low confidence."
)

# Place mode: the arm has already let go. An empty gripper is the SUCCESS case,
# so the question is only about the marked spot the object started on.
PLACED_PROMPT = (
    "Look at this robot arm and table. The robot just tried to move the {object} "
    "from its marked spot on the table into the tray.\n"
    "Question: is the {object} GONE from its marked spot on the table?\n"
    'Reply with JSON only: {"held": true or false, "confidence": 0.0 to 1.0, '
    '"reason": "one short sentence"}.\n'
    "Answer held=true if the spot where the {object} was sitting is now empty, or you can "
    "see the {object} in the tray. Answer held=false if the {object} is still sitting on "
    "the table where it started.\n"
    "The robot's gripper being empty is EXPECTED and does not mean failure — it has "
    "already let go. Judge only by whether the {object} has left its spot. "
    "If you cannot tell, say so in the reason and give a low confidence."
)


def _verify_prompt() -> str:
    return PLACED_PROMPT if config.PICK_MODE == "place" else HELD_PROMPT


def confirm_held(object_name: str, frame=None) -> HeldCheck:
    """Ask Gemini whether the action actually worked.

    What that means depends on config.PICK_MODE: in "hold" mode the object must
    be in the jaws; in "place" mode — what the recorded macros actually do — it
    must be GONE from its marked spot, and an empty gripper is expected.
    ``held`` is the answer to whichever question was asked.

    This is the real G5 check. It never raises: verification runs while the arm
    may still be holding something, so a network failure must come back as "I
    could not tell" rather than as an exception in the middle of a grasp.
    """
    object_name = (object_name or "").strip() or "object"
    try:
        if frame is None:
            frame = capture_frame()
        model, raw = _ask(encode_jpeg(frame), _verify_prompt().replace("{object}", object_name))
    except Exception as exc:
        return HeldCheck(None, 0.0, f"could not run the visual check: {exc}")

    try:
        payload = _extract_json(raw)
    except ValueError as exc:
        return HeldCheck(None, 0.0, f"unreadable reply: {exc}", raw=raw, model=model)

    # The prompt asks for an object; a model that wraps it in a list is not wrong
    # enough to throw away.
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return HeldCheck(None, 0.0, f"expected a JSON object, got {type(payload).__name__}",
                         raw=raw, model=model)

    held = payload.get("held")
    if not isinstance(held, bool):
        held = None

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    reason = str(payload.get("reason") or "").strip() or "no reason given"
    check = HeldCheck(held=held, confidence=confidence, reason=reason, raw=raw, model=model)
    log_event("eyes_confirm_held", object=object_name, **check.as_log())
    return check


def annotate(frame, detections: list[Detection]):
    """Draw detections on a COPY of the frame, for the operator to eyeball."""
    import cv2

    canvas = frame.copy()
    for detection in detections:
        x, y = detection.point
        colour = (0, 255, 0) if detection.confidence >= config.EYES_CONF_THRESHOLD else (0, 165, 255)
        cv2.drawMarker(canvas, (x, y), colour, cv2.MARKER_CROSS, 24, 2)
        cv2.circle(canvas, (x, y), 14, colour, 2)
        cv2.putText(
            canvas,
            f"{detection.label} {detection.confidence:.2f}",
            (max(0, x - 60), max(18, y - 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
        )
    return canvas
