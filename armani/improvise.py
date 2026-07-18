"""Claude writes novel moves; we trust none of it (safety rule 8).

The model returns JSON keyframes. Everything after that is our code: strict
schema validation, then the `policy` clamp profile — the conservative envelope,
because this is LLM-originated motion, not a human recording — then ordinary
interpolated `goto`s inside SafeMotion.

A rejected plan is a normal outcome, not an error. The model gets one retry with
the validation error appended, then we give up and say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from armani import config, safety
from armani.logutil import get_logger, log_event

log = get_logger("improvise")

Action = dict[str, float]

SYSTEM_PROMPT = f"""You choreograph short moves for a 6-DOF SO-101 robot arm.

Reply with JSON ONLY — no prose, no markdown fences. A JSON array of keyframes:

[{{"pose": {{"shoulder_pan": 10.0, "wrist_flex": -20.0}}, "seconds": 1.0}}]

Rules:
- At most {config.IMPROVISE_MAX_KEYFRAMES} keyframes.
- "seconds" is the time to reach that pose, between {config.IMPROVISE_MIN_SECONDS} and {config.IMPROVISE_MAX_SECONDS}.
- "pose" maps joint names to targets. Include only the joints you want to move.
- Joints and their safe ranges (degrees, except gripper which is 0-100 percent):
{chr(10).join(f"    {j}: {lo:g} to {hi:g}" for j, (lo, hi) in config.JOINT_LIMITS.items())}
- Neutral is roughly all zeros with the gripper near 50.
- Stay well inside the ranges. Expressive and readable beats extreme.
"""


class ImproviseError(RuntimeError):
    """The model did not produce a usable plan."""


@dataclass(frozen=True)
class Keyframe:
    pose: Action
    seconds: float


def build_prompt(description: str) -> str:
    return f"Choreograph this move: {description}"


def extract_json(raw: str) -> object:
    """Pull JSON out of a reply that may be fenced or padded with prose."""
    text = raw.strip()
    if "```" in text:
        # Take the largest fenced block; strip an optional language tag.
        blocks = text.split("```")
        text = max((b for i, b in enumerate(blocks) if i % 2 == 1), key=len, default=text)
        if "\n" in text:
            head, _, rest = text.partition("\n")
            if head.strip().isalpha():
                text = rest
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        raise ImproviseError(f"no JSON found in the reply: {raw[:200]!r}")
    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}"))
    if end <= start:
        raise ImproviseError(f"truncated JSON in the reply: {raw[:200]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ImproviseError(f"reply is not valid JSON: {exc}") from None


def validate(payload: object) -> list[Keyframe]:
    """Turn parsed JSON into keyframes, or raise with a message the model can act on."""
    if isinstance(payload, dict):
        # Tolerate {"keyframes": [...]} — a common shape for models to emit.
        for key in ("keyframes", "moves", "frames"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list) or not payload:
        raise ImproviseError("expected a non-empty JSON array of keyframes")
    if len(payload) > config.IMPROVISE_MAX_KEYFRAMES:
        raise ImproviseError(
            f"{len(payload)} keyframes, maximum is {config.IMPROVISE_MAX_KEYFRAMES}"
        )

    keyframes: list[Keyframe] = []
    for index, item in enumerate(payload):
        where = f"keyframe {index}"
        if not isinstance(item, dict):
            raise ImproviseError(f"{where} is not an object")
        unknown_keys = set(item) - {"pose", "seconds"}
        if unknown_keys:
            raise ImproviseError(f"{where} has unexpected key(s) {sorted(unknown_keys)}")
        if "pose" not in item or "seconds" not in item:
            raise ImproviseError(f"{where} needs both 'pose' and 'seconds'")

        seconds = item["seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ImproviseError(f"{where}: 'seconds' must be a number")
        if not config.IMPROVISE_MIN_SECONDS <= seconds <= config.IMPROVISE_MAX_SECONDS:
            raise ImproviseError(
                f"{where}: 'seconds' is {seconds}, must be between "
                f"{config.IMPROVISE_MIN_SECONDS} and {config.IMPROVISE_MAX_SECONDS}"
            )

        pose = item["pose"]
        if not isinstance(pose, dict) or not pose:
            raise ImproviseError(f"{where}: 'pose' must be a non-empty object")
        clean: Action = {}
        for joint, value in pose.items():
            if joint not in config.JOINT_LIMITS:
                raise ImproviseError(
                    f"{where}: unknown joint {joint!r}; valid: {', '.join(config.JOINT_LIMITS)}"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ImproviseError(f"{where}: {joint} must be a number, got {value!r}")
            clean[joint] = float(value)

        # Clamp AFTER validation: a plan that needed clamping is still usable,
        # it just gets pulled inside the policy envelope.
        # clamp_action raises ValueError on non-finite values, and json.loads
        # happily parses the literals NaN/Infinity. Convert it so validate()
        # only ever raises ImproviseError — otherwise a model emitting NaN
        # escapes the retry loop and surfaces as an unhandled traceback.
        try:
            safe_pose = safety.clamp_action(clean, profile="policy")
        except ValueError as exc:
            raise ImproviseError(f"{where}: {exc}") from None
        keyframes.append(Keyframe(pose=safe_pose, seconds=float(seconds)))

    return keyframes


def request_plan(description: str) -> list[Keyframe]:
    """Ask Claude for a plan, validate it, retry once with the error attached."""
    key = config.api_key("ANTHROPIC_API_KEY")
    if key is None:
        raise ImproviseError("ANTHROPIC_API_KEY is not set")

    import anthropic

    client = anthropic.Anthropic(api_key=key)
    messages = [{"role": "user", "content": build_prompt(description)}]

    last_error: ImproviseError | None = None
    for attempt in range(config.IMPROVISE_MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            raise ImproviseError(f"Claude call failed: {type(exc).__name__}: {exc}") from None

        raw = "".join(block.text for block in response.content if block.type == "text")
        try:
            keyframes = validate(extract_json(raw))
            log_event(
                "improvise_plan",
                description=description,
                attempt=attempt,
                keyframes=[{"pose": k.pose, "seconds": k.seconds} for k in keyframes],
            )
            return keyframes
        except ImproviseError as exc:
            last_error = exc
            log.warning("attempt %d rejected: %s", attempt + 1, exc)
            log_event("improvise_rejected", description=description, attempt=attempt, error=str(exc))
            if attempt == config.IMPROVISE_MAX_RETRIES:
                break
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That was rejected: {exc}. Reply with corrected JSON only."},
            ]

    raise ImproviseError(f"no valid plan after {config.IMPROVISE_MAX_RETRIES + 1} attempts: {last_error}")


def describe_plan(keyframes: list[Keyframe]) -> str:
    lines = [f"{len(keyframes)} keyframes, {sum(k.seconds for k in keyframes):.1f}s total:"]
    for index, frame in enumerate(keyframes, 1):
        pose = " ".join(f"{j}={v:+.1f}" for j, v in sorted(frame.pose.items()))
        lines.append(f"  {index}. {frame.seconds:.1f}s  {pose}")
    return "\n".join(lines)


def perform(arm, keyframes: list[Keyframe]) -> None:
    """Execute a validated plan as sequential interpolated moves."""
    from armani import motion

    with safety.SafeMotion(arm, description="improvised move"):
        for index, frame in enumerate(keyframes, 1):
            if safety.stop_requested():
                log_event("improvise_aborted", at_keyframe=index)
                return
            log.info("keyframe %d/%d over %.1fs", index, len(keyframes), frame.seconds)
            motion.goto(arm, dict(frame.pose), frame.seconds, profile="policy")
    log_event("improvise_done", keyframes=len(keyframes))


def improvise(arm, description: str) -> list[Keyframe]:
    """Plan and perform. Returns the validated plan."""
    keyframes = request_plan(description)
    perform(arm, keyframes)
    return keyframes
