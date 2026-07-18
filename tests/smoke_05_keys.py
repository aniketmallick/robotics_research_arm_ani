#!/usr/bin/env python
"""Smoke 05 — the three API keys are live. Cheap calls only.

OpenAI    : list models (free).
Gemini    : one vision call on tests/out/frame.jpg, walking the fallback model
            list from CLAUDE.md and reporting which model actually answered.
Anthropic : a 1-token completion.

Keys are never printed — only whether they are set, and the result of using them.
"""

from __future__ import annotations

import numpy as np
from _bootstrap import banner, fail, ok, parse_args, skip

from armani import config
from armani.logutil import log_event

GEMINI_PROMPT = "List the objects you see. Reply with a short comma-separated list."


def _missing_keys() -> list[str]:
    return [name for name in config.API_KEY_VARS if config.api_key(name) is None]


def check_openai() -> tuple[bool, str]:
    key = config.api_key("OPENAI_API_KEY")
    if key is None:
        return False, "OPENAI_API_KEY not set"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        models = list(client.models.list())
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    names = {m.id for m in models}
    note = f"{len(names)} models visible"
    if config.REALTIME_MODEL in names:
        note += f"; {config.REALTIME_MODEL} available"
    else:
        note += f"; WARNING {config.REALTIME_MODEL} NOT in the list — check stage 3"
    return True, note


def _frame_bytes() -> tuple[bytes, str]:
    """The saved camera frame, or a synthetic image if smoke 03 has not run."""
    import cv2

    frame_path = config.TEST_OUT_DIR / "frame.jpg"
    if frame_path.is_file():
        return frame_path.read_bytes(), f"using {frame_path.name} from smoke 03"

    # A plain synthetic image still proves the vision endpoint answers.
    image = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(image, (200, 150), (440, 330), (40, 40, 200), -1)
    cv2.circle(image, (120, 120), 60, (40, 200, 40), -1)
    encoded, buffer = cv2.imencode(".jpg", image)
    if not encoded:
        raise RuntimeError("could not encode the synthetic test image")
    return buffer.tobytes(), "no frame.jpg — used a synthetic image (run smoke 03 for a real one)"


def check_gemini() -> tuple[bool, str]:
    key = config.api_key("GOOGLE_API_KEY")
    if key is None:
        return False, "GOOGLE_API_KEY not set"
    try:
        from google import genai
        from google.genai import types

        image_bytes, source_note = _frame_bytes()
        client = genai.Client(api_key=key)
    except Exception as exc:
        return False, f"setup failed: {type(exc).__name__}: {exc}"

    errors = []
    for model in config.GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    GEMINI_PROMPT,
                ],
            )
            reply = (response.text or "").strip().replace("\n", " ")
            print(f"    {model} -> {reply[:160]}")
            return True, f"model {model} answered ({source_note})"
        except Exception as exc:
            # Keep the message, not just the class: an auth failure and a
            # model-not-found are both ClientError and need different fixes.
            detail = str(exc).replace("\n", " ")[:120]
            errors.append(f"{model}: {type(exc).__name__}: {detail}")
            print(f"    {model} unavailable ({type(exc).__name__}: {detail})")
    return False, "no Gemini model answered — " + "; ".join(errors)


def check_anthropic() -> tuple[bool, str]:
    key = config.api_key("ANTHROPIC_API_KEY")
    if key is None:
        return False, "ANTHROPIC_API_KEY not set"
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"{config.ANTHROPIC_MODEL} responded"


def main() -> int:
    args = parse_args(__doc__ or "")
    banner("Smoke 05: API keys")

    print("Key presence (values are never printed):")
    for name in config.API_KEY_VARS:
        print(f"  {name}: {'set' if config.api_key(name) else 'MISSING'}")

    if args.dry_run:
        print("\n[dry-run] would call: OpenAI models.list, Gemini vision on frame.jpg, Anthropic 1-token ping")
        return ok("dry run complete")

    missing = _missing_keys()
    if len(missing) == len(config.API_KEY_VARS):
        return skip("no API keys set — copy .env.example to .env and fill them in")

    checks = (("OpenAI", check_openai), ("Gemini", check_gemini), ("Anthropic", check_anthropic))
    results: dict[str, tuple[bool, str]] = {}
    for label, check in checks:
        print(f"\n  {label}...")
        try:
            results[label] = check()
        except Exception as exc:  # a broken check must not hide the other two
            results[label] = (False, f"check crashed: {type(exc).__name__}: {exc}")
        passed, note = results[label]
        print(f"  {label}: {'OK' if passed else 'FAIL'} — {note}")

    log_event("smoke_05", results={k: {"ok": v[0], "note": v[1]} for k, v in results.items()})

    failed = [label for label, (passed, _) in results.items() if not passed]
    if failed:
        return fail(f"API check failed for: {', '.join(failed)}")
    return ok("all three APIs responded")


if __name__ == "__main__":
    raise SystemExit(main())
