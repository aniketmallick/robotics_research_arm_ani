#!/usr/bin/env python
"""Smoke 08 — the realtime voice agent.

Dry run: build the agent against a DryRunArm, verify all six tools register with
their schemas, then run ONE text round trip ("introduce yourself in one
sentence") and print the reply. No audio, no motion, no microphone. Needs
OPENAI_API_KEY for the round trip; without it the build/schema check still runs
and the round trip SKIPs.

Live: hand the operator the scripted checklist and launch the real voice session
(scripts/run_agent.py) — that is where audio and motion actually happen.
"""

from __future__ import annotations

import asyncio

from _bootstrap import banner, fail, ok, parse_args, skip  # isort: skip

from armani import agent, config, motion  # noqa: E402
from armani.logutil import log_event  # noqa: E402

EXPECTED_TOOLS = {"list_gestures", "play_gesture", "improvise_move", "go_home", "get_status", "stop_motion"}


def _tool_names(tools: list) -> set[str]:
    return {getattr(t, "name", getattr(t, "__name__", "?")) for t in tools}


def check_schema() -> tuple[bool, str]:
    """Build the agent with a fake arm and confirm every tool registered."""
    worker = agent.MotionWorker(motion.DryRunArm(), motion_enabled=False)
    built = agent.build_agent(worker)
    names = _tool_names(built.tools)
    missing = EXPECTED_TOOLS - names
    extra = names - EXPECTED_TOOLS
    if missing or extra:
        return False, f"tool mismatch: missing={sorted(missing)} extra={sorted(extra)}"
    return True, f"all {len(EXPECTED_TOOLS)} tools registered: {', '.join(sorted(names))}"


async def _text_round_trip() -> str:
    """One no-audio turn through the real model. Returns ARM-ANI's reply."""
    worker = agent.MotionWorker(motion.DryRunArm(), motion_enabled=False)
    worker.start()
    try:
        async with await agent.build_session(worker, text_only=True) as session:
            await session.send_message("Introduce yourself in one sentence.")
            await agent.request_response(session)
            return await agent.collect_text_reply(session, timeout=30.0)
    finally:
        worker.shutdown()


def main() -> int:
    args = parse_args(__doc__ or "")
    banner("Smoke 08: realtime voice agent")

    passed, note = check_schema()
    print(f"  tool schemas: {'OK' if passed else 'FAIL'} — {note}")
    if not passed:
        return fail(note)

    if not args.dry_run:
        # Live mode is the real session; hand it to the operator explicitly
        # rather than trying to drive audio from a test harness.
        print(
            "\nLive voice session is run directly, with the operator:\n"
            "    python scripts/run_agent.py\n"
            "Follow the printed checklist: introduce, list gestures, take a bow\n"
            "(talk-while-moving), improvise, decline 'pick up the banana', barge-in,\n"
            "'stop' mid-gesture, then Ctrl-C freeze and [s] recovery.\n"
        )
        return ok("agent built; run scripts/run_agent.py for the witnessed live session")

    if config.api_key("OPENAI_API_KEY") is None:
        return skip("OPENAI_API_KEY not set — schema check passed; set the key for the text round trip")

    print("\n  text round trip (no audio)...")
    try:
        reply = asyncio.run(_text_round_trip())
    except Exception as exc:
        return fail(f"text round trip failed: {type(exc).__name__}: {exc}")

    log_event("smoke_08", reply=reply)
    if not reply.strip():
        return fail("model returned an empty reply")
    print(f"  ARM-ANI> {reply}")
    return ok("agent built, tools registered, and the model answered a text turn")


if __name__ == "__main__":
    raise SystemExit(main())
