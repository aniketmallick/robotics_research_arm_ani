#!/usr/bin/env python
"""Smoke 12 — the trust gates. The stage-6 deliverable.

    --dry-run   scripted end-to-end runs of the whole pipeline with injected
                vision, console clarify/approve callables and a stubbed macro.
                No camera, no network, no arm. Six scenarios, including the
                10-second stand-down. This is what doctor runs.

    (default)   real vision against the camera, real gates, stubbed macro. Shows
                the true confidence number for what is actually on your table
                without moving anything.

    --live      OPERATOR + HARDWARE. The three demo acts for real: a clean pick,
                an ambiguous one, and a low-confidence one that stands down.

Every run appends a gate-by-gate record to logs/decisions.jsonl. That log is the
audit trail the judges see and the thing stage 7 renders, so this test also
checks its shape.

    python tests/smoke_12_gates.py --dry-run
    python tests/smoke_12_gates.py --object "red block"
    python tests/smoke_12_gates.py --live
"""

from __future__ import annotations

import argparse
import json
import time

from _bootstrap import EXIT_SKIP, banner, fail, ok, skip

from armani import config, eyes, gates, motion, pick, safety, zones
from armani.logutil import log_event

DEFAULT_OBJECT = next(iter(config.OBJECT_CATALOG))

# Short enough to keep the test snappy, long enough that it is a real wait.
TEST_TIMEOUT_S = 1.0

FRAME = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

# Two spots 200 px apart: the midpoint is on the table but exactly between them.
SCRIPT_ZONES = zones.ZoneSet(
    zones=(
        zones.Zone("z1", "front-left", (200, 300), 0),
        zones.Zone("z2", "front-right", (400, 300), 1),
    ),
    frame_size=FRAME,
    created="smoke-12",
)
ON_Z1 = (202, 302)
BETWEEN = (300, 300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="scripted scenarios, nothing real")
    parser.add_argument("--live", action="store_true", help="OPERATOR + HARDWARE: the three acts")
    parser.add_argument("--object", default=DEFAULT_OBJECT, help="what to pick")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    banner("Smoke 12: trust gates")
    print(f"approval threshold : {config.CONF_APPROVAL:.0%}")
    print(f"stand-down timeout : {config.APPROVAL_TIMEOUT_S:.0f}s")

    if args.dry_run:
        return _scripted()
    if args.live:
        return _live(args)
    return _real_vision(args)


# --- scripted scenarios --------------------------------------------------


class Recorder:
    """Stands in for the macro. Records what it was asked to run, runs nothing."""

    def __init__(self, held: bool = True):
        self.zones: list[str] = []
        self.held = held

    def __call__(self, zone) -> gates.PerformOutcome:
        self.zones.append(zone.id)
        return gates.PerformOutcome(completed=True, gripper_percent=30.0)


def _detection(point=ON_Z1, confidence=0.95, candidates=1, label="red block"):
    return eyes.Detection(
        label=label, point=point, confidence=confidence,
        frame_size=FRAME, model="scripted", candidates=candidates,
    )


def _scripted() -> int:
    """Six scenarios through the real pipeline with everything else injected."""
    print("\n[dry-run] no camera, no network, no arm — scripted gate scenarios\n")

    original = (eyes.locate, eyes.capture_frame, eyes.list_visible,
                zones.load_zones, pick.macro_available, pick.verify_held)
    log_start = _log_size()
    results: list[tuple[str, bool, str]] = []

    try:
        zones.load_zones = lambda *a, **k: SCRIPT_ZONES
        gates.zones.load_zones = lambda *a, **k: SCRIPT_ZONES
        gates.pick.macro_available = lambda zone: True
        gates.eyes.capture_frame = lambda *a, **k: object()

        for scenario in (
            _clean_pick, _ambiguous, _low_confidence_approved,
            _low_confidence_timeout, _unseen, _verify_mismatch,
        ):
            name, passed, detail = scenario()
            results.append((name, passed, detail))
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            for line in detail.splitlines():
                print(f"         {line}")
    finally:
        (eyes.locate, eyes.capture_frame, eyes.list_visible,
         zones.load_zones, pick.macro_available, pick.verify_held) = original
        gates.eyes.locate, gates.eyes.capture_frame, gates.eyes.list_visible = original[:3]
        gates.zones.load_zones, gates.pick.macro_available, gates.pick.verify_held = original[3:]

    audit_ok, audit_detail = _check_audit_trail(log_start, expected_runs=len(results))
    results.append(("decision log is a clean gate-by-gate audit trail", audit_ok, audit_detail))
    print(f"  [{'PASS' if audit_ok else 'FAIL'}] decision log is a clean gate-by-gate audit trail")
    for line in audit_detail.splitlines():
        print(f"         {line}")

    failed = [name for name, passed, _ in results if not passed]
    print("\n[dry-run] the macro stub was the only thing that could have moved an arm.")
    if failed:
        return fail(f"{len(failed)} scenario(s) failed: {', '.join(failed)}")
    return ok(f"all {len(results)} gate scenarios passed")


def _see(detection, held: bool = True, candidates_at=None) -> None:
    """Point the injected vision at a scripted answer."""
    gates.eyes.locate = lambda *a, **k: detection
    gates.eyes.list_visible = lambda names, frame=None: list(candidates_at or [])
    gates.pick.verify_held = (
        lambda obj, gripper, frame=None, use_vlm=True: pick.VerifyResult(
            gripper_percent=gripper, held=held, reason="scripted"
        )
    )


def _boom(*args, **kwargs):
    raise AssertionError("this gate should not have been reached")


def _clean_pick() -> tuple[str, bool, str]:
    _see(_detection(confidence=0.95))
    macro = Recorder()
    result = gates.run_gated_pick(
        _arm(), "red block", clarify=_boom, approve=_boom,
        perform=macro, verify_vlm=False, approval_timeout_s=TEST_TIMEOUT_S,
    )
    passed = result.ok and macro.zones == ["z1"] and not result.approval_required
    return (
        "clean pick: every gate passes, nobody is asked anything",
        passed,
        f"confidence {result.confidence:.0%} >= {config.CONF_APPROVAL:.0%}, "
        f"ran {macro.zones}, verified={result.verified}",
    )


def _ambiguous() -> tuple[str, bool, str]:
    _see(_detection(point=BETWEEN))
    macro = Recorder()
    asked: list[str] = []

    def clarify(question, options):
        asked.append(question)
        return "front-right"  # the human's answer

    result = gates.run_gated_pick(
        _arm(), "red block", clarify=clarify, approve=lambda p, t: True,
        perform=macro, verify_vlm=False, approval_timeout_s=TEST_TIMEOUT_S,
    )
    passed = result.ok and result.clarified and macro.zones == ["z2"]
    return (
        "ambiguous: asks which spot, and the HUMAN'S answer picks the zone",
        passed,
        f"asked {asked[0] if asked else '(nothing)'!r}\n"
        f"answered 'front-right' -> ran {macro.zones} (z2 = front-right)",
    )


def _low_confidence_approved() -> tuple[str, bool, str]:
    _see(_detection(confidence=0.5))
    macro = Recorder()
    prompts: list[str] = []

    def approve(prompt, timeout_s):
        prompts.append(prompt)
        return True

    result = gates.run_gated_pick(
        _arm(), "red block", clarify=_boom, approve=approve,
        perform=macro, verify_vlm=False, approval_timeout_s=TEST_TIMEOUT_S,
    )
    passed = result.ok and result.approval_required and result.approved and macro.zones == ["z1"]
    return (
        "low confidence: states the number, asks, proceeds on yes",
        passed,
        f"{prompts[0] if prompts else '(never asked)'}\napproved -> ran {macro.zones}",
    )


def _low_confidence_timeout() -> tuple[str, bool, str]:
    """The gate that matters most: silence means the arm does not move."""
    _see(_detection(confidence=0.5))
    macro = Recorder()
    arm = _arm()

    def never_answers(prompt, timeout_s):
        time.sleep(TEST_TIMEOUT_S * 30)  # far past the deadline
        return True  # and would have said yes, far too late

    started = time.perf_counter()
    result = gates.run_gated_pick(
        arm, "red block", clarify=_boom, approve=never_answers,
        perform=macro, verify_vlm=False, approval_timeout_s=TEST_TIMEOUT_S,
    )
    elapsed = time.perf_counter() - started

    passed = (
        result.timed_out
        and result.stopped_at == gates.G4_CONFIDENCE
        and not result.moved
        and macro.zones == []
        and arm.sends == 0
        and elapsed < TEST_TIMEOUT_S * 5
    )
    return (
        "NO ANSWER: stands down after the deadline, with ZERO sends",
        passed,
        f"waited {elapsed:.2f}s (deadline {TEST_TIMEOUT_S:.1f}s), macro calls {len(macro.zones)}, "
        f"arm sends {arm.sends}\nreason: {result.reason}",
    )


def _unseen() -> tuple[str, bool, str]:
    _see(None)
    macro = Recorder()
    arm = _arm()
    result = gates.run_gated_pick(
        arm, "red block", clarify=_boom, approve=_boom,
        perform=macro, verify_vlm=False, approval_timeout_s=TEST_TIMEOUT_S,
    )
    passed = (
        result.stopped_at == gates.G1_SEEN
        and not result.moved
        and macro.zones == []
        and arm.sends == 0
    )
    return (
        "unseen: G1 stops it before anything else runs",
        passed,
        f"reason: {result.reason}",
    )


def _verify_mismatch() -> tuple[str, bool, str]:
    """The macro ran and the object is NOT held. Saying otherwise would be a lie."""
    _see(_detection(confidence=0.95), held=False)
    macro = Recorder()
    result = gates.run_gated_pick(
        _arm(), "red block", clarify=_boom, approve=_boom,
        perform=macro, verify_vlm=False, approval_timeout_s=TEST_TIMEOUT_S,
    )
    passed = (
        not result.ok
        and result.moved
        and result.verified is False
        and result.stopped_at == gates.G5_VERIFY
    )
    return (
        "G5 mismatch: it moved, it did not get it, and it admits both",
        passed,
        f"moved={result.moved} verified={result.verified}\nsays: {result.speak()}",
    )


def _arm():
    return motion.DryRunArm()


# --- the audit trail -----------------------------------------------------


def _log_size() -> int:
    try:
        return config.DECISION_LOG.stat().st_size
    except OSError:
        return 0


def _check_audit_trail(offset: int, expected_runs: int) -> tuple[bool, str]:
    """Every run must leave one readable, ordered gate record behind."""
    try:
        with config.DECISION_LOG.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            lines = [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        return False, f"could not read {config.DECISION_LOG}: {exc}"

    runs = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError as exc:
            return False, f"decision log has a line that is not valid JSON: {exc}"
        if record.get("kind") == "gated_pick":
            runs.append(record)

    if len(runs) != expected_runs:
        return False, f"expected {expected_runs} gated_pick records, found {len(runs)}"

    order = [gates.G1_SEEN, gates.G2_AMBIGUOUS, gates.G3_REACHABLE,
             gates.G4_CONFIDENCE, gates.G5_VERIFY]
    for record in runs:
        gate_names = [g["gate"] for g in record.get("gates", [])]
        if not gate_names:
            return False, f"a run recorded no gates at all: {record.get('reason')}"
        positions = [order.index(name) for name in gate_names if name in order]
        if positions != sorted(positions):
            return False, f"gates out of order: {gate_names}"
        for field in ("ok", "object", "stopped_at", "confidence", "moved"):
            if field not in record:
                return False, f"a gated_pick record is missing {field!r}"

    summary = "\n".join(
        f"{r['object']}: " + " -> ".join(
            f"{g['gate']}{'' if g['passed'] else ' STOP'}" for g in r["gates"]
        )
        for r in runs
    )
    return True, f"{len(runs)} runs, each ordered and complete:\n{summary}"


# --- real vision, no motion ---------------------------------------------


def _real_vision(args) -> int:
    """Real gates and real eyes against your actual table. Nothing moves."""
    if config.api_key("GOOGLE_API_KEY") is None:
        return skip("GOOGLE_API_KEY is not set")
    if zones.load_zones() is None:
        return skip("no zones defined — run: python scripts/define_zones.py")

    macro = Recorder()
    print(f"\nLooking for {args.object!r} on the real table. The macro is stubbed — nothing moves.\n")
    result = gates.run_gated_pick(
        motion.DryRunArm(), args.object,
        clarify=_console_clarify, approve=_console_approve,
        perform=macro, verify_vlm=False,
    )
    _report(result)
    if macro.zones:
        print(f"\nwould have run the taught macro for {macro.zones}")
    return ok(f"gates ran: {result.speak()}") if result.ok else skip(result.reason)


def _console_clarify(question: str, options: list[str]) -> str | None:
    print(f"\nARM-ANI asks: {question}")
    try:
        return input(f"  your answer {options}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None


def _console_approve(prompt: str, timeout_s: float) -> bool:
    print(f"\nARM-ANI asks: {prompt}")
    print(f"  (it stands down on its own after {timeout_s:.0f}s of silence)")
    try:
        return input("  approve? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# --- live, three acts ----------------------------------------------------


def _live(args) -> int:
    if config.api_key("GOOGLE_API_KEY") is None:
        return skip("GOOGLE_API_KEY is not set")
    if zones.load_zones() is None:
        return skip("no zones defined — run: python scripts/define_zones.py")
    from armani import gestures

    if gestures.episode_count(config.PICK_DATASET_ROOT) == 0:
        return skip(f"no pick macros recorded. See {pick.RUNBOOK}.")
    if not config.HOME_VERIFIED:
        return skip("home pose is not verified — run scripts/capture_home.py (safety rule 4)")

    print(
        "\nThe three demo acts, for real:\n"
        f"  ACT 1  one {args.object!r} clearly on a spot        -> clean pick\n"
        "  ACT 2  two of the same object on two spots       -> it asks which\n"
        "  ACT 3  an object between two spots, say nothing  -> it stands down\n"
    )
    if not safety.require_operator("run the gated pick demo — the arm will move"):
        return skip("operator did not confirm presence")

    safety.install_kill_switch()
    try:
        arm = motion.connect()
    except Exception as exc:
        return fail(f"could not connect: {type(exc).__name__}: {exc}")

    outcomes: list[str] = []
    try:
        for act, instruction in enumerate(
            (
                f"put ONE {args.object} clearly on a marked spot",
                f"put TWO {args.object}s on two different spots",
                f"put ONE {args.object} BETWEEN two spots, then stay silent when it asks",
            ),
            start=1,
        ):
            print(f"\n===== ACT {act} =====\n  {instruction}")
            try:
                input("  press ENTER when the table is set, or Ctrl-C to stop... ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            result = gates.run_gated_pick(
                arm, args.object, clarify=_console_clarify, approve=_console_approve,
            )
            _report(result)
            outcomes.append(f"act {act}: {result.stopped_at or 'completed'}")
            log_event("smoke_12_act", act=act, **result.as_log())
    except KeyboardInterrupt:
        print("\ninterrupted — the arm holds where it is.")
        return EXIT_SKIP
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"  (warning: disconnect failed: {exc})")

    if not outcomes:
        return skip("no acts were run")
    return ok("; ".join(outcomes))


def _report(result: gates.GatedResult) -> None:
    print("\n--- gates ---")
    for record in result.records:
        print(f"  {'PASS' if record.passed else 'STOP'}  {record.gate:<14} {record.detail}")
    print(f"\n  confidence : {result.confidence:.0%} (vision {result.vision_confidence:.0%})")
    print(f"  zone       : {result.zone_label}")
    if result.approval_required:
        print(f"  approval   : approved={result.approved} timed_out={result.timed_out}")
    print(f"  moved      : {result.moved}")
    print(f"  verified   : {result.verified}")
    print(f"\n  it says: {result.speak()}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(EXIT_SKIP) from None
