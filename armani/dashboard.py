"""The robot's mind, as data. Presentation only — this never touches the arm.

Everything here is derived from two things the rest of the system already
produces: the decision log (`logs/decisions.jsonl`) and the last frame
perception looked at (`logs/last_frame.jpg`). Nothing is re-run, no gate is
re-evaluated, and the camera is never opened — the agent needs it, and two
processes fighting over one C920 mid-demo is a risk with no upside.

That constraint is also what makes the screen honest: it can only show what
actually happened and was written down.

``scripts/run_dashboard.py`` serves this over HTTP; a replay source feeds it a
past log so the gate story can still be told when the venue's wifi cannot.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from armani import config, zones
from armani.logutil import get_logger

log = get_logger("dashboard")

# Only ever read the tail: the log grows all session and re-parsing megabytes on
# every poll would make the screen stutter exactly when it is being watched.
TAIL_BYTES = 512 * 1024

# The gates, in pipeline order, so the screen can show the ones a run never
# reached as "not reached" rather than silently omitting them.
GATE_ORDER: tuple[tuple[str, str], ...] = (
    ("G1_seen", "SEEN"),
    ("G2_ambiguous", "WHICH ONE"),
    ("G3_reachable", "REACHABLE"),
    ("G4_confidence", "CONFIDENCE"),
    ("G5_verify", "VERIFIED"),
)

# Kinds worth showing in the scrolling feed, with how to summarise each.
FEED_KINDS = {
    "gated_pick": "pick",
    "gate_clarify_asked": "asked",
    "gate_clarify_answered": "answered",
    "eyes_locate": "looked",
    "eyes_confirm_held": "checked",
    "motion_enqueued": "motion",
    "motion_completed": "motion",
    "freeze": "FREEZE",
    "freeze_choice": "freeze",
    "stop_requested": "STOP",
    "operator_check": "operator",
    "zones_defined": "setup",
    "dataset_backup": "backup",
}


@dataclass(frozen=True)
class Source:
    """Where the dashboard reads from. A file, live or replayed."""

    path: Path
    replay: bool = False
    interval_s: float = 4.0

    def label(self) -> str:
        return f"REPLAY {self.path.name}" if self.replay else "LIVE"


def read_records(source: Source) -> list[dict]:
    """Parse the tail of a decision log into records, newest last.

    A truncated or half-written line is skipped rather than fatal: the agent
    appends to this file while we read it.
    """
    try:
        size = source.path.stat().st_size
        with source.path.open("rb") as handle:
            if size > TAIL_BYTES:
                handle.seek(size - TAIL_BYTES)
                handle.readline()  # discard the partial first line
            raw = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []

    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _picks(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("kind") == "gated_pick"]


def current_pick(records: list[dict], source: Source) -> dict | None:
    """The pick the screen should be showing.

    Live: the most recent one. Replay: whichever one the clock has reached, so a
    recorded session tells its story at a watchable pace instead of flashing the
    last frame of it.
    """
    picks = _picks(records)
    if not picks:
        return None
    if not source.replay:
        return picks[-1]
    step = int(time.monotonic() / max(source.interval_s, 0.5)) % len(picks)
    return picks[step]


def gate_rows(pick: dict | None) -> list[dict]:
    """Every gate in order, marked passed / stopped / not reached."""
    recorded = {g.get("gate"): g for g in (pick or {}).get("gates", [])}
    rows = []
    for name, caption in GATE_ORDER:
        entry = recorded.get(name)
        if entry is None:
            state = "pending"
            detail = ""
        elif entry.get("passed"):
            state = "passed"
            detail = str(entry.get("detail", ""))
        else:
            state = "stopped"
            detail = str(entry.get("detail", ""))
        rows.append({"gate": name, "caption": caption, "state": state, "detail": detail})
    return rows


def last_detection(records: list[dict]) -> dict | None:
    """The most recent successful sighting, for drawing on the frame."""
    for record in reversed(records):
        if record.get("kind") == "eyes_locate" and record.get("found") and record.get("point"):
            point = record["point"]
            if isinstance(point, list) and len(point) == 2:
                return {
                    "point": [int(point[0]), int(point[1])],
                    "label": str(record.get("object") or record.get("label") or ""),
                    "confidence": float(record.get("confidence") or 0.0),
                }
    return None


def _summarise(record: dict) -> str:
    kind = record.get("kind")
    if kind == "gated_pick":
        if record.get("ok"):
            return f"picked {record.get('object')} from {record.get('zone_label')} at {_pct(record.get('confidence'))}"
        return f"{record.get('object')}: {record.get('reason') or record.get('stopped_at')}"
    if kind == "gate_clarify_asked":
        return str(record.get("question", ""))
    if kind == "gate_clarify_answered":
        return f"answered {record.get('answer')!r} -> {record.get('resolved')}"
    if kind == "eyes_locate":
        if not record.get("found"):
            return f"could not see {record.get('object')}"
        return f"saw {record.get('object')} at {_pct(record.get('confidence'))}"
    if kind == "eyes_confirm_held":
        return f"held={record.get('vlm_held')} ({_pct(record.get('vlm_confidence'))}) {record.get('vlm_reason', '')}"
    if kind in ("motion_enqueued", "motion_completed"):
        return f"{record.get('action')} {record.get('status', '')}".strip()
    if kind == "freeze_choice":
        return f"operator chose {record.get('choice')}"
    if kind == "operator_check":
        return f"presence {'confirmed' if record.get('approved') else 'declined'}"
    if kind == "dataset_backup":
        return f"{record.get('dataset')} ({record.get('episodes')} episodes)"
    return ""


def _pct(value) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "?"


def feed(records: list[dict], limit: int | None = None) -> list[dict]:
    """The scrolling audit trail, newest first."""
    if limit is None:
        limit = config.DASHBOARD_FEED_LIMIT
    rows = []
    for record in reversed(records):
        kind = record.get("kind")
        if kind not in FEED_KINDS:
            continue
        summary = _summarise(record)
        if not summary:
            continue
        rows.append({
            "ts": record.get("ts"),
            "clock": _clock(record.get("ts")),
            "tag": FEED_KINDS[kind],
            "kind": kind,
            "text": summary,
            "bad": _is_bad(record),
        })
        if len(rows) >= limit:
            break
    return rows


def _is_bad(record: dict) -> bool:
    """Did this record represent the system declining or failing? Colour cue."""
    kind = record.get("kind")
    if kind == "gated_pick":
        return not record.get("ok")
    if kind == "eyes_locate":
        return not record.get("found")
    if kind in ("freeze", "stop_requested"):
        return True
    if kind == "motion_completed":
        return record.get("status") not in ("done", None)
    return False


def _clock(ts) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError):
        return "--:--:--"


def zone_overlay() -> list[dict]:
    """The marked spots, for drawing on the frame."""
    zone_set = zones.load_zones()
    if zone_set is None:
        return []
    return [
        {"id": z.id, "label": z.label, "x": z.pixel_center[0], "y": z.pixel_center[1]}
        for z in zone_set.zones
    ]


def frame_size() -> tuple[int, int]:
    zone_set = zones.load_zones()
    if zone_set is not None:
        return zone_set.frame_size
    return (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)


def build_state(source: Source) -> dict:
    """Everything the screen needs, in one JSON payload."""
    records = read_records(source)
    pick = current_pick(records, source)
    picks = _picks(records)

    width, height = frame_size()
    state = {
        "mode": source.label(),
        "generated_at": time.time(),
        "clock": time.strftime("%H:%M:%S"),
        "threshold": config.CONF_APPROVAL,
        "timeout_s": config.APPROVAL_TIMEOUT_S,
        "frame": {"width": width, "height": height},
        "zones": zone_overlay(),
        "detection": last_detection(records),
        "gates": gate_rows(pick),
        "feed": feed(records),
        "totals": _totals(picks),
        "pick": None,
    }

    if pick is not None:
        # Confidence only exists once G4 computes it. A run that stopped at G1,
        # G2 or G3 never had one, and showing "0%" would read as "it was certain
        # the answer was no" instead of "it never got that far".
        computed = pick.get("stopped_at") not in ("G1_seen", "G2_ambiguous", "G3_reachable")
        state["pick"] = {
            "object": pick.get("object"),
            "zone": pick.get("zone"),
            "zone_label": pick.get("zone_label"),
            "confidence": _as_float(pick.get("confidence")) if computed else None,
            "vision_confidence": _as_float(pick.get("vision_confidence")),
            "ok": bool(pick.get("ok")),
            "stopped_at": pick.get("stopped_at"),
            "reason": pick.get("reason") or "",
            "moved": bool(pick.get("moved")),
            "verified": pick.get("verified"),
            "clarified": bool(pick.get("clarified")),
            "approval_required": bool(pick.get("approval_required")),
            "approved": pick.get("approved"),
            "timed_out": bool(pick.get("timed_out")),
            "clock": _clock(pick.get("ts")),
            "headline": _headline(pick),
        }
    return state


def _as_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _headline(pick: dict) -> str:
    """The one line a judge should be able to read from three metres away."""
    if pick.get("ok"):
        return "PICKED IT"
    stopped = pick.get("stopped_at")
    if pick.get("timed_out"):
        return "STOOD DOWN — NO ANSWER"
    if stopped == "G1_seen":
        return "CAN'T SEE IT"
    if stopped == "G2_ambiguous":
        return "WHICH ONE?"
    if stopped == "G3_reachable":
        return "NOT TAUGHT THAT SPOT"
    if stopped == "G4_confidence":
        return "NOT APPROVED"
    if stopped == "G5_verify":
        return "MISSED IT — AND SAID SO"
    return "STOPPED"


def _totals(picks: list[dict]) -> dict:
    """Session tally. The stand-down count is the one judges should notice."""
    return {
        "picks": len(picks),
        "completed": sum(1 for p in picks if p.get("ok")),
        "stood_down": sum(1 for p in picks if p.get("timed_out")),
        "clarified": sum(1 for p in picks if p.get("clarified")),
        "refused": sum(1 for p in picks if not p.get("ok")),
    }
