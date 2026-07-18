"""Taught zones: fixed marked spots on the table, in PIXEL space only.

This is the ratified demo pick path (CLAUDE.md, Grasp). The idea is to move the
hard part off the machine and onto the human: instead of computing where an
object is in robot coordinates and solving IK to reach it, the operator
teleop-records a *working* pick at each marked spot, and the arm replays it.

What that buys, all at once: no homography error, no IK verticality problem, no
riser geometry, and a camera bump costs a two-minute re-click instead of a full
recalibration. What vision has to do shrinks to IDENTITY — "which of these
marked spots holds the banana?" — which is a coarse call that tolerates being
tens of pixels wrong.

Deliberately absent from this file: robot-frame coordinates, metres, homography
and inverse kinematics. A zone is a pixel and an episode index. If you find
yourself wanting a metre here, you are rebuilding stage 4.

Nothing here moves the arm or imports motion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from armani import config, eyes
from armani.logutil import get_logger, log_event

log = get_logger("zones")


@dataclass(frozen=True)
class Zone:
    """One marked spot: where it looks like, and which macro picks from it."""

    id: str
    label: str
    pixel_center: tuple[int, int]
    pick_episode: int


@dataclass(frozen=True)
class ZoneSet:
    """The zones as defined by scripts/define_zones.py, plus their provenance."""

    zones: tuple[Zone, ...]
    frame_size: tuple[int, int]
    created: str

    def __len__(self) -> int:
        return len(self.zones)

    def by_id(self, zone_id: str) -> Zone | None:
        return next((z for z in self.zones if z.id == zone_id), None)


@dataclass(frozen=True)
class ZoneMatch:
    """Which zone an object was assigned to, and how close the call was.

    ``margin_px`` is the whole point of this type. The nearest zone alone would
    always produce an answer, including when the object sits exactly between two
    spots. The distance to the RUNNER-UP is what tells us whether the answer
    means anything, and stage 6's G2 gate asks "which one did you mean?" when it
    is small. This class exposes that; it never resolves it.
    """

    zone: Zone | None
    distance_px: float = math.inf
    runner_up: Zone | None = None
    runner_up_distance_px: float = math.inf
    reason: str = ""

    @property
    def margin_px(self) -> float:
        """How much closer the winner is than the runner-up. inf if it is alone."""
        if self.zone is None:
            return 0.0
        return self.runner_up_distance_px - self.distance_px

    @property
    def ambiguous(self) -> bool:
        """True when the object is close to being on two spots at once."""
        return self.zone is not None and self.margin_px < config.ASSIGNMENT_MARGIN_PX

    @property
    def ok(self) -> bool:
        """A zone was found AND the call was not too close to make."""
        return self.zone is not None and not self.ambiguous

    def __bool__(self) -> bool:
        return self.ok

    def as_log(self) -> dict[str, object]:
        return {
            "zone": None if self.zone is None else self.zone.id,
            "zone_label": None if self.zone is None else self.zone.label,
            "distance_px": None if math.isinf(self.distance_px) else round(self.distance_px, 1),
            "runner_up": None if self.runner_up is None else self.runner_up.id,
            # None when there is no winner (nothing to be clear OF) and when the
            # winner stands alone (an infinite margin, which is not valid JSON).
            "margin_px": (
                None if self.zone is None or math.isinf(self.margin_px)
                else round(self.margin_px, 1)
            ),
            "ambiguous": self.ambiguous,
            "reason": self.reason,
        }


# --- Loading -------------------------------------------------------------


def load_zones(path: Path | None = None) -> ZoneSet | None:
    """Read zones.json. None means "not defined yet", which is not an error."""
    if path is None:
        path = config.ZONES_PATH
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        width, height = payload["frame_size"]
        zones = tuple(
            Zone(
                id=str(entry["id"]),
                label=str(entry["label"]),
                pixel_center=(int(entry["pixel_center"][0]), int(entry["pixel_center"][1])),
                pick_episode=int(entry["pick_episode"]),
            )
            for entry in payload["zones"]
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        log.error("ignoring unreadable zones file at %s: %s", path, exc)
        return None

    if not zones:
        log.error("%s defines no zones", path)
        return None

    duplicates = {z.id for z in zones if sum(1 for other in zones if other.id == z.id) > 1}
    if duplicates:
        # by_id() would silently return the first of them and picks would go to
        # the wrong spot, so refuse the whole file rather than half-trust it.
        log.error("%s has duplicate zone id(s) %s", path, sorted(duplicates))
        return None

    return ZoneSet(
        zones=zones,
        frame_size=(int(width), int(height)),
        created=str(payload.get("created", "unknown")),
    )


def zones_available() -> bool:
    return load_zones() is not None


def describe(zone_set: ZoneSet | None) -> str:
    if zone_set is None:
        return f"NO ZONES — run: python scripts/define_zones.py  (writes {config.ZONES_PATH.name})"
    spots = ", ".join(f"{z.id}:{z.label}" for z in zone_set.zones)
    return (
        f"{len(zone_set)} zones defined {zone_set.created} at "
        f"{zone_set.frame_size[0]}x{zone_set.frame_size[1]} — {spots}"
    )


# --- Assignment ----------------------------------------------------------


def assign_pixel(
    point: tuple[int, int], zone_set: ZoneSet | None = None
) -> ZoneMatch:
    """Nearest zone to a pixel, with the distance to the runner-up.

    The core of zone assignment, taking a bare pixel so it is testable without
    building a Detection or touching a camera.
    """
    if zone_set is None:
        zone_set = load_zones()
    if zone_set is None or not zone_set.zones:
        # load_zones() already refuses an empty file, but a caller can hand one
        # in directly. Fail closed rather than IndexError on the nearest lookup.
        return ZoneMatch(None, reason="no zones defined — run scripts/define_zones.py")

    ranked = sorted(
        ((math.dist(point, z.pixel_center), z) for z in zone_set.zones),
        key=lambda pair: pair[0],
    )
    nearest_distance, nearest = ranked[0]
    runner_up_distance, runner_up = ranked[1] if len(ranked) > 1 else (math.inf, None)

    if nearest_distance > config.ZONE_MAX_DISTANCE_PX:
        # Not at a marked spot at all: on the floor, in someone's hand, or the
        # camera has moved. Assigning the nearest zone anyway would send the arm
        # to grasp a spot the object is demonstrably not on.
        return ZoneMatch(
            None,
            distance_px=nearest_distance,
            reason=(
                f"nearest zone {nearest.label!r} is {nearest_distance:.0f} px away, past the "
                f"{config.ZONE_MAX_DISTANCE_PX:.0f} px limit — the object is not on a marked spot "
                "(or the camera has moved, in which case re-run scripts/define_zones.py)"
            ),
        )

    match = ZoneMatch(
        zone=nearest,
        distance_px=nearest_distance,
        runner_up=runner_up,
        runner_up_distance_px=runner_up_distance,
    )
    if match.ambiguous:
        # replace() rather than rebuilding: two constructions of the same match
        # that have to stay identical except for one field is how they drift.
        other = runner_up.label if runner_up is not None else "?"
        return replace(
            match,
            reason=(
                f"between {nearest.label!r} and {other!r} — only {match.margin_px:.0f} px "
                f"apart, under the {config.ASSIGNMENT_MARGIN_PX:.0f} px margin"
            ),
        )
    return match


def assign_zone(detection: eyes.Detection, zone_set: ZoneSet | None = None) -> ZoneMatch:
    """Assign a vision detection to its marked spot."""
    if zone_set is None:
        zone_set = load_zones()
    if zone_set is None:
        return ZoneMatch(None, reason="no zones defined — run scripts/define_zones.py")

    if detection.frame_size != zone_set.frame_size:
        # Pixel coordinates only mean anything at the size they were measured
        # at. Silently comparing across sizes would put every object in the
        # wrong zone with total confidence.
        return ZoneMatch(
            None,
            reason=(
                f"detection is from a {detection.frame_size[0]}x{detection.frame_size[1]} frame but "
                f"zones were defined at {zone_set.frame_size[0]}x{zone_set.frame_size[1]}; "
                "re-run scripts/define_zones.py at the current camera resolution"
            ),
        )

    match = assign_pixel(detection.point, zone_set)
    log_event("zone_assign", object=detection.label, pixel=list(detection.point), **match.as_log())
    return match


def list_zone_objects(
    candidates: list[str], frame=None, zone_set: ZoneSet | None = None
) -> dict[str, str]:
    """What is sitting on each marked spot, as {zone_id: object_name}.

    One Gemini query for all candidates, each result assigned to its nearest
    zone. For "what's on the table?" and for stage-6 disambiguation.

    If two objects land on the same zone the CLOSER one wins — the spot holds one
    object, and the further detection is either a mistake or an object that is
    not on a spot at all. Zones with nothing on them are simply absent from the
    result rather than mapped to None.
    """
    if zone_set is None:
        zone_set = load_zones()
    if zone_set is None:
        return {}

    detections = eyes.list_visible(candidates, frame=frame)
    best: dict[str, tuple[float, str]] = {}
    for detection in detections:
        if detection.frame_size != zone_set.frame_size:
            continue
        match = assign_pixel(detection.point, zone_set)
        if match.zone is None:
            continue
        current = best.get(match.zone.id)
        if current is None or match.distance_px < current[0]:
            best[match.zone.id] = (match.distance_px, detection.label)

    result = {zone_id: label for zone_id, (_, label) in best.items()}
    log_event("zone_contents", candidates=candidates, contents=result)
    return result
