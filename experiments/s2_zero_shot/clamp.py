"""Policy-limit clamp for the S2 zero-shot runner.

The one law this spike must not break (CLAUDE.md safety rule 2, and the spike's
own guardrail): **no raw policy action ever reaches the bus.** Every action the
VLA predicts passes through :func:`policy_clamp` before any send.

Source of truth is the REAL project clamp — ``armani.safety.clamp_action`` with
the ``policy`` profile — so the limit table lives in exactly one place. If
``armani`` cannot import in this parallel env (it can, once python-dotenv is
present — see env_report.md), we fall back to an embedded copy of the policy
limits so the clamp is ALWAYS in the path, and :func:`clamp_source` says which
one is active. A test asserts the two agree whenever armani is importable, so
the fallback cannot silently drift from safety rule 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# CLAUDE.md safety rule 2, policy profile. ONLY used if the real armani.safety
# import fails; test_clamp_matches_armani keeps it honest against the source.
_FALLBACK_POLICY_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-60.0, 60.0),
    "elbow_flex": (-60.0, 60.0),
    "wrist_flex": (-60.0, 60.0),
    "wrist_roll": (-150.0, 150.0),
    "gripper": (0.0, 100.0),  # percent, not degrees
}

try:
    from armani import safety as _armani_safety

    _HAVE_ARMANI = True
    _IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - exercised only in a stripped env
    _armani_safety = None
    _HAVE_ARMANI = False
    _IMPORT_ERROR = _exc

# Floating-point slack below which a value is treated as unchanged by the clamp.
CLAMP_EPS = 1e-6


@dataclass(frozen=True)
class ClampResult:
    """Outcome of clamping one action.

    ``bit`` is the metric the spike asks for: did the clamp actually change any
    joint this step? ``bit_joints`` names which, so we can report WHICH joints
    the untuned policy pushed out of bounds most often.
    """

    clamped: dict[str, float]
    bit: bool
    bit_joints: tuple[str, ...]


def clamp_source() -> str:
    """Which clamp is active — for the report and the run banner."""
    return "armani.safety.clamp_action(policy)" if _HAVE_ARMANI else "embedded-fallback"


def _fallback_clamp(action: dict[str, float]) -> dict[str, float]:
    clamped: dict[str, float] = {}
    for joint, value in action.items():
        if joint not in _FALLBACK_POLICY_LIMITS:
            raise ValueError(f"unknown joint {joint!r}; expected one of {sorted(_FALLBACK_POLICY_LIMITS)}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"joint {joint!r} got non-finite value {numeric!r}")
        low, high = _FALLBACK_POLICY_LIMITS[joint]
        clamped[joint] = min(high, max(low, numeric))
    return clamped


def policy_clamp(action: dict[str, float]) -> ClampResult:
    """Clamp ``action`` to the policy envelope and report whether it bit.

    Raises ``ValueError`` on NaN/inf or an unknown joint — a garbage prediction
    must fail loudly and be dropped by the caller, never quietly sent.
    """
    if _HAVE_ARMANI:
        # log_clamps=False: the runner logs raw-vs-clamped itself, per step, into
        # the episode JSONL; the safety logger would double-write every clamp.
        clamped = _armani_safety.clamp_action(action, profile="policy", log_clamps=False)
    else:
        clamped = _fallback_clamp(action)

    bit_joints = tuple(
        joint for joint in action if abs(clamped[joint] - float(action[joint])) > CLAMP_EPS
    )
    return ClampResult(clamped=clamped, bit=bool(bit_joints), bit_joints=bit_joints)
