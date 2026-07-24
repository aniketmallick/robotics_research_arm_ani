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
import os
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

# Eval send-path clamp envelopes. 'policy' is the default and the S2 baseline's
# envelope. 'recorded' is the wider physical−2° envelope, ARCHITECT-RATIFIED for the
# FINE-TUNED S3 eval ONLY (see eval.md): a fine-tuned policy cloned from teleop demos
# is closer to "replaying teleop" (recorded) than to "untrusted LLM JSON" (policy), and
# clamping it tighter than its own training data guarantees a false-negative grasp.
EVAL_CLAMP_PROFILES = ("policy", "recorded")


def resolve_clamp_profile(cli_profile: str | None, is_base: bool) -> str:
    """The send-path clamp profile for the eval. Pure/testable.

    Precedence: ``--clamp-profile``, else ``ARMANI_EVAL_CLAMP_PROFILE``, else ``policy``.
    Default ``policy`` is the S2 behaviour — zero change. ``recorded`` (the wider
    physical−2° envelope) is REFUSED on the base model: it is ratified for the fine-tuned
    S3 eval ONLY, so the closed S2 baseline (0/2) can never be re-measured under a wider
    envelope. It also requires the real ``armani.safety`` source (the embedded fallback
    carries only policy limits — we never approximate a wider-than-policy bound).
    """
    profile = (cli_profile or os.getenv("ARMANI_EVAL_CLAMP_PROFILE") or "policy").strip()
    if profile not in EVAL_CLAMP_PROFILES:
        raise SystemExit(
            f"clamp profile must be one of {EVAL_CLAMP_PROFILES}, got {profile!r} "
            "(--clamp-profile / ARMANI_EVAL_CLAMP_PROFILE)."
        )
    if profile == "recorded":
        if is_base:
            raise SystemExit(
                "clamp profile 'recorded' is REFUSED on the base model. It is ratified for the "
                "FINE-TUNED S3 eval ONLY (pass --policy-path <checkpoint>). The closed S2 baseline "
                "(0/2) must never be re-measured under a wider envelope."
            )
        if not _HAVE_ARMANI:
            raise SystemExit(
                "clamp profile 'recorded' needs the real armani.safety (not importable here); "
                "the embedded fallback carries only policy limits."
            )
    return profile


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


def clamp_source(profile: str = "policy") -> str:
    """Which clamp is active — for the report and the run banner."""
    return f"armani.safety.clamp_action({profile})" if _HAVE_ARMANI else f"embedded-fallback({profile})"


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


def policy_clamp(action: dict[str, float], profile: str = "policy") -> ClampResult:
    """Clamp ``action`` to the given envelope and report whether it bit.

    ``profile`` is ``policy`` (default, the S2 envelope) or ``recorded`` (the wider
    physical−2° envelope, ratified for the fine-tuned S3 eval only — see
    :func:`resolve_clamp_profile`). ``recorded`` requires the real ``armani.safety``;
    the embedded fallback carries only policy limits and refuses a wider request rather
    than silently substituting a tighter bound.

    Raises ``ValueError`` on NaN/inf or an unknown joint — a garbage prediction
    must fail loudly and be dropped by the caller, never quietly sent.
    """
    if _HAVE_ARMANI:
        # log_clamps=False: the runner logs raw-vs-clamped itself, per step, into
        # the episode JSONL; the safety logger would double-write every clamp.
        clamped = _armani_safety.clamp_action(action, profile=profile, log_clamps=False)
    elif profile == "policy":
        clamped = _fallback_clamp(action)
    else:
        raise SystemExit(
            f"clamp profile {profile!r} needs the real armani.safety (not importable here); "
            "the embedded fallback carries only policy limits."
        )

    bit_joints = tuple(
        joint for joint in action if abs(clamped[joint] - float(action[joint])) > CLAMP_EPS
    )
    return ClampResult(clamped=clamped, bit=bool(bit_joints), bit_joints=bit_joints)
