"""Shared plumbing for the standalone smoke tests.

Each smoke test is runnable on its own (``python tests/smoke_01_ports.py``);
this module only removes the sys.path and argument-parsing boilerplate that
would otherwise be copy-pasted six times.

Exit codes, which scripts/doctor.py reads:
    0 = PASS
    1 = FAIL
    2 = SKIP (couldn't run, with a stated reason — not a failure)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 2

# Modules that only exist in the blessed conda env. If these are missing we are
# running under the wrong interpreter, which is by far the most likely cause and
# produces six meaningless "No module named X" failures if left undiagnosed.
_CORE_MODULES = ("lerobot", "numpy", "cv2")


def find_project_interpreter() -> str | None:
    """Locate the conda env that actually has the project's dependencies."""
    candidates = []
    conda_root = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX")
    if conda_root:
        candidates.append(Path(conda_root).parent / "lerobot" / "bin" / "python")
        candidates.append(Path(conda_root) / "envs" / "lerobot" / "bin" / "python")
    for base in ("miniforge3", "miniconda3", "anaconda3", "mambaforge"):
        candidates.append(Path.home() / base / "envs" / "lerobot" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def missing_core_modules() -> list[str]:
    missing = []
    for name in _CORE_MODULES:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def wrong_interpreter_message(missing: list[str]) -> str:
    interpreter = find_project_interpreter()
    fix = (
        f"  conda activate lerobot\n  python {{script}}\n\nor use it directly:\n  {interpreter} {{script}}"
        if interpreter
        else "  conda activate lerobot"
    )
    return (
        f"\nWRONG PYTHON ENVIRONMENT.\n\n"
        f"  running: {sys.executable}\n"
        f"  missing: {', '.join(missing)}\n\n"
        f"This project must run in the conda 'lerobot' env — the only one with lerobot\n"
        f"and the Feetech servo SDK. See docs/env_report.md.\n\n"
        f"{fix}\n"
    )


def require_project_env() -> None:
    """Abort immediately, and legibly, if this is the wrong interpreter."""
    missing = missing_core_modules()
    if not missing:
        return
    # argv[0] as invoked, so the suggested command is copy-pasteable as-is.
    script = sys.argv[0] or "scripts/doctor.py"
    sys.stdout.flush()
    print(wrong_interpreter_message(missing).replace("{script}", script), file=sys.stderr)
    raise SystemExit(EXIT_FAIL)


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what the test would do without touching hardware or the network",
    )
    return parser.parse_args()


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def ok(message: str) -> int:
    print(f"PASS: {message}")
    sys.stdout.flush()
    return EXIT_PASS


def fail(message: str) -> int:
    # Flush stdout first: it is block-buffered when piped to a file, while
    # stderr is not, so without this the verdict prints before its own evidence.
    sys.stdout.flush()
    print(f"FAIL: {message}", file=sys.stderr)
    sys.stderr.flush()
    return EXIT_FAIL


def skip(message: str) -> int:
    print(f"SKIP: {message}")
    sys.stdout.flush()
    return EXIT_SKIP


def _check_on_import() -> None:
    """Every smoke test imports this module first, so one call here guards all six.

    Deliberately a side effect at import time: the wrong-interpreter case must be
    caught before a test prints a banner and reports a misleading FAIL.
    """
    require_project_env()


def permission_hint(pane: str, why: str) -> None:
    """Tell the operator exactly which macOS toggle to flip."""
    sys.stdout.flush()
    print(
        f"\n  macOS permission needed: {why}\n"
        f"  Open: System Settings > Privacy & Security > {pane}\n"
        f"  Enable your terminal app (Terminal / iTerm / VS Code), then RESTART it.\n"
        f"  Permissions are granted to the terminal app, not to Python.\n",
        file=sys.stderr,
    )


_check_on_import()
