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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 2


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
