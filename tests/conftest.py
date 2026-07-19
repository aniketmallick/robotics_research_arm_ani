"""Pytest-wide fixtures.

The one job here is keeping the test suite OUT of the real decision log.
``logs/decisions.jsonl`` is the audit trail the judges read and the thing stage
7's dashboard renders; a test run that appends a few hundred synthetic
`gated_pick` records to it is writing fiction into the evidence. (Measured on
2026-07-19: one suite run added 260 of them.)

The smoke tests are separate scripts, not pytest, so they still exercise the
real log on purpose — smoke_12 reads it back to check the audit trail's shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armani import config  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _decision_log_to_tmp(tmp_path_factory):
    """Redirect the decision log for the whole session."""
    scratch = tmp_path_factory.mktemp("decisions")
    original_log, original_dir = config.DECISION_LOG, config.LOG_DIR
    config.LOG_DIR = scratch
    config.DECISION_LOG = scratch / "decisions.jsonl"
    try:
        yield config.DECISION_LOG
    finally:
        config.DECISION_LOG, config.LOG_DIR = original_log, original_dir
