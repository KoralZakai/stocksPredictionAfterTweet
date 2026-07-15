"""Shared pytest fixtures.

Keep this file boring: anything clever here applies to every test in the suite.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_prediction_log(tmp_path, monkeypatch):
    """Point the Endpoint's prediction log at a throwaway file for EVERY test.

    Without this, any test that calls /predict appends to the real default log
    (runs/real/prediction_log.jsonl) — the same file jobs/feedback.py scores as a
    prospective replication set. A CI run would quietly inject fabricated tweets into
    the evidence base, and because the log is gitignored nobody would ever see it
    happen. Autouse because the damage is silent and opt-in would be forgotten.
    """
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "prediction_log.jsonl"))
