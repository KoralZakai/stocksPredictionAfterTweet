"""The point-in-time leak guard for the experimental macro-context provider.

This is the invariant that keeps System B honest: the injected macro context may
only contain events dated STRICTLY BEFORE t0. If this ever fails, the contextual
experiment is leaking the future and its numbers are void.
"""

from __future__ import annotations

from datetime import datetime, timezone

from experiments.context.macro_calendar import _as_utc_date, context_asof, events_asof


def test_never_returns_event_on_or_after_t0() -> None:
    for t0 in ("2025-01-29T14:00:00+00:00", "2025-03-20T10:00:00+00:00",
               "2025-12-31T23:59:00+00:00", "2026-06-17T13:00:00+00:00"):
        t0d = _as_utc_date(t0)
        assert all(e[0] < t0d for e in events_asof(t0)), f"future leak at {t0}"


def test_same_day_event_does_not_leak() -> None:
    # FOMC on 2025-01-29: a tweet AT 2025-01-29 must NOT see it (outcome not yet known).
    same = events_asof("2025-01-29T14:00:00+00:00")
    assert all(e[0].isoformat() != "2025-01-29" for e in same)
    # A tweet the next day may see it.
    nxt = events_asof("2025-01-30T14:00:00+00:00")
    assert any(e[0].isoformat() == "2025-01-29" for e in nxt)


def test_lookback_window_bounds() -> None:
    far = events_asof("2025-06-01T14:00:00+00:00", lookback_days=3)
    assert far == []


def test_context_string_is_empty_when_no_events() -> None:
    # Well before the calendar starts -> no context, never a crash.
    assert context_asof("2024-01-01T14:00:00+00:00") == ""


def test_naive_timestamp_is_treated_as_utc() -> None:
    naive = datetime(2025, 1, 30, 14, 0)              # no tzinfo
    aware = datetime(2025, 1, 30, 14, 0, tzinfo=timezone.utc)
    assert events_asof(naive) == events_asof(aware)
