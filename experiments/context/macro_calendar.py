"""Leak-safe point-in-time macro context. EXPERIMENTAL — see experiments/context.

The ONLY macro context we inject is a real, dated, PUBLIC calendar of scheduled
macro events (FOMC decision days, CPI release days) — objective facts published in
advance, not model memory and not simulated sentiment. `context_asof(t0)` returns
ONLY events dated STRICTLY BEFORE t0, so a feature can never see the future
(CLAUDE.md 3.1). A test asserts that invariant.

We deliberately encode only the EXISTENCE and TYPE of a dated event (e.g. "an FOMC
rate decision occurred on 2025-01-29"), never its outcome or any after-the-fact
market interpretation — that would leak the thing we are trying to predict.

DATA QUALITY: these dates are transcribed from the public Fed/BLS calendars and
should be re-verified against the official sources before any published claim.
Leak-safety does NOT depend on their accuracy — the strict `date < t0` filter
holds regardless. Fill gaps by extending the tuples below.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

# (iso_date, event_type, neutral_label) — NO outcome, NO sentiment.
# Verify against federalreserve.gov (FOMC) and bls.gov (CPI) before publishing.
_FOMC_DECISIONS: tuple[str, ...] = (
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
)
_CPI_RELEASES: tuple[str, ...] = (
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
    "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-15",
    "2025-11-13", "2025-12-10",
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10", "2026-05-13",
    "2026-06-10",
)

# Flatten to a sorted event list once.
_EVENTS: list[tuple[date, str, str]] = sorted(
    [(date.fromisoformat(d), "FOMC", "FOMC rate decision") for d in _FOMC_DECISIONS]
    + [(date.fromisoformat(d), "CPI", "CPI inflation release") for d in _CPI_RELEASES],
    key=lambda e: e[0],
)


def _as_utc_date(t0: str | datetime) -> date:
    dt = datetime.fromisoformat(t0) if isinstance(t0, str) else t0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date()


def context_asof(t0: str | datetime, *, lookback_days: int = 30) -> str:
    """Point-in-time macro context string for a tweet at t0.

    Returns only scheduled macro events with `event_date < t0` (strictly), within
    `lookback_days`. Empty string if none. NEVER includes t0's own day or later —
    that is the leak guard, enforced by test_no_future_leak.
    """
    t0d = _as_utc_date(t0)
    recent = [e for e in _EVENTS if e[0] < t0d and (t0d - e[0]).days <= lookback_days]
    if not recent:
        return ""
    parts = [f"{e[0].isoformat()}: {e[2]}" for e in recent]
    return "Recent scheduled macro events (public calendar, before the post): " + "; ".join(parts)


def events_asof(t0: str | datetime, *, lookback_days: int = 30) -> list[tuple[date, str, str]]:
    """Same filter as context_asof, structured — for the leak test + diagnostics."""
    t0d = _as_utc_date(t0)
    return [e for e in _EVENTS if e[0] < t0d and (t0d - e[0]).days <= lookback_days]


def _demo() -> None:
    """CPU self-check: the leak guard holds and lookback bounds work."""
    # A t0 the day AFTER an FOMC decision sees it; the decision day itself does not leak.
    fomc = "2025-01-29"
    after = events_asof("2025-01-30T14:00:00+00:00")
    assert any(e[0].isoformat() == fomc for e in after), "should see the prior-day FOMC"
    same = events_asof("2025-01-29T14:00:00+00:00")
    assert all(e[0].isoformat() != fomc for e in same), "LEAK: t0's own day returned"
    far = events_asof("2025-06-01T14:00:00+00:00", lookback_days=5)
    assert far == [], "lookback window not respected"
    # Every returned event is strictly before t0, always.
    for t0 in ("2025-03-20T10:00:00+00:00", "2026-01-15T21:00:00+00:00"):
        assert all(e[0] < _as_utc_date(t0) for e in events_asof(t0)), "LEAK: event >= t0"
    print("macro_calendar self-check OK (leak guard holds)")


if __name__ == "__main__":
    _demo()
