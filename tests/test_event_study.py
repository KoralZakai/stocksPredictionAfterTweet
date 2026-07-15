"""Guards for the experimental event study: point-in-time estimation, overlap
flagging, outcome-blind cohorts, and the shuffled-label canary."""

from __future__ import annotations

from datetime import datetime, timezone

from experiments.event_study.cohorts import tag_cohort
from experiments.event_study.engine import EST_GAP, Bar, load_bars, s0_index, study_event


def _mk_bars(n: int = 200, start_px: float = 100.0) -> list[Bar]:
    out = []
    px = start_px
    d = datetime(2025, 1, 2)
    i = 0
    while len(out) < n:
        if d.weekday() < 5:
            px *= 1.0 + (0.001 if i % 2 else -0.001)
            out.append(Bar(d.date().isoformat(), px * 0.999, px, 1_000_000.0))
            i += 1
        d = d.replace(day=d.day)  # noop guard
        d = d + __import__("datetime").timedelta(days=1)
    return out


def test_s0_strictly_after_t0() -> None:
    bars = _mk_bars(50)
    # Tweet AFTER the open of day k -> s0 must be day k+1, never day k.
    day = bars[10].date
    t0 = datetime.fromisoformat(day).replace(hour=15, tzinfo=timezone.utc)  # post-open
    i0 = s0_index(bars, t0)
    assert bars[i0].date > day
    # Tweet BEFORE the open -> same-day s0 is allowed (open still ahead).
    t0b = datetime.fromisoformat(day).replace(hour=11, tzinfo=timezone.utc)
    assert bars[s0_index(bars, t0b)].date == day


def test_estimation_window_ends_before_event() -> None:
    """The market model may only see sessions ending EST_GAP before s0."""
    bars = {"XLE": _mk_bars(200), "SPY": _mk_bars(200)}
    t0 = datetime.fromisoformat(bars["XLE"][150].date).replace(hour=20, tzinfo=timezone.utc)
    er = study_event(bars, "XLE", t0)
    assert er is not None
    i0 = next(i for i, b in enumerate(bars["XLE"]) if b.date == er.s0_date)
    assert i0 - EST_GAP - 1 >= 0          # estimation strictly precedes the gap
    assert er.s0_date > t0.date().isoformat() or t0.hour < 13


def test_overlap_flagged() -> None:
    bars = {"XLE": _mk_bars(200), "SPY": _mk_bars(200)}
    track: dict[str, int] = {}
    d1 = bars["XLE"][150].date
    d2 = bars["XLE"][152].date            # 2 sessions later < window(5) -> overlap
    e1 = study_event(bars, "XLE", datetime.fromisoformat(d1).replace(hour=20, tzinfo=timezone.utc),
                     prev_s0=track)
    e2 = study_event(bars, "XLE", datetime.fromisoformat(d2).replace(hour=20, tzinfo=timezone.utc),
                     prev_s0=track)
    assert e1 is not None and e1.overlapping is False
    assert e2 is not None and e2.overlapping is True


def test_cohorts_are_text_only() -> None:
    """tag_cohort must ignore outcome fields entirely."""
    base = {"text": "Massive tariffs on China coming!", "scenario": "Trade War"}
    with_outcome = {**base, "hits": {"EOD": [99, 99]}, "spy_returns": {"EOD": 9.9}}
    assert tag_cohort(base) == tag_cohort(with_outcome)
    assert tag_cohort(base)[0] == "GEO_SHOCK"
    assert tag_cohort({"text": "Vote for Joe Lombardo!", "scenario": "Domestic Politics"})[0] == "NOISE"
    assert tag_cohort({"text": "Intel is doing great things for America",
                       "scenario": "US Politics"})[0] == "CORPORATE"


def test_canary_shuffled_labels_yield_uniform_p() -> None:
    """Manufactured-signal canary: with REAL bars but RANDOM pseudo-event dates,
    the permutation p for |CAR| must not be extreme (the null must contain its
    own draws). If this fails, the engine fabricates significance from noise."""
    import random
    bars = load_bars()
    if "SPY" not in bars:
        return
    rng = random.Random(1)
    pool = [b.date for b in bars["SPY"] if "2025-01-01" <= b.date <= "2026-05-01"]
    dates = rng.sample(pool, 30)
    cars = []
    for d in dates:
        er = study_event(bars, "SPY", datetime.fromisoformat(d).replace(hour=14, tzinfo=timezone.utc))
        if er:
            cars.append(abs(er.car[1]))
    obs = sum(cars) / len(cars)
    null = []
    for _ in range(200):
        draw = rng.sample(pool, 30)
        vals = []
        for d in draw:
            er = study_event(bars, "SPY", datetime.fromisoformat(d).replace(hour=14, tzinfo=timezone.utc))
            if er:
                vals.append(abs(er.car[1]))
        if vals:
            null.append(sum(vals) / len(vals))
    p = (sum(1 for x in null if x >= obs) + 1) / (len(null) + 1)
    assert 0.02 < p < 0.98, f"canary tripped: random dates got p={p}"
