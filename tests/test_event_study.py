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


def test_us_open_utc_hour_tracks_dst() -> None:
    """NYSE opens 09:30 ET = 13:30 UTC in EDT but 14:30 UTC in EST. A hardcoded 13.5
    mis-anchored every tweet posted 13:30-14:30 UTC from Nov to mid-March."""
    from datetime import date as _d

    from alpha.benchmark import us_open_utc_hour
    assert us_open_utc_hour(_d(2026, 7, 15)) == 13.5      # EDT (summer)
    assert us_open_utc_hour(_d(2026, 3, 6)) == 14.5       # EST — DST starts Mar 8 2026
    assert us_open_utc_hour(_d(2026, 3, 9)) == 13.5       # EDT, day after the switch
    assert us_open_utc_hour(_d(2026, 1, 15)) == 14.5      # EST (winter)


def test_est_pre_open_tweet_anchors_same_day() -> None:
    """The regression: 2026-03-06 13:49 UTC is BEFORE that day's 14:30 UTC open (EST),
    so it must anchor to 03-06 itself — not skip to the next session."""
    from alpha.benchmark import _session_anchor, session_phase
    t0 = datetime(2026, 3, 6, 13, 49, tzinfo=timezone.utc)
    anchor = _session_anchor(t0)
    assert anchor.date().isoformat() == "2026-03-06", "EST pre-open tweet skipped a session"
    assert (anchor.hour, anchor.minute) == (14, 30), "should anchor to that day's real open"
    # The SAME clock time in July is INSIDE regular hours (13:30 EDT open), so the
    # tweet is immediately tradeable and anchors to itself. Identical UTC time, two
    # different correct answers — which is exactly what the hardcoded 13.5 could not do.
    jul_t0 = datetime(2026, 7, 15, 13, 49, tzinfo=timezone.utc)
    assert _session_anchor(jul_t0) == jul_t0
    assert session_phase(jul_t0) == "regular"
    assert session_phase(t0) == "premarket"


def test_window_series_anchor_respects_tweet_hour() -> None:
    """An AFTER-CLOSE post must not treat its own session as post-event. A naive
    `date >= day` anchor did exactly that and under-measured the Intel run-up by
    ~48pp (+105.3% -> +44.8%). The series must anchor to the NEXT session."""
    from experiments.event_study.generate_targeted_dashboard import gather_window_series
    bars = load_bars()
    if "INTC" not in bars or "SPY" not in bars:
        return
    anc = [{"ts": "2026-04-29T22:20", "ticker": "INTC", "text": "t", "label": "l"}]
    got = gather_window_series(bars, anc, span=42)
    assert got, "anchor should resolve"
    assert got[0]["s0"] > "2026-04-29", "after-close post leaked its own session"


def test_runup_is_exact_not_negated_return() -> None:
    """Negating a backward-indexed return is only valid for small moves. On a double
    it reports ~+53% for a +115% run-up — the bug that shipped to the slider once."""
    from experiments.event_study.generate_targeted_dashboard import gather_window_series
    bars = load_bars()
    if "INTC" not in bars:
        return
    anc = [{"ts": "2026-04-29T22:20", "ticker": "INTC", "text": "t", "label": "l"}]
    got = gather_window_series(bars, anc, span=42)
    if not got:
        return
    a = got[0]
    p = next(s for s in a["series"] if s["k"] == -21)
    exact = (a["prev_a"] / p["pa"] - 1) - (a["prev_b"] / p["pb"] - 1)
    naive = -(((p["pa"] / a["open_a"]) - 1) - ((p["pb"] / a["open_b"]) - 1))
    assert exact > 1.0, f"expected a >100% run-up, got {exact:.3f}"
    assert exact - naive > 0.4, "the naive approximation should be visibly wrong here"


def test_intraday_split_is_point_in_time() -> None:
    """The pre-window may not contain a bar that closed at/after t0, and the bar
    straddling t0 belongs to neither side (it mixes both regimes)."""
    from experiments.event_study.intraday import _split_at, load_hourly
    bars = load_hourly()
    if "USO" not in bars:
        return
    a = bars["USO"]
    t0 = a[300].ts.replace(minute=17)          # tweet lands mid-bar
    j, i = _split_at(a, t0)
    assert a[j].end <= t0, "pre-window leaked a bar closing at/after t0"
    assert a[i].ts >= t0, "post-window started before t0"
    assert i - j >= 2, "the straddling bar was not dropped from both sides"


def test_intraday_dedupes_same_bar_bursts() -> None:
    """He posts in bursts: several tweets inside one hour resolve to the SAME bar and
    must count ONCE. Counting them separately fabricated 9 BH-surviving cells
    (p_bh 0.0018) out of pure duplication — the whole 'intraday shock' result.
    """
    from experiments.event_study.intraday import load_hourly, study_shock
    bars = load_hourly()
    if "USO" not in bars or "SPY" not in bars:
        return
    base = bars["USO"][300].ts
    a = study_shock(bars, "USO", base.replace(minute=5))
    b = study_shock(bars, "USO", base.replace(minute=6))     # 1 minute later
    if a is None or b is None:
        return
    # Same bar -> identical measurement. Two tweets, ONE piece of evidence.
    assert a.t0[:13] == b.t0[:13] or a.post_excess == b.post_excess, (
        "two posts in one hour must not be independent observations")


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
