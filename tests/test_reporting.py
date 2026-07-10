"""Reporting job: the aggregates must not lie, and the render must be safe.

The load-bearing test is `test_clustered_z_is_not_the_naive_z`: rows spawned by
one post are correlated, so a naive row-count SE overstates significance. If that
distinction is ever lost, this dashboard starts manufacturing findings.
"""

from __future__ import annotations

import pandas as pd
import pytest

from llm.schema import TweetSignal
from reportgen.render import render_html
from reportgen.stats import build_report


def _sig(event: str = "tariff_trade", intent: str = "bullish") -> TweetSignal:
    return TweetSignal(
        event_type=event, direction_of_intent=intent, urgency="low",  # type: ignore[arg-type]
        magnitude="low", certainty="low", names_country=False, names_company=False,
    )


ETF_POOL = ("XLI", "XLK", "XLE", "SPY")


def _frame(n_posts: int = 6, n_etfs: int = 1, n_stocks: int = 3,
           alternate: bool = False) -> pd.DataFrame:
    """A post links n_etfs sector ETFs plus n_stocks single names — the real
    (post x asset) cross-product shape. Hit rates use the ETF rows only.

    alternate=True flips the outcome sign per post, so per-post hit rates vary
    and the clustered SE has something to measure."""
    rows = []
    for p in range(n_posts):
        # 1-in-3 posts flipped -> hit rate 2/3, imbalanced enough that both
        # z-scores are non-zero and can be compared.
        sign = -1.0 if (alternate and p % 3 == 0) else 1.0
        for a in list(ETF_POOL[:n_etfs]) + [f"S{i}" for i in range(n_stocks)]:
            rows.append({
                "post_id": f"p{p}", "timestamp_utc": f"2025-01-0{p+1}T15:00:00Z",
                "text": "Massive tariffs on China, our manufacturing comes home",
                "asset": a, "is_etf": int(a in ETF_POOL),
                "relevance": 0.5, "abn_1": 0.01 * sign,
                "abn_3": -0.02 * sign, "abn_5": 0.03 * sign,
            })
    return pd.DataFrame(rows)


def test_build_report_shapes_and_counts() -> None:
    d = _frame()
    r = build_report(d, {f"p{i}": _sig() for i in range(6)}, "test-model")
    assert r.n_posts == 6
    assert r.n_rows == 24
    assert r.signal_model == "test-model"
    assert [h.horizon for h in r.hit] == ["1d", "3d", "5d"]


def test_majority_baseline_is_reported_and_can_beat_the_signal() -> None:
    # every abn_1 is positive -> majority baseline is a perfect 1.0, and a
    # bullish signal also scores 1.0, so it does NOT *beat* the baseline.
    d = _frame()
    r = build_report(d, {f"p{i}": _sig(intent="bullish") for i in range(6)}, "m")
    h1 = next(h for h in r.hit if h.horizon == "1d")
    assert h1.hit == 1.0
    assert h1.majority == 1.0
    assert h1.beats_majority is False


def test_clustered_z_is_not_the_naive_z() -> None:
    """Correlated rows within a post must not inflate significance."""
    d = _frame(n_posts=6, n_etfs=4, n_stocks=0, alternate=True)
    r = build_report(d, {f"p{i}": _sig() for i in range(6)}, "m")
    h1 = next(h for h in r.hit if h.horizon == "1d")
    assert h1.n_rows == 24 and h1.n_posts == 6   # 4 correlated ETF rows per post
    # The naive SE pretends all 24 rows are independent and so overstates |z|.
    assert abs(h1.z_naive) > abs(h1.z_clustered)


def test_missing_signal_is_an_error_not_a_silent_zero() -> None:
    d = _frame(n_posts=3)
    with pytest.raises(KeyError):
        build_report(d, {"p0": _sig()}, "m")  # p1, p2 absent


def test_render_is_pure_ascii_and_escapes_text() -> None:
    d = _frame(n_posts=3)
    d.loc[0, "text"] = 'tariffs <script>alert("x")</script> & "quotes"'
    r = build_report(d, {f"p{i}": _sig() for i in range(3)}, "m")
    out = render_html(r, "abc123")
    assert all(ord(c) < 128 for c in out)          # renders under any charset
    assert "<script>alert" not in out              # escaped, not injected
    assert "abc123" in out                         # run id stamped
