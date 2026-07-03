"""Assemble the final labeled dataset (§5, build-order -> dataset_build_job).

One row per tweet that maps to a sector: point-in-time features (via the single
decide() path) joined to the vol-scaled labels (via labeling/). This is the
artifact modeling (step 8) and evaluation (step 7) consume. Feature side is
strictly < t0; label side is strictly > t0 — they never touch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone

from config.settings import SETTINGS, Settings
from core.calendar import TradingCalendar
from core.decide import decide
from core.market_state import market_state_as_of
from data.sources.interfaces import PriceSource, Tweet
from labeling.thresholds import backward_vol, label
from labeling.windows import compute_outcome
from sector_mapping.rules import map_tweet

_WIDE = (datetime(1990, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


@dataclass(frozen=True)
class Row:
    tweet_id: str
    ticker: str
    t0: datetime
    s0_date: date
    map_confidence: float
    features: dict[str, float]
    ret: dict[int, float | None]
    label: dict[int, str]


def build_dataset(
    tweets: Sequence[Tweet],
    prices: PriceSource,
    cfg: Settings = SETTINGS,
) -> list[Row]:
    rows: list[Row] = []
    spy = prices.get_daily_bars(cfg.benchmark, *_WIDE)
    for tw in tweets:
        m = map_tweet(tw.text)
        if m.ticker is None:  # maps to NONE -> excluded (§6)
            continue
        bars = prices.get_daily_bars(m.ticker, *_WIDE)
        cal = TradingCalendar([b.session_date.date() for b in bars])
        outcome = compute_outcome(tw.timestamp_utc, bars, cal, cfg.horizon_days)
        if outcome is None:
            continue
        state = market_state_as_of(tw.timestamp_utc, m.ticker, bars, spy, cal)
        features = decide(tw, state).features  # SAME path as batch/endpoint (§3.2)
        sigma = backward_vol(bars, outcome.s0_date, cfg.vol_window_sessions)
        labels = {h: label(outcome.ret[h], sigma, cfg.k) for h in cfg.horizon_days}
        rows.append(
            Row(tw.tweet_id, m.ticker, tw.timestamp_utc, outcome.s0_date,
                m.confidence, features, outcome.ret, labels)
        )
    return rows
