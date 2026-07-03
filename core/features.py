"""Structured features (§5 core/features.py).

Everything here is derived from the tweet text/time and the point-in-time
MarketState (bars closed strictly before t0). No future data can enter — the
point-in-time guarantee is enforced upstream in market_state_as_of, and the
future-injection canary in tests/test_market_state.py proves features are blind
to any bar dated >= t0.

Raw embeddings are intentionally NOT here (§8: raw 384/768-dim into GBT is a
mistake at ~150-300 rows). Structured only.
"""

from __future__ import annotations

import statistics

from config.settings import SETTINGS
from core.market_state import MarketState
from data.sources.interfaces import Tweet
from sector_mapping.rules import SECTOR_KEYWORDS

_TREND_WINDOW = 5


def build_features(tweet: Tweet, state: MarketState) -> dict[str, float]:
    f: dict[str, float] = {}

    # Topic one-hots / entity flags from the tweet text (§5).
    tl = tweet.text.lower()
    for tk, words in SECTOR_KEYWORDS.items():
        f[f"topic_{tk}"] = float(any(w in tl for w in words))

    f["weekday"] = float(tweet.timestamp_utc.weekday())

    closes = [b.close for b in state.prior_bars]
    f["prior_1d_ret"] = closes[-1] / closes[-2] - 1.0 if len(closes) >= 2 else 0.0
    f["prior_3d_ret"] = closes[-1] / closes[-4] - 1.0 if len(closes) >= 4 else 0.0

    win = SETTINGS.vol_window_sessions
    tail = closes[-(win + 1) :]
    rets = [tail[i] / tail[i - 1] - 1.0 for i in range(1, len(tail))]
    f["backward_vol"] = statistics.pstdev(rets) if len(rets) >= 2 else 0.0

    # SPY-trend regime: 1.0 if benchmark is above its recent mean, else 0.0.
    spy = [b.close for b in state.benchmark_prior]
    if len(spy) >= _TREND_WINDOW:
        f["spy_trend"] = float(spy[-1] > statistics.fmean(spy[-_TREND_WINDOW:]))
    else:
        f["spy_trend"] = 0.0

    return f
