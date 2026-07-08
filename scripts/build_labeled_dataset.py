"""Build the labeled modeling dataset: post x asset x horizon (abnormal return).

Train backbone = Twitter 2017-2021 originals (per locked decision). Each mapped
tweet expands to its candidate sectors -> each sector's ETF + top-5 stocks + SPY.
For each (post, asset, horizon h in 1/3/5):
  raw_h  = asset_close(s0+h-1)/asset_open(s0) - 1
  spy_h  = SPY_close(...)/SPY_open(s0) - 1
  abn_h  = raw_h - spy_h                         # benchmark-adjusted (anti-false-attribution)
  label  = UP/DOWN/NEUTRAL via vol-scaled band  ±k*sigma_backward*sqrt(h)  (§3.5, backward-only)

All feature inputs are strictly pre-t0 (point-in-time). Writes data/real/labeled.csv
and prints 10 event->market sanity examples.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/build_labeled_dataset.py
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config.settings import SETTINGS
from config.universe import SECTOR_STOCKS, TRUMP_EXPOSED, assets_for_sector
from core.calendar import TradingCalendar
from core.market_state import market_state_as_of
from data.sources.local import LocalPriceSource, load_corpus
from labeling.windows import compute_outcome
from sector_mapping.rules import map_tweet_multi

CORPUS = "data/real/corpus.csv"
BARS = "data/real/bars.csv"
OUT = "data/real/labeled.csv"
HORIZONS = (1, 3, 5)
PRE = (1, 3, 5, 10, 20)
K = SETTINGS.k
VOLW = SETTINGS.vol_window_sessions
LO = datetime(2016, 1, 1, tzinfo=timezone.utc)
TRAIN_START = datetime(2017, 1, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2021, 1, 9, tzinfo=timezone.utc)


def pre_features(closes: list[float]) -> dict[str, float] | None:
    """Backward returns + vol from prior closes (chronological). None if too short."""
    if len(closes) < VOLW + 1:
        return None
    feat = {f"r{n}": closes[-1] / closes[-1 - n] - 1.0 for n in PRE}
    dr = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    feat["vol"] = statistics.pstdev(dr[-VOLW:])
    return feat


def label(abn: float | None, vol: float, h: int) -> str | None:
    if abn is None:
        return None
    thr = K * vol * (h ** 0.5)
    return "UP" if abn > thr else "DOWN" if abn < -thr else "NEUTRAL"


def main() -> None:
    tweets = load_corpus(CORPUS, TRAIN_START, TRAIN_END, platforms=("twitter",))
    price = LocalPriceSource(BARS)
    tickers = {"SPY", *SETTINGS.etfs}
    for s in SECTOR_STOCKS.values():
        tickers.update(s)
    bars = {t: price.get_daily_bars(t, LO, TRAIN_END + pd.Timedelta(days=40).to_pytimedelta())
            for t in tickers}
    cal = TradingCalendar([b.session_date.date() for b in bars["SPY"]])

    rows: list[dict[str, object]] = []
    for tw in tweets:
        maps = map_tweet_multi(tw.text)
        if not maps:
            continue
        spy_out = compute_outcome(tw.timestamp_utc, bars["SPY"], cal, HORIZONS)
        if spy_out is None:
            continue
        topic = {f"topic_{e}": 1 for e in SETTINGS.etfs if e in {m.ticker for m in maps}}
        for m in maps:
            etf = m.ticker
            assert etf is not None
            for asset in assets_for_sector(etf):
                ms = market_state_as_of(tw.timestamp_utc, asset, bars[asset], bars["SPY"], cal)
                pf = pre_features([b.close for b in ms.prior_bars])
                out = compute_outcome(tw.timestamp_utc, bars[asset], cal, HORIZONS)
                if pf is None or out is None:
                    continue
                spy5 = pre_features([b.close for b in
                       market_state_as_of(tw.timestamp_utc, "SPY", bars["SPY"], bars["SPY"], cal).prior_bars])
                abn: dict[int, float | None] = {}
                for h in HORIZONS:
                    ra, rs = out.ret[h], spy_out.ret[h]
                    abn[h] = ra - rs if ra is not None and rs is not None else None
                labs = {h: label(abn[h], pf["vol"], h) for h in HORIZONS}
                if labs[5] is None:
                    continue
                hour = tw.timestamp_utc.hour
                row: dict[str, object] = {
                    "post_id": tw.tweet_id, "timestamp_utc": tw.timestamp_utc.isoformat(),
                    "platform": tw.platform, "sector": etf, "asset": asset,
                    "is_etf": int(asset == etf), "is_spy": int(asset == "SPY"),
                    "relevance": round(m.confidence, 4),
                    "trump_exposed": int(asset in TRUMP_EXPOSED),
                    "weekday": tw.timestamp_utc.weekday(), "hour": hour,
                    "after_hours": int(hour < 14 or hour >= 21),  # ~ET pre-open/after-close
                    "rel_spy5": round(pf["r5"] - (spy5["r5"] if spy5 else 0.0), 6),
                    **{f"pre_{k}": round(v, 6) for k, v in pf.items()},
                    **{f"topic_{e}": topic.get(f"topic_{e}", 0) for e in SETTINGS.etfs},
                    "text": tw.text[:140],
                }
                for h in HORIZONS:
                    rv, sv, av = out.ret[h], spy_out.ret[h], abn[h]
                    row[f"raw_{h}"] = None if rv is None else round(rv, 6)
                    row[f"spy_{h}"] = None if sv is None else round(sv, 6)
                    row[f"abn_{h}"] = None if av is None else round(av, 6)
                    row[f"lab_{h}"] = labs[h]
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"labeled rows (post x asset x): {len(df)}  ->  {OUT}")
    print(f"unique posts: {df['post_id'].nunique()}   assets: {df['asset'].nunique()}")
    print("\n5d label balance:", df["lab_5"].value_counts().to_dict())
    print("\n--- 10 EVENT -> MARKET SANITY EXAMPLES (5-day) ---")
    print(f"{'date':11}{'sector':7}{'asset':6}{'raw5d':>8}{'spy5d':>8}{'abn5d':>8}  text")
    for _, r in df[df["is_spy"] == 0].head(10).iterrows():
        def pct(x: Any) -> str:  # noqa: ANN401 — pandas cell
            return f"{float(x)*100:+.2f}" if pd.notna(x) else "   n/a"
        print(f"{str(r['timestamp_utc'])[:10]:11}{r['sector']:7}{r['asset']:6}"
              f"{pct(r['raw_5']):>8}{pct(r['spy_5']):>8}{pct(r['abn_5']):>8}  {str(r['text'])[:52]}")

    # ponytail: one runnable check — abnormal must equal raw - spy on a real row
    s = df[df["abn_5"].notna()].iloc[0]
    assert abs(float(s["abn_5"]) - (float(s["raw_5"]) - float(s["spy_5"]))) < 1e-5


if __name__ == "__main__":
    main()
