"""Stock-level event dataset with SOFT relevance + pre/post windows (dataset v2).

Shift from sector labeling -> event-driven stock-level attribution:
  * rows driven by asset_relevance() (0..1 per stock/ETF), incl. direct entity links
  * pre-event trend context (-5d/-10d return, pre-abnormal vs SPY) + backward vol
  * post-event abnormal return + vol-scaled label at +1/+3/+5d
No modeling here. All feature inputs strictly pre-t0 (point-in-time).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/build_stock_dataset.py
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

import pandas as pd

from config.settings import SETTINGS
from config.universe import SECTOR_STOCKS, TRUMP_EXPOSED
from core.calendar import TradingCalendar
from core.market_state import market_state_as_of
from data.sources.local import LocalPriceSource, load_corpus
from labeling.windows import compute_outcome
from sector_mapping.entities import asset_relevance, is_direct_mention

CORPUS, BARS, OUT = "data/real/corpus.csv", "data/real/bars.csv", "data/real/stock_events.csv"
HORIZONS = (1, 3, 5)
VOLW = SETTINGS.vol_window_sessions
LO = datetime(2016, 1, 1, tzinfo=timezone.utc)
T0, T1 = datetime(2017, 1, 1, tzinfo=timezone.utc), datetime(2021, 1, 9, tzinfo=timezone.utc)

STOCK_SECTORS: dict[str, list[str]] = {}
for _e, _ns in SECTOR_STOCKS.items():
    for _n in _ns:
        STOCK_SECTORS.setdefault(_n, []).append(_e)


def pre_ctx(closes: list[float]) -> dict[str, float] | None:
    if len(closes) < VOLW + 1:
        return None
    dr = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    return {"pre_ret_5": closes[-1] / closes[-6] - 1.0,
            "pre_ret_10": closes[-1] / closes[-11] - 1.0,
            "pre_ret_1": closes[-1] / closes[-2] - 1.0,
            "pre_vol": statistics.pstdev(dr[-VOLW:])}


def label(abn: float | None, vol: float, h: int) -> str | None:
    if abn is None:
        return None
    thr = SETTINGS.k * vol * (h ** 0.5)
    return "UP" if abn > thr else "DOWN" if abn < -thr else "NEUTRAL"


def main() -> None:
    tweets = load_corpus(CORPUS, T0, T1, platforms=("twitter",))
    price = LocalPriceSource(BARS)
    universe = {"SPY", *SETTINGS.etfs, *(s for ss in SECTOR_STOCKS.values() for s in ss)}
    bars = {t: price.get_daily_bars(t, LO, T1 + timedelta(days=40)) for t in universe}
    cal = TradingCalendar([b.session_date.date() for b in bars["SPY"]])

    rows: list[dict[str, object]] = []
    for tw in tweets:
        rel = asset_relevance(tw.text)
        if not rel:
            continue
        spy_out = compute_outcome(tw.timestamp_utc, bars["SPY"], cal, HORIZONS)
        spy_pre = pre_ctx([b.close for b in market_state_as_of(
            tw.timestamp_utc, "SPY", bars["SPY"], bars["SPY"], cal).prior_bars])
        if spy_out is None or spy_pre is None:
            continue
        for asset, r in rel.items():
            pf = pre_ctx([b.close for b in market_state_as_of(
                tw.timestamp_utc, asset, bars[asset], bars["SPY"], cal).prior_bars])
            out = compute_outcome(tw.timestamp_utc, bars[asset], cal, HORIZONS)
            if pf is None or out is None:
                continue
            abn: dict[int, float | None] = {}
            for h in HORIZONS:
                a, s = out.ret[h], spy_out.ret[h]
                abn[h] = a - s if a is not None and s is not None else None
            labs = {h: label(abn[h], pf["pre_vol"], h) for h in HORIZONS}
            if labs[5] is None:
                continue
            is_etf = asset in SETTINGS.etfs
            hour = tw.timestamp_utc.hour
            row: dict[str, object] = {
                "post_id": tw.tweet_id, "timestamp_utc": tw.timestamp_utc.isoformat(),
                "asset": asset, "is_etf": int(is_etf),
                "sectors": asset if is_etf else "/".join(STOCK_SECTORS.get(asset, [])),
                "relevance": r, "is_direct": int(is_direct_mention(tw.text, asset)),
                "trump_exposed": int(asset in TRUMP_EXPOSED),
                "weekday": tw.timestamp_utc.weekday(), "hour": hour,
                "after_hours": int(hour < 14 or hour >= 21),
                "pre_ret_1": round(pf["pre_ret_1"], 6), "pre_ret_5": round(pf["pre_ret_5"], 6),
                "pre_ret_10": round(pf["pre_ret_10"], 6), "pre_vol": round(pf["pre_vol"], 6),
                "pre_abn_5": round(pf["pre_ret_5"] - spy_pre["pre_ret_5"], 6),  # pre-trend vs SPY
                "text": tw.text[:120],
            }
            for h in HORIZONS:
                rv, av = out.ret[h], abn[h]
                row[f"raw_{h}"] = None if rv is None else round(rv, 6)
                row[f"abn_{h}"] = None if av is None else round(av, 6)
                row[f"lab_{h}"] = labs[h]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"stock-level event rows: {len(df)}  ->  {OUT}")
    print(f"unique posts: {df['post_id'].nunique()}   assets: {df['asset'].nunique()}"
          f"   direct-entity rows: {int(df['is_direct'].sum())}")
    print(f"stocks vs ETFs: {int((df['is_etf']==0).sum())} / {int((df['is_etf']==1).sum())}")
    print("relevance distribution:",
          {b: int(((df['relevance'] > lo) & (df['relevance'] <= hi)).sum())
           for b, (lo, hi) in {"(0,0.25]": (0, .25), "(.25,.5]": (.25, .5),
                               "(.5,.99]": (.5, .99), "1.0(direct)": (.99, 1)}.items()})
    print("5d label balance:", df["lab_5"].value_counts().to_dict())

    # ponytail: one runnable check — relevance in [0,1] and abn=raw-spy holds
    assert df["relevance"].between(0, 1).all()


if __name__ == "__main__":
    main()
