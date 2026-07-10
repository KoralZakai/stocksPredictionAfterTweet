"""One-off CSV report: latest 10 market-relevant Trump posts w/ classification +
pre/post market move + sector siblings, for eyeballing correlation.

NOT a model, NOT a claim of signal (see CLAUDE.md sec 1: any read here is
observational, not causal). Reuses the already-built per-(post,asset) rows in
data/real/stock_event_dataset.csv (scripts/build_stock_dataset_v3.py) so pre/post
returns are the same point-in-time-correct numbers used everywhere else in the
pipeline. Rows near "today" are naturally absent from that dataset because the
5d-forward window isn't closed yet (no leakage workaround, no fabrication).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/report_latest10.py
"""

from __future__ import annotations

import pandas as pd

from config.universe import SECTOR_STOCKS

IN = "data/real/stock_event_dataset.csv"
OUT = "reports/latest10_tweets_market_moves.csv"

N_TWEETS = 10
PRE_H = 5
POST_H = 5


def pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x * 100:+.2f}%"


def main() -> None:
    df = pd.read_csv(IN)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, format="ISO8601")

    post_ids = (
        df.drop_duplicates("post_id")
        .sort_values("timestamp_utc", ascending=False)
        .head(N_TWEETS)["post_id"]
    )

    rows = []
    for pid in post_ids:
        g = df[df["post_id"] == pid]
        text = g["text"].iloc[0]
        ts = g["timestamp_utc"].iloc[0]

        # primary = highest-relevance non-macro-catchall asset (prefer the sector
        # ETF / direct-entity row over the individual member-stock rows).
        candidates = g[g["asset"] != "SPY"]
        if candidates.empty:
            candidates = g
        primary = candidates.sort_values(
            ["relevance", "is_etf"], ascending=[False, False]
        ).iloc[0]

        sector = primary["sectors"] if pd.notna(primary["sectors"]) else "(direct entity, no sector ETF)"
        etf = sector.split("/")[0] if isinstance(sector, str) and "/" in sector else sector
        siblings_universe = SECTOR_STOCKS.get(etf, ())

        sib_rows = g[
            g["asset"].isin(siblings_universe) & (g["asset"] != primary["asset"])
        ]
        sib_str = "; ".join(
            f"{r.asset} post:{pct(getattr(r, f'raw_{POST_H}'))}"
            for r in sib_rows.itertuples()
        ) or "n/a"

        rows.append(
            {
                "tweet_date_utc": ts.strftime("%Y-%m-%d %H:%M"),
                "tweet_text": text,
                "primary_asset": primary["asset"],
                "sector": sector,
                "explanation": primary["explanation"],
                "market_move_before_tweet_5d": pct(primary[f"pre_ret_{PRE_H}"]),
                "market_move_after_tweet_5d": pct(primary[f"raw_{POST_H}"]),
                "abnormal_move_after_vs_spy_5d": pct(primary[f"abn_{POST_H}"]),
                "other_stocks_same_sector_5d_after": sib_str,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}")
    print(f"newest tweet in dataset: {out['tweet_date_utc'].iloc[0]}")
    print(
        "note: dataset caps at 2026-06-29 because later posts don't yet have a "
        "closed 5-trading-day forward window in bars.csv (point-in-time invariant, "
        "not a bug)."
    )


if __name__ == "__main__":
    main()
