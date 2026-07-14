"""Build data/real/labeled_multibench.csv — the reframed labeled dataset.

One row per (tweet x directly/indirectly linked stock). Each row carries, at
every horizon in the ladder, the stock's return and its abnormal return vs THREE
averaged benchmark groups (indices, sector ETFs, peers) plus the folded label:

  UP    = LLM stance positive AND stock beat index AND sector AND peers (> band)
  DOWN  = LLM stance negative AND stock underperformed all three
  NEUTRAL / NA otherwise

Reuses (no second path): load_corpus, entity_matches/entity_spans, the LLM signal
cache (llm/), compute_bench_outcome (labeling/benchmarks.py), benchmarks_for
(config/membership.py). Stance comes from the LLM's direction_of_intent, NOT the
weak lexicon (bullish->positive, bearish->negative, else neutral). Point-in-time
holds because every return flows through compute_outcome.

Daily horizons (EOD..1mo) are fully multi-benchmark. Intraday (30m/1h) is joined
from data/real/intraday_reactions.csv when present and is SECTOR-ONLY + sparse
(index/peer intraday abn needs a bulk intraday fetch that has not been run) — the
coverage is printed, never hidden.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/build_multibench.py            # full corpus
  PYTHONPATH=. .venv/Scripts/python.exe scripts/build_multibench.py --limit 300 # quick
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd

from config.membership import benchmarks_for, name_of
from config.settings import SETTINGS
from core.calendar import TradingCalendar
from core.market_state import market_state_as_of
from data.sources.local import LocalPriceSource, load_corpus
from labeling.benchmarks import compute_bench_outcome
from models.multibench_features import pre_context
from llm.cache import SignalCache
from llm.extract import DEFAULT_MODEL, default_extractor
from sector_mapping.entities import entity_matches

BARS = "data/real/bars.csv"
CORPUS = "data/real/corpus_v3.csv"
SIGNALS = "runs/real/llm_signals.json"
INTRADAY = "data/real/intraday_reactions.csv"
OUT = "data/real/labeled_multibench.csv"

T0, T1 = datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 7, tzinfo=timezone.utc)
LO = datetime(2008, 1, 1, tzinfo=timezone.utc)

# session-offset horizon -> human suffix (EOD = close of s0 = 1 session).
SUF: dict[int, str] = {1: "eod", 2: "2d", 3: "3d", 5: "1w", 10: "2w", 15: "3w", 21: "1mo"}
HORIZONS = SETTINGS.label_horizons

# LLM directional intent -> stance vocabulary used by the label/verdict logic.
_STANCE = {"bullish": "positive", "bearish": "negative", "neutral": "neutral"}


def _load_bars() -> tuple[dict, TradingCalendar]:
    price = LocalPriceSource(BARS)
    # load every ticker in the file so any resolved benchmark/peer is present.
    tickers = sorted(pd.read_csv(BARS, usecols=["ticker"])["ticker"].unique())
    bars = {t: price.get_daily_bars(t, LO, T1 + timedelta(days=100)) for t in tickers}
    cal = TradingCalendar([b.session_date.date() for b in bars["SPY"]])
    return bars, cal


def _stance_for(tweet, cache: SignalCache, extractor, model: str) -> tuple[str, str]:
    """(stance, source). LLM cache first; extract + cache on miss."""
    sig = cache.get(tweet.tweet_id, tweet.text, model)
    src = "cache"
    if sig is None:
        sig = extractor.extract(tweet.text)
        cache.put(tweet.tweet_id, tweet.text, model, sig)
        src = "fresh"
    return _STANCE.get(sig.direction_of_intent, "neutral"), src


def _intraday_index(path: str) -> dict[tuple[str, str], dict]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    return {(str(r["post_id"]), str(r["entity"])): r for _, r in df.iterrows()}


def _coarse_intraday_label(stance: str, abn: float | None) -> str:
    """Sector-only intraday verdict. Fixed 0.3% flat band (no per-window vol)."""
    if abn is None or stance not in ("positive", "negative"):
        return "NA" if stance in ("positive", "negative") else "NEUTRAL"
    if abs(abn) < 0.003:
        return "NEUTRAL"
    up = abn > 0
    return ("UP" if up else "DOWN") if (up == (stance == "positive")) else "NEUTRAL"


def build(limit: int | None = None) -> None:
    bars, cal = _load_bars()
    tweets = load_corpus(CORPUS, T0, T1)
    if limit:
        tweets = tweets[:limit]

    model = DEFAULT_MODEL
    extractor = default_extractor(model)
    cache = SignalCache(SIGNALS)
    intraday = _intraday_index(INTRADAY)

    rows: list[dict] = []
    balance: dict[str, Counter] = {SUF[h]: Counter() for h in HORIZONS}
    intra_cov = Counter()
    stance_src = Counter()

    for tw in tweets:
        stance, src = _stance_for(tw, cache, extractor, model)
        stance_src[src] += 1
        for tk, m in entity_matches(tw.text).items():
            if m.tier == "competitor":
                continue
            o = compute_bench_outcome(tw.timestamp_utc, tk, bars, cal, HORIZONS)
            if o is None:
                continue
            bset = benchmarks_for(tk)
            # pre-t0 features (point-in-time): vol + prior returns from closed bars.
            st = market_state_as_of(tw.timestamp_utc, tk, bars[tk], bars["SPY"], cal)
            pv, pr1, pr3 = pre_context([b.close for b in st.prior_bars])
            t0 = tw.timestamp_utc
            hour = t0.hour + t0.minute / 60.0
            after_hours = int(t0.weekday() >= 5 or not (13.5 <= hour < 20.0))
            row: dict[str, object] = {
                "post_id": tw.tweet_id, "tweet_date": t0.date().isoformat(),
                "entity": tk, "entity_name": name_of(tk), "match_tier": m.tier,
                "stance": stance, "stance_source": src,
                "indices_used": " ".join(bset.indices),
                "sectors_used": " ".join(bset.sectors),
                "peers_used": " ".join(bset.peers),
                "used_fallback": int(bset.used_fallback),
                "pre_vol": _r(pv), "pre_ret_1": _r(pr1), "pre_ret_3": _r(pr3),
                "weekday": t0.weekday(), "after_hours": after_hours,
                "text": tw.text[:200],
            }
            for h in HORIZONS:
                s = SUF[h]
                lab = o.label(stance, h)
                balance[s][lab] += 1
                row[f"raw_{s}"] = _r(o.raw[h])
                row[f"abn_idx_{s}"] = _r(o.abn_index[h])
                row[f"abn_sec_{s}"] = _r(o.abn_sector[h])
                row[f"abn_peer_{s}"] = _r(o.abn_peer[h])
                row[f"band_{s}"] = _r(o.band[h])
                row[f"label_{s}"] = lab

            # intraday (sector-only, sparse) joined from the reaction panel.
            ir = intraday.get((tw.tweet_id, tk))
            for win in ("30m", "1h"):
                abn = None
                if ir is not None and f"abn_{win}" in ir and not pd.isna(ir[f"abn_{win}"]):
                    abn = float(ir[f"abn_{win}"])
                    intra_cov[win] += 1
                row[f"abn_sec_{win}"] = _r(abn)
                row[f"label_{win}"] = _coarse_intraday_label(stance, abn)
            rows.append(row)

    cache.save()
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print(f"wrote {len(df)} rows ({df['post_id'].nunique()} tweets) -> {OUT}")
    print(f"stance source: {dict(stance_src)}  (fresh = extracted now; set "
          f"ANTHROPIC_API_KEY for Claude stance, else deterministic heuristic)")
    print("\nclass balance per daily horizon (§3.5 diagnostic — sparse UP/DOWN is expected):")
    for h in HORIZONS:
        s = SUF[h]
        b = balance[s]
        print(f"  {s:4}  UP={b['UP']:4} DOWN={b['DOWN']:4} NEUTRAL={b['NEUTRAL']:5} NA={b['NA']:4}")
    print(f"\nintraday coverage (sector-only): 30m={intra_cov['30m']} 1h={intra_cov['1h']} "
          f"of {len(df)} rows — bulk intraday fetch unlocks index/peer intraday abn.")


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


if __name__ == "__main__":
    lim = None
    if "--limit" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1])
    build(lim)
