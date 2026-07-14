"""Micro track — single-stock tweets under the dynamic-benchmark, EOD relative-alpha standard.

Router: a tweet is MICRO when it directly names one company in our stock universe
(sector_mapping.entities). For that stock we resolve its DYNAMIC benchmark from
config.membership.benchmarks_for (its indices + sector ETF(s) + peer group) and score with
labeling.benchmarks.compute_bench_outcome — a HIT requires the stock to beat ALL THREE
(mean(indices) AND sector AND peer-median) beyond its vol-band, in the LLM-predicted direction.

Primary label = EOD (matches the macro track's front-loaded finding). Secondary = 1mo (full
three-benchmark, daily) and 30m/1h (sector-relative only, --intraday; index/peer intraday needs
the bulk Alpaca fetch noted in docs/DUAL_TRACK_ARCHITECTURE.md).

Emits reports/micro_dataset.csv with the SAME chronological 60/20/20 split structure as the macro
dataset, so the two feed one combined predictor.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/nebius_micro_backtest.py --limit 120
  ... --intraday          # also fill 30m/1h (sector-relative, slower Alpaca calls)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from config.membership import benchmarks_for, name_of
from config.universe import all_stocks
from core.calendar import TradingCalendar
from data.sources.local import LocalPriceSource, load_corpus
from labeling.benchmarks import compute_bench_outcome
from scripts.nebius_macro_backtest import (
    execution_window, has_action_verbs, is_targeted_action, session_phase,
)
from scripts.nebius_macro_validate import (
    _env, _load_dotenv, _parse_json, intraday_returns, relative_hit,
)
from sector_mapping.entities import entity_matches

BARS = "data/real/bars.csv"
CORPUS = "data/real/corpus_v3.csv"
CACHE = "runs/real/nebius_micro_predictions.json"
DATASET = "reports/micro_dataset.csv"
DATASET_CFG = "reports/micro_dataset_config.json"
PROMPT_VERSION = "micro-v1"

T0, T1 = datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 7, tzinfo=timezone.utc)
LO = datetime(2008, 1, 1, tzinfo=timezone.utc)
EOD_SESSION, MO_SESSION = 1, 21          # EOD = close of entry session; 1mo = 21 sessions
STOCKS = frozenset(all_stocks())

_SYSTEM = ("You judge whether a political social-media post is bullish or bearish for ONE named "
           "company's stock, reasoning from the post's content alone (no market data). Return ONLY JSON.")


def _micro_prompt(text: str, ticker: str, name: str) -> str:
    return (f'Post is about {name} ({ticker}). Return ONLY JSON:\n'
            f'{{"direction":"up|down|neutral","intensity":7,'
            f'"summary":"plain-English what the post says about {name}",'
            f'"macro_link":"why this should move {ticker} vs its peers",'
            f'"hypothesis_short":"expected 30m-1h reaction vs sector/peers and WHY",'
            f'"hypothesis_long":"expected drift by EOD/1mo vs sector/peers and WHY"}}\n'
            f'direction = the stock\'s expected move RELATIVE to its sector/peers. '
            f'intensity 1-10. Keep text fields under 200 chars.\n\nPOST:\n{text}')


def micro_classify(text: str, ticker: str, name: str, *, base_url: str, api_key: str,
                   model: str) -> dict:
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "system", "content": _SYSTEM},
                           {"role": "user", "content": _micro_prompt(text, ticker, name)}]},
        timeout=90,
    )
    if resp.status_code != 200:
        sys.exit(f"Nebius API error {resp.status_code}: {resp.text[:300]}")
    try:
        return _parse_json(resp.json()["choices"][0]["message"]["content"])
    except Exception:   # malformed JSON from the model -> neutral (dropped downstream), not a crash
        return {"direction": "neutral", "_parse_failed": True}


def _load_bars() -> tuple[dict, TradingCalendar]:
    price = LocalPriceSource(BARS)
    tickers = sorted(pd.read_csv(BARS, usecols=["ticker"])["ticker"].unique())
    bars = {t: price.get_daily_bars(t, LO, T1 + timedelta(days=100)) for t in tickers}
    cal = TradingCalendar([b.session_date.date() for b in bars["SPY"]])
    return bars, cal


def _micro_candidates(limit: int) -> list:
    """Direct single-company mentions (the router's MICRO branch), deduped by (ticker, date)."""
    tweets = load_corpus(CORPUS, T0, T1)
    seen: set[tuple[str, str]] = set()
    out = []
    for tw in tweets:
        tk = next((t for t, m in entity_matches(tw.text).items()
                   if m.tier == "direct" and t in STOCKS), None)
        if tk is None:
            continue
        key = (tk, tw.timestamp_utc.date().isoformat())
        if key in seen:
            continue
        seen.add(key)
        out.append((tw, tk))
    out.sort(key=lambda x: x[0].timestamp_utc)
    if len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


def _dyn_hit(o, direction: str, h_session: int) -> bool | None:
    """Beat-all-three (mean(indices) AND sector AND peer-median) in the predicted direction
    at horizon h. Strict beat (band=0, abn>0/<0) to match the macro track's Option-B baseline
    and keep the null a clean ~50% — the vol-band `outperformed()` here is far too strict for
    a 1-day EOD move and yields a degenerate all-miss label."""
    if o is None or direction not in ("up", "down"):
        return None
    legs = [o.abn_index.get(h_session), o.abn_sector.get(h_session), o.abn_peer.get(h_session)]
    if any(a is None for a in legs):
        return None
    return all(a > 0 for a in legs) if direction == "up" else all(a < 0 for a in legs)


def _soft_hit(o, direction: str, h_session: int) -> bool | None:
    """SOFT blended label: stock beats the BLENDED baseline mean(indices, sector, peers)
    as one wall, not three. stock_ret - mean(groups) = mean(abn_index, abn_sector, abn_peer),
    so hit = that blended abn > 0 (up) / < 0 (down)."""
    if o is None or direction not in ("up", "down"):
        return None
    legs = [o.abn_index.get(h_session), o.abn_sector.get(h_session), o.abn_peer.get(h_session)]
    if any(a is None for a in legs):
        return None
    blended = sum(legs) / len(legs)     # type: ignore[arg-type]
    return blended > 0 if direction == "up" else blended < 0


def run(limit: int, model: str, base_url: str, api_key: str, do_intraday: bool) -> None:
    bars, cal = _load_bars()
    cache = json.loads(Path(CACHE).read_text()) if Path(CACHE).exists() else {}
    cands = _micro_candidates(limit)
    print(f"-> {len(cands)} MICRO (single-stock) tweets, model={model}, "
          f"intraday={'on' if do_intraday else 'off (EOD+1mo full; 30m/1h skipped)'}\n")

    rows: list[dict] = []
    for i, (tw, tk) in enumerate(cands, 1):
        ck = f"{tw.tweet_id}|{model}|{PROMPT_VERSION}"
        if ck in cache:
            pred = cache[ck]
        else:
            pred = micro_classify(tw.text, tk, name_of(tk), base_url=base_url,
                                  api_key=api_key, model=model)
            cache[ck] = pred
            Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
            Path(CACHE).write_text(json.dumps(cache, indent=1))

        direction = str(pred.get("direction", "neutral")).lower()
        bset = benchmarks_for(tk)
        o = compute_bench_outcome(tw.timestamp_utc, tk, bars, cal, (EOD_SESSION, MO_SESSION))
        eod = _dyn_hit(o, direction, EOD_SESSION)
        if eod is None:          # not scoreable at EOD -> not a training example
            continue
        mo = _dyn_hit(o, direction, MO_SESSION)

        h30 = h1h = None
        if do_intraday and bset.sectors:
            si = intraday_returns(tk, tw.timestamp_utc)
            ei = intraday_returns(bset.sectors[0], tw.timestamp_utc)  # sector-relative only
            h30 = relative_hit(direction, si.get("30m"), ei.get("30m"))
            h1h = relative_hit(direction, si.get("1h"), ei.get("1h"))

        t0 = tw.timestamp_utc
        # SIGNED strict label: +1 winning long, -1 winning short (beat all 3 downward), 0 else.
        signed = (1 if direction == "up" else -1 if direction == "down" else 0) if eod else 0
        rows.append({
            "post_date": t0.date().isoformat(), "hour_utc": round(t0.hour + t0.minute / 60.0, 2),
            "phase": session_phase(t0), "weekend_flag": int(session_phase(t0) == "weekend"),
            "market_closed": int(session_phase(t0) != "regular"),
            "intensity": pred.get("intensity"), "ticker": tk, "ticker_name": name_of(tk),
            "indices": " ".join(bset.indices), "sector": " ".join(bset.sectors),
            "n_peers": len(bset.peers), "track": "micro",
            # engineered common-denominator features:
            "is_targeted_action": is_targeted_action(tw.text),
            "has_action_verbs": has_action_verbs(tw.text),
            "execution_window": execution_window(t0),
            # LLM reasoning text -> semantic features for the engine (Option C):
            "summary": pred.get("summary", ""), "macro_link": pred.get("macro_link", ""),
            "label_eod": signed,                    # SIGNED: +1 long-win / -1 short-win / 0 noise
            "label_eod_soft": None if (sh := _soft_hit(o, direction, EOD_SESSION)) is None
            else int(sh),                                            # soft: beat blended mean
            "label_1mo_soft": None if (mh := _soft_hit(o, direction, MO_SESSION)) is None
            else int(mh),
            "hit_30m": None if h30 is None else int(h30),
            "hit_1h": None if h1h is None else int(h1h),
            "hit_1mo": None if mo is None else int(mo),
        })
        print(f"  [{i}/{len(cands)}] {t0.date()} {tk:5} {name_of(tk)[:18]:18} "
              f"dir={direction:8} EOD-beat-all3={'HIT' if eod else 'miss'}")

    _emit(rows)


def _emit(rows: list[dict]) -> None:
    rows.sort(key=lambda x: x["post_date"])
    n = len(rows)
    tr, va = int(round(0.6 * n)), int(round(0.8 * n))
    for i, r in enumerate(rows):
        r["split"] = "train" if i < tr else "val" if i < va else "test"

    Path(DATASET).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(DATASET, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}

    def rate(label: str) -> float | None:
        vals = [r[label] for r in rows if r.get(label) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    longs = sum(r["label_eod"] == 1 for r in rows)
    shorts = sum(r["label_eod"] == -1 for r in rows)
    noise = sum(r["label_eod"] == 0 for r in rows)
    cfg = {
        "track": "micro", "label": "label_eod", "label_type": "signed_multiclass",
        "primary_horizon": "EOD",
        "classes": {"+1": "predicted UP and stock beat all 3 benchmarks (winning LONG)",
                    "-1": "predicted DOWN and stock beat all 3 downward (winning SHORT)",
                    "0": "wrong direction / noise / inside band"},
        "labels_secondary": {
            "label_eod_soft": "SOFT: stock beat the BLENDED baseline mean(indices, sector, peers) at EOD",
            "label_1mo_soft": "SOFT blended baseline at 1 month."},
        "benchmark": "config.membership.benchmarks_for -> labeling.benchmarks.compute_bench_outcome",
        "features": ["intensity", "phase", "weekend_flag", "market_closed", "ticker", "sector",
                     "is_targeted_action", "has_action_verbs", "execution_window"],
        "split": "time_ordered_60_20_20", "n_total": n,
        "n_train": counts["train"], "n_val": counts["val"], "n_test": counts["test"],
        "class_balance": {"+1_long": longs, "-1_short": shorts, "0_noise": noise},
        "win_rate": round((longs + shorts) / n, 3) if rows else None,
        "positive_rate_soft": rate("label_eod_soft"),
    }
    Path(DATASET_CFG).write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("MICRO DATASET (single-stock, dynamic benchmark, SIGNED EOD label)")
    print("=" * 80)
    print(f"  -> {DATASET}   ({n} tweets: {counts['train']} train / {counts['val']} val "
          f"/ {counts['test']} test)")
    print(f"  -> {DATASET_CFG}")
    print(f"  label_eod SIGNED  balance: +1={longs} / -1={shorts} / 0={noise}  "
          f"win_rate={cfg['win_rate']}  (soft positive_rate={cfg['positive_rate_soft']})")
    print("  engineered: is_targeted_action, has_action_verbs, execution_window")
    print("\n  Signed win-rate by split (|label_eod|!=0; long / short):")
    for s in ("train", "val", "test"):
        grp = [r for r in rows if r["split"] == s]
        if grp:
            lo = sum(r["label_eod"] == 1 for r in grp)
            sh = sum(r["label_eod"] == -1 for r in grp)
            print(f"    {s:5} n={len(grp):3}  win-rate={100*(lo+sh)/len(grp):5.1f}%  "
                  f"(long {lo} / short {sh} / noise {len(grp)-lo-sh})")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--intraday", action="store_true", help="also fill 30m/1h (sector-relative)")
    ap.add_argument("--model", default=_env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL",
                                            default="meta-llama/Llama-3.3-70B-Instruct"))
    ap.add_argument("--base-url", default=_env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL",
                                               default="https://api.studio.nebius.ai/v1"))
    a = ap.parse_args()
    api_key = _env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    if not api_key:
        sys.exit("No NEBIUS_API_KEY. Put it in .env (gitignored) or export it.")
    run(a.limit, a.model, a.base_url, api_key, a.intraday)


if __name__ == "__main__":
    main()
