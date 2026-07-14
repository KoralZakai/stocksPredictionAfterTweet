"""Live macro-hypothesis pipeline: Nebius LLM -> predicted instruments -> validate vs real market.

End-to-end, self-contained (no repo modules required):
  1. Send a tweet to a model on Nebius Token Factory (OpenAI-compatible /chat/completions).
  2. The model classifies the tweet into a macro scenario and returns the instruments it
     should move (indices / sector ETFs / commodities / barometers) with a predicted
     direction (up/down/neutral).
  3. yfinance fetches the REAL daily bars for those instruments around the tweet date.
  4. Validation: for each instrument, did the market move the way the LLM predicted?
     Prints a per-instrument table + a hit-rate ("correlation") per horizon.

Point-in-time: entry = the first session OPEN strictly after the tweet timestamp (no
same-day leak), returns measured forward from there. This is an ASSOCIATION check on one
tweet — descriptive, zero evidential weight; it does not prove causation.

Setup (key stays uncommitted — .env is gitignored):
  1. Get a key at https://studio.nebius.com  (Token Factory / AI Studio).
  2. Put it in .env  (or export it):   NEBIUS_API_KEY=<your-key>
     Optional:  NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
                NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1
     (This repo's .env also accepts EXPO_PUBLIC_NEBIUS_API_KEY / _BASE_URL / _MODEL.)

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/nebius_macro_validate.py
  ... --text "HOPEFULLY RUSSIA AND UKRAINE WILL MAKE A DEAL THIS WEEK..." --date 2025-04-20
  ... --model mistralai/Mistral-Nemo-Instruct-2407
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------- config / secrets
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # valid Nebius id; `--model` to override
DEFAULT_BASE = "https://api.studio.nebius.ai/v1"
US_OPEN_UTC_HOUR = 13.5  # ~9:30 ET; a session's open is "after t0" if this hour > t0.

# Two timescales, ONE ordered ladder:
#   30m/1h  -> Alpaca 1-min IEX bars (yfinance has no intraday history past ~60d).
#   EOD..1mo -> yfinance daily bars, in trading sessions (EOD = close of entry session).
INTRADAY_WINS: dict[str, int] = {"30m": 30, "1h": 60}       # minutes after t0 becomes actionable
DAILY_SESS: dict[str, int] = {"EOD": 1, "3d": 3, "1w": 5, "1mo": 21}
HORIZONS: list[str] = ["30m", "1h", "EOD", "3d", "1w", "1mo"]  # display / scoring order

# A default worked example — the Russia/Ukraine "peace deal" tweet (macro, not a chip stock).
DEMO_TEXT = ("HOPEFULLY RUSSIA AND UKRAINE WILL MAKE A DEAL THIS WEEK. BOTH WILL THEN "
             "START TO DO BIG BUSINESS WITH THE UNITED STATES OF AMERICA, WHICH IS "
             "THRIVING, AND MAKE A FORTUNE!")
DEMO_DATE = "2025-04-20"


def _load_dotenv(path: str = ".env") -> None:
    """Minimal stdlib .env loader (shell env wins). Keeps the key out of the code."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


# ---------------------------------------------------------------- 1) Nebius LLM call
_SYSTEM = (
    "You are a macro markets analyst. Read a political social-media post and decide, from "
    "the CONTENT ALONE (not any market data), which scenario it fits and which liquid, "
    "US-listed instruments its content should push, and in which direction. Reason from the "
    "event itself. Watch for typos: 'RUSSIA AMD UKRAINE' means 'AND', not the chip company AMD. "
    "Return ONLY a JSON object, no prose, no markdown fences."
)
_INSTRUCT = (
    'Return ONLY a JSON object with this exact shape:\n'
    '{"scenario":"Geopolitics / Peace","intensity":7,'
    '"summary":"plain-English: what the person actually said, one sentence a layperson gets",'
    '"macro_link":"the economic logic: why this scenario moves markets",'
    '"hypothesis_short":"what should happen in the first 30m-1h and WHY (immediate reaction)",'
    '"hypothesis_long":"what should happen by 1 month and WHY (as the policy plays out)",'
    '"rationale":"one-sentence thesis",'
    '"instruments":[{"ticker":"ITA","name":"Defense & Aerospace ETF",'
    '"role":"bearish_sector","predicted_direction":"down"}]}\n'
    "intensity = integer 1-10: how forceful, certain and market-moving the post is "
    "(10 = decisive policy action stated as fact / a done deal; 1 = vague musing or opinion). "
    "predicted_direction in {up,down,neutral}. Use real yfinance-valid symbols "
    "(indices SPY/QQQ/DIA; sectors XLI/ITA/XLE/XLK/XLF/XLV; infra PAVE; fear VIXY; "
    "commodities USO/WEAT/CORN/DBC; barometers CAT/LMT/AAPL). 3-8 instruments. "
    "Keep every text field under 240 characters. "
    "If the post is not market-relevant, return an empty instruments list."
)


def classify_tweet(text: str, *, base_url: str, api_key: str, model: str) -> dict:
    """Call Nebius (OpenAI-compatible) and parse the JSON scenario/instrument prediction."""
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"{_INSTRUCT}\n\nPOST:\n{text}"},
            ],
        },
        timeout=90,
    )
    if resp.status_code != 200:
        sys.exit(f"Nebius API error {resp.status_code}: {resp.text[:400]}")
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_json(content)


def _parse_json(content: str) -> dict:
    """Tolerant JSON extraction: strip ``` fences, grab the outermost {...}."""
    s = content.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1].lstrip("json").strip() if "```" in s[3:] else s.strip("`")
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        sys.exit(f"model did not return JSON:\n{content[:400]}")
    return json.loads(s[a : b + 1])


# ---------------------------------------------------------------- 2) real market data
def _sessions(df: pd.DataFrame) -> list[tuple[datetime, float, float]]:
    """(date, open, close) rows, sorted, tz-naive dates."""
    out = []
    for ts, r in df.iterrows():
        d = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
        o, c = float(r["Open"]), float(r["Close"])
        if o == o and c == c:  # drop NaN
            out.append((d, o, c))
    return sorted(out, key=lambda x: x[0])


def daily_returns(ticker: str, t0: datetime) -> dict[str, float]:
    """{EOD/3d/1w/1mo: cumulative return from the entry open}. Entry = first session
    whose open is strictly after t0 (no same-day leak)."""
    df = yf.download(ticker, start=(t0 - timedelta(days=10)).date(),
                     end=(t0 + timedelta(days=45)).date(), interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return {}
    if isinstance(df.columns, pd.MultiIndex):       # single-ticker MultiIndex -> flatten
        df.columns = df.columns.get_level_values(0)
    s = _sessions(df)
    t0h = t0.hour + t0.minute / 60.0
    i0 = next((i for i, (d, _, _) in enumerate(s)
               if d.date() > t0.date() or (d.date() == t0.date() and US_OPEN_UTC_HOUR > t0h)), None)
    if i0 is None:
        return {}
    entry = s[i0][1]
    out: dict[str, float] = {}
    for name, h in DAILY_SESS.items():
        j = i0 + h - 1
        if j < len(s):
            out[name] = s[j][2] / entry - 1.0
    return out


def _session_anchor(t0: datetime) -> datetime:
    """When the tweet becomes tradeable: t0 if in regular hours, else the next
    session open (~13:30 UTC). Ignores holidays — fine for an intraday reaction."""
    hour = t0.hour + t0.minute / 60.0
    if t0.weekday() < 5 and US_OPEN_UTC_HOUR <= hour < 20.0:
        return t0
    if t0.weekday() < 5 and hour < US_OPEN_UTC_HOUR:
        return t0.replace(hour=13, minute=30, second=0, microsecond=0)
    d = (t0 + timedelta(days=1)).replace(hour=13, minute=30, second=0, microsecond=0)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def intraday_returns(ticker: str, t0: datetime) -> dict[str, float]:
    """{30m/1h: return over that window after the tweet is actionable}, from Alpaca
    1-min IEX bars. {} if no creds / no bars (thin IEX coverage is honest, not fatal)."""
    try:
        from scripts.fetch_alpaca_30m import KEY, SECRET, fetch_window
    except Exception:
        return {}
    if not (KEY and SECRET):
        return {}
    ref = _session_anchor(t0)
    try:
        rows = fetch_window([ticker], ref - timedelta(minutes=20),
                            ref + timedelta(minutes=max(INTRADAY_WINS.values()) + 10))
    except Exception:
        return {}
    bars = sorted(((pd.Timestamp(r["ts_utc"]).to_pydatetime(), float(r["close"]))
                   for r in rows if r.get("ticker") == ticker and r.get("close") is not None),
                  key=lambda x: x[0])
    if not bars:
        return {}

    def price_at(when: datetime) -> float | None:
        prior = [c for ts, c in bars if ts <= when]
        return prior[-1] if prior else None

    base = price_at(ref)
    if base is None or base == 0:
        return {}
    out: dict[str, float] = {}
    for name, mins in INTRADAY_WINS.items():
        p = price_at(ref + timedelta(minutes=mins))
        if p is not None:
            out[name] = p / base - 1.0
    return out


def forward_returns(ticker: str, t0: datetime) -> dict[str, float] | None:
    """All six horizons merged: intraday (Alpaca) + daily (yfinance). None if BOTH empty."""
    out = {**intraday_returns(ticker, t0), **daily_returns(ticker, t0)}
    return out or None


# ---------------------------------------------------------------- 3) validation
def _dir(x: float, band: float = 0.001) -> str:
    return "up" if x > band else "down" if x < -band else "flat"


# Option B — RELATIVE ALPHA labels. A prediction is a HIT only if the instrument
# beats the SPY benchmark in the predicted direction (abn = ret - spy_ret). This
# strips out market beta, so the null is a clean ~50% coin-flip, not the 74%
# "everything drifted up" majority. Band = 0 -> strict beat; raise for a cushion.
RELATIVE_BAND = 0.0


def relative_hit(pred: str, ret: float | None, spy_ret: float | None,
                 band: float = RELATIVE_BAND) -> bool | None:
    """True if the instrument beat SPY in the predicted direction. None if unscoreable."""
    if ret is None or spy_ret is None:
        return None
    abn = ret - spy_ret
    if pred == "up":
        return abn > band
    if pred == "down":
        return abn < -band
    return None


def validate(instruments: list[dict], t0: datetime, fwd=forward_returns
             ) -> tuple[list[dict], dict[str, list[int]], dict[str, float]]:
    """Attach returns + RELATIVE (beat-SPY) hit/miss per horizon.
    Returns (rows, hits{horizon:[hit,total]}, spy_returns).

    `fwd` = the return-fetcher (default forward_returns); the backtest injects a
    memoised version. `spy_returns` is returned so a run can be re-labelled later
    (e.g. a different band) without re-fetching any market data."""
    spy = fwd("SPY", t0) or {}
    rows: list[dict] = []
    hits: dict[str, list[int]] = {h: [0, 0] for h in HORIZONS}
    for ins in instruments:
        tk = str(ins.get("ticker", "")).upper()
        pred = str(ins.get("predicted_direction", "neutral")).lower()
        actual = fwd(tk, t0)
        row = {"ticker": tk, "name": ins.get("name", ""), "role": ins.get("role", ""),
               "predicted": pred, "returns": actual, "abn": {}, "hit": {}}
        if actual:
            for h in HORIZONS:
                if h in actual and h in spy:
                    row["abn"][h] = actual[h] - spy[h]     # vs S&P 500
                # SPY cannot beat itself -> the benchmark is never scored.
                if pred in ("up", "down") and tk != "SPY":
                    hit = relative_hit(pred, actual.get(h), spy.get(h))
                    if hit is not None:
                        row["hit"][h] = hit
                        hits[h][0] += int(hit)
                        hits[h][1] += 1
        rows.append(row)
    return rows, hits, spy


# ---------------------------------------------------------------- reporting
def _pct(v: float | None) -> str:
    return "   n/a " if v is None else f"{v * 100:+6.2f}%"


def report(text: str, t0: datetime, pred: dict, rows: list[dict], hits: dict[str, list[int]],
           model: str) -> None:
    print("=" * 100)
    print(f"NEBIUS LLM MACRO VALIDATION   model={model}   tweet_date={t0.date()}")
    print("=" * 100)
    print(f'TWEET: "{text[:160]}"')
    print(f"\nSCENARIO: {pred.get('scenario', '(none)')}")
    print(f"WHY:      {pred.get('rationale', '')}\n")
    if not rows:
        print("Model returned no market-relevant instruments — nothing to validate.")
        return

    hdr = f"{'instrument':28}{'role':16}{'pred':6}" + "".join(f"{h:>9}" for h in HORIZONS) + "   hits"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        label = f"{r['name']} ({r['ticker']})"[:27]
        cells = []
        nhit = ntot = 0
        for h in HORIZONS:
            ret = (r["returns"] or {}).get(h)
            cells.append(_pct(ret))
            if isinstance(r["hit"].get(h), bool):
                ntot += 1
                nhit += int(r["hit"][h])
        mark = "n/a" if ntot == 0 else f"{nhit}/{ntot}"
        print(f"{label:28}{str(r['role'])[:15]:16}{r['predicted']:6}"
              + "".join(f"{c:>9}" for c in cells) + f"   {mark}")

    print("\nHIT-RATE per horizon (did the market move as the LLM predicted?):")
    for h in HORIZONS:
        hit, tot = hits[h]
        rate = "n/a" if tot == 0 else f"{100 * hit / tot:5.1f}%  ({hit}/{tot})"
        print(f"  {h:5} {rate}")
    print("\nNote: raw direction shown; last column counts directional hits. Returns are also "
          "measured vs SPY (abnormal) internally. One tweet = descriptive, not causal proof.")


# ---------------------------------------------------------------- main
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEMO_TEXT, help="tweet text to analyze")
    ap.add_argument("--date", default=DEMO_DATE, help="tweet date YYYY-MM-DD (or ISO datetime)")
    ap.add_argument("--model", default=_env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL",
                                            default=DEFAULT_MODEL))
    ap.add_argument("--base-url", default=_env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL",
                                               default=DEFAULT_BASE))
    ap.add_argument("--html", default="reports/nebius_macro_card.html", help="visual card output")
    a = ap.parse_args()

    api_key = _env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    if not api_key:
        sys.exit("No NEBIUS_API_KEY. Put it in .env (gitignored) or export it. See the header.")

    t0 = datetime.fromisoformat(a.date)
    if t0.tzinfo is None:
        t0 = t0.replace(hour=t0.hour, tzinfo=timezone.utc)

    print(f"-> Nebius {a.model} classifying the tweet ...")
    pred = classify_tweet(a.text, base_url=a.base_url, api_key=api_key, model=a.model)
    instruments = pred.get("instruments", []) or []
    print(f"-> {len(instruments)} predicted instruments; fetching real market data via yfinance ...\n")
    rows, hits, _spy = validate(instruments, t0)
    report(a.text, t0, pred, rows, hits, a.model)

    # green/red visual card (option B) — the scannable, non-expert view.
    from reportgen.macro_card import render_page
    result = {
        "text": a.text, "date": t0.date().isoformat(), "model": a.model,
        "scenario": pred.get("scenario", ""), "rationale": pred.get("rationale", ""),
        "horizons": list(HORIZONS),
        "instruments": [{"ticker": r["ticker"], "name": r["name"], "role": r["role"],
                         "predicted": r["predicted"], "returns": r["returns"] or {},
                         "hit": r["hit"]} for r in rows],
        "hits": hits,
    }
    out = Path(a.html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(result), encoding="ascii")
    print(f"\n-> visual card -> {out}")


if __name__ == "__main__":
    main()
