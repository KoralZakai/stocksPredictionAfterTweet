"""Live macro-hypothesis pipeline: Nebius LLM -> predicted instruments -> validate vs real market.

THIN CLI WRAPPER (Stage 1). The engine now lives in signal/ so the batch Job and
the serving Endpoint share ONE code path (no train/serve skew, CLAUDE.md 3.2).
This file only marshals argparse I/O, prints a table, and renders the HTML card;
all classification/return/scoring logic is imported from alpha.*.

Point-in-time: entry = the first session OPEN strictly after the tweet timestamp
(no same-day leak). This is an ASSOCIATION check on one tweet — descriptive, zero
evidential weight; it does not prove causation.

Setup (key stays uncommitted — .env is gitignored):
  1. Get a key at https://studio.nebius.com  (Token Factory / AI Studio).
  2. Put it in .env  (or export it):   NEBIUS_API_KEY=<your-key>
     (This repo's .env also accepts EXPO_PUBLIC_NEBIUS_API_KEY / _BASE_URL / _MODEL.)

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/nebius_macro_validate.py
  ... --text "HOPEFULLY RUSSIA AND UKRAINE WILL MAKE A DEAL THIS WEEK..." --date 2025-04-20
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Re-export the shared engine so existing importers (scripts/nebius_macro_backtest,
# scripts/nebius_micro_backtest) keep working unchanged — proven byte-identical.
from alpha.benchmark import (  # noqa: F401
    DAILY_SESS, HORIZONS, INTRADAY_WINS, RELATIVE_BAND, US_OPEN_UTC_HOUR,
    _dir, _session_anchor, _sessions, daily_returns, forward_returns,
    intraday_returns, relative_hit, validate,
)
from alpha.classify import (  # noqa: F401
    DEFAULT_BASE, DEFAULT_MODEL, _INSTRUCT, _SYSTEM, _parse_json,
    classify_tweet, prompt_template_hash,
)
from alpha.env import _env, _load_dotenv, env, load_dotenv  # noqa: F401

# A default worked example — the Russia/Ukraine "peace deal" tweet (macro, not a chip stock).
DEMO_TEXT = ("HOPEFULLY RUSSIA AND UKRAINE WILL MAKE A DEAL THIS WEEK. BOTH WILL THEN "
             "START TO DO BIG BUSINESS WITH THE UNITED STATES OF AMERICA, WHICH IS "
             "THRIVING, AND MAKE A FORTUNE!")
DEMO_DATE = "2025-04-20"


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
