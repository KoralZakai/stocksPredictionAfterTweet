"""Decision-plane router: classified tweet -> LONG/SHORT/ABSTAIN + instruments.

PURE and TWEET-ONLY. Input is the dict returned by `alpha.classify.classify_tweet`
(itself built from tweet text alone). This function touches NO market data — it is
the decision plane of the firewall (serving/app.py). Its output feeds the market
plane, never the reverse.

Decision rule mirrors the validated `signed_eod` aggregation
(scripts/nebius_macro_backtest.py): a tweet's stance is the MAJORITY predicted
direction of its instruments (ties -> LONG, matching `ups >= downs`). Macro track
=> every instrument is scored against SPY.
"""

from __future__ import annotations

from typing import Any

from config.membership import TICKER_NAME
from config.universe import SECTOR_STOCKS

from alpha.schema import Instrument, RoutedDecision

# The instruments the macro classifier is instructed to use (alpha/classify.py
# _INSTRUCT), plus every single-name/ETF we hold benchmarks for. A ticker outside
# this set cannot be resolved, so it triggers ABSTAIN (mission rule).
_PROMPT_UNIVERSE = frozenset({
    "SPY", "QQQ", "DIA", "XLI", "ITA", "XLE", "XLK", "XLF", "XLV", "XLY", "XLP",
    "XLB", "SMH", "SOXX", "PAVE", "VIXY", "USO", "WEAT", "CORN", "DBC",
    "CAT", "LMT", "AAPL",
})
WHITELIST: frozenset[str] = (
    _PROMPT_UNIVERSE
    | frozenset(TICKER_NAME)
    | frozenset(SECTOR_STOCKS)
    | frozenset({s for names in SECTOR_STOCKS.values() for s in names})
)

MACRO_BENCHMARK = "SPY"


def route_decision(classified: dict[str, Any]) -> RoutedDecision:
    """Collapse the LLM's instrument basket into a single LONG/SHORT/ABSTAIN call.

    ABSTAIN when the post is not market-relevant (no instruments), or no instrument
    resolves to the whitelist, or no instrument carries a directional (up/down) call.
    """
    scenario = str(classified.get("scenario", "") or "")
    reasoning = str(classified.get("rationale", "") or classified.get("macro_link", "") or "")
    raw = classified.get("instruments") or []

    resolved: list[Instrument] = []
    for ins in raw:
        tk = str(ins.get("ticker", "")).upper().strip()
        direction = str(ins.get("predicted_direction", "")).lower().strip()
        if direction not in ("up", "down"):
            continue                      # neutral / missing -> not a directional call
        if tk == MACRO_BENCHMARK:
            continue                      # SPY is the benchmark; it cannot beat itself
        if tk not in WHITELIST:
            continue                      # outside whitelist -> unresolvable
        resolved.append(Instrument(ticker=tk, direction=direction, benchmark=MACRO_BENCHMARK))

    if not raw:
        return RoutedDecision("ABSTAIN", [], scenario, reasoning,
                              abstain_reason="not market-relevant (no instruments)")
    if not resolved:
        return RoutedDecision("ABSTAIN", [], scenario, reasoning,
                              abstain_reason="no whitelisted directional instrument resolved")

    ups = sum(1 for i in resolved if i.direction == "up")
    downs = sum(1 for i in resolved if i.direction == "down")
    decision = "LONG" if ups >= downs else "SHORT"   # ties -> LONG (mirrors signed_eod)
    return RoutedDecision(decision, resolved, scenario, reasoning)
