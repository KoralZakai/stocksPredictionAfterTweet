"""Policy-theme taxonomy -> asset links + the market-relevance filter (dataset v3).

Third relevance layer on top of sector keywords (rules.py) and direct entity
mentions (entities.py): a post about TARIFFS is market-relevant even when it
names no company and no sector keyword — the policy theme itself maps to assets
with a pre-registered link strength. Everything here is deterministic regex on
the post text alone (knowable pre-t0); no ML, no price data (§3.1, §6).

Layers combined in `combined_relevance()`:
    direct entity mention   -> 1.0                      (entities.py)
    sector keyword share    -> conf / 0.5*conf stocks   (entities.py)
    policy theme link       -> strength / 0.5*strength  (this module)
    relevance(asset) = max over layers.

Macro themes (stock_market / taxes / jobs_economy / agriculture) mark a post
market-RELEVANT for the filter but link no asset in our universe — abnormal
return is measured vs SPY, so a pure-macro post has no asset-specific target
(and agriculture names like DE/ADM are outside the 45-asset universe).
"""

from __future__ import annotations

import re

from config.universe import SECTOR_STOCKS
from sector_mapping.entities import asset_relevance
from sector_mapping.rules import sector_scores

# Theme -> weighted, word-boundaried triggers. Weight 3 = unambiguous policy
# phrase, 2 = solid domain token, 1 = generic (needs company/repeats to pass).
THEME_TRIGGERS: dict[str, tuple[tuple[str, int], ...]] = {
    "tariffs_trade": (
        (r"\btariffs?\b", 3), (r"\btrade wars?\b", 3), (r"\btrade deals?\b", 3),
        (r"\btrade deficits?\b", 3), (r"\bnafta\b", 3), (r"\busmca\b", 3),
        (r"\btrans.pacific\b", 3), (r"\bwto\b", 3), (r"\bdumping\b", 2),
        (r"\bimport (?:tax|dut)\w*\b", 3), (r"\btrade barriers?\b", 3),
        (r"\bcurrency manipulat\w+\b", 3), (r"\bintellectual property theft\b", 3),
    ),
    "china": (
        (r"\bchina\b", 2), (r"\bchinese\b", 2), (r"\bxi jinping\b", 3),
        (r"\bbeijing\b", 2), (r"\bpresident xi\b", 3), (r"\bhong kong\b", 2),
    ),
    "oil_energy_geopolitics": (
        (r"\bopec\b", 3), (r"\bsaudi arabia\b", 2), (r"\boil prices?\b", 3),
        (r"\bstrategic petroleum reserve\b", 3), (r"\benergy independen\w+\b", 3),
        (r"\bparis (?:climate )?(?:accord|agreement)\b", 3), (r"\biran sanctions?\b", 3),
        (r"\bnord stream\b", 3),
    ),
    "defense_geopolitics": (
        (r"\bnorth korea\b", 2), (r"\bnato\b", 2), (r"\biran\b", 2),
        (r"\bsyria\b", 2), (r"\bafghanistan\b", 2), (r"\bkim jong\b", 3),
        (r"\bdefense spending\b", 3), (r"\bmilitary budget\b", 3),
        (r"\bspace force\b", 3), (r"\bnuclear (?:deal|weapons?|arsenal)\b", 2),
    ),
    "pharma_healthcare": (
        (r"\bdrug (?:prices?|pricing|costs?)\b", 3), (r"\bvaccines?\b", 2),
        (r"\bopioids?\b", 2), (r"\binsulin\b", 3), (r"\bobamacare\b", 2),
        (r"\baffordable care act\b", 2), (r"\bprescription\w*\b", 2),
        (r"\bright to try\b", 3), (r"\bfavored nations? clause\b", 3),
    ),
    "fed_rates": (
        (r"\bfederal reserve\b", 3), (r"\bthe fed\b", 3), (r"\bjerome powell\b", 3),
        (r"\binterest rates?\b", 3), (r"\bquantitative (?:easing|tightening)\b", 3),
        (r"\brate (?:cuts?|hikes?)\b", 3), (r"\bstrong dollar\b", 3),
        (r"\bweak dollar\b", 3), (r"\bdevalu\w+\b", 2), (r"\binflation\b", 2),
    ),
    "semis_tech_supplychain": (
        (r"\bhuawei\b", 3), (r"\b5g\b", 2), (r"\btiktok\b", 3), (r"\btaiwan\b", 2),
        (r"\bsemiconductors?\b", 3), (r"\bchip (?:makers?|industry|plants?)\b", 3),
        (r"\bexport (?:controls?|bans?)\b", 3),
    ),
    "evs_autos": (
        (r"\belectric vehicles?\b", 3), (r"\bev mandates?\b", 3),
        (r"\bemissions? standards?\b", 3), (r"\bfuel (?:economy|efficiency)\b", 3),
        (r"\bauto ?(?:makers?|workers?|industry|plants?)\b", 2), (r"\bcar compan\w+\b", 2),
    ),
    "tech_antitrust": (
        (r"\bsection 230\b", 3), (r"\bantitrust\b", 3), (r"\bbig tech\b", 3),
        (r"\bbreak\w* up (?:big tech|tech)\b", 3), (r"\btech (?:giants?|monopol\w+)\b", 3),
    ),
    "banks_regulation": (
        (r"\bdodd.frank\b", 3), (r"\bglass.steagall\b", 3),
        (r"\bcommunity banks?\b", 3), (r"\bbank(?:ing)? regulat\w+\b", 3),
    ),
    "manufacturing_reshoring": (
        (r"\bmade in (?:the )?(?:usa|america)\b", 3), (r"\breshor\w+\b", 3),
        (r"\bmanufactur\w+\b", 2), (r"\bfactor(?:y|ies)\b", 2),
        (r"\bsteel (?:workers?|industry|mills?)\b", 3), (r"\bbring\w* (?:back )?jobs\b", 2),
    ),
    "aerospace_aviation": (
        (r"\bair force one\b", 3), (r"\b747\b", 2), (r"\bfaa\b", 2),
        (r"\b737 ?max\b", 3),
    ),
    # ---- macro themes: market-relevant for the FILTER, but no asset link ----
    "stock_market_macro": (
        (r"\bstock markets?\b", 3), (r"\bdow(?: jones)?\b", 3), (r"\bnasdaq\b", 3),
        (r"\bs&p ?500?\b", 3), (r"\b401\(?k\)?s?\b", 3), (r"\ball.time high\b", 2),
        (r"\bwall street\b", 2), (r"\bbull market\b", 3),
    ),
    "taxes": (
        (r"\btax (?:cuts?|bill|reform|plan)\b", 3), (r"\bcorporate tax(?:es)?\b", 3),
        (r"\btax(?:es)? (?:cut|reduc\w+)\b", 3), (r"\bcapital gains\b", 3),
    ),
    "jobs_economy": (
        (r"\bjobs? (?:report|numbers?|growth)\b", 3), (r"\bunemployment\b", 2),
        (r"\bgdp\b", 3), (r"\beconomy\b", 1), (r"\beconomic growth\b", 2),
        (r"\bjobs\b", 1), (r"\brecession\b", 2),
    ),
    "agriculture": (
        (r"\bfarmers?\b", 2), (r"\bsoybeans?\b", 3), (r"\bethanol\b", 3),
        (r"\bcorn\b", 2), (r"\bcrops?\b", 2), (r"\bagricultur\w+\b", 2),
    ),
}

# Theme -> (asset, link strength in (0,1]). Assets may be ETFs (member stocks
# inherit 0.5x, same rule as the sector layer) or single names (as-is).
# Macro themes deliberately have NO entry here.
THEME_ASSETS: dict[str, tuple[tuple[str, float], ...]] = {
    "tariffs_trade": (("XLI", 0.6), ("XLB", 0.5), ("SMH", 0.4), ("XLY", 0.3)),
    "china": (("SMH", 0.5), ("XLI", 0.4), ("XLB", 0.3)),
    "oil_energy_geopolitics": (("XLE", 0.6),),
    "defense_geopolitics": (("ITA", 0.5),),
    "pharma_healthcare": (("XLV", 0.6),),
    "fed_rates": (("XLF", 0.6),),
    "semis_tech_supplychain": (("SMH", 0.6), ("XLK", 0.3)),
    "evs_autos": (("TSLA", 0.6), ("XLY", 0.3)),
    "tech_antitrust": (("XLK", 0.5),),
    "banks_regulation": (("XLF", 0.5),),
    "manufacturing_reshoring": (("XLI", 0.5), ("XLB", 0.3)),
    "aerospace_aviation": (("BA", 0.5),),
}

MIN_THEME_SCORE = 2   # same bar as rules.MIN_SCORE: a lone weight-1 token never fires
MIN_RELEVANCE = 0.1

_COMPILED = {
    th: tuple((re.compile(p, re.I), w) for p, w in trigs)
    for th, trigs in THEME_TRIGGERS.items()
}


def theme_scores(text: str) -> dict[str, int]:
    """Weighted trigger score per theme (0 entries omitted)."""
    out: dict[str, int] = {}
    for th, pats in _COMPILED.items():
        s = sum(len(rx.findall(text)) * w for rx, w in pats)
        if s:
            out[th] = s
    return out


def active_themes(text: str) -> list[str]:
    """Themes scoring >= MIN_THEME_SCORE, strongest first (deterministic tie-break)."""
    sc = theme_scores(text)
    return sorted((t for t, s in sc.items() if s >= MIN_THEME_SCORE),
                  key=lambda t: (-sc[t], t))


def theme_asset_relevance(text: str) -> dict[str, float]:
    """ticker -> relevance from active policy themes only (ETF fan-out at 0.5x)."""
    rel: dict[str, float] = {}
    for th in active_themes(text):
        for asset, strength in THEME_ASSETS.get(th, ()):
            rel[asset] = max(rel.get(asset, 0.0), strength)
            for stock in SECTOR_STOCKS.get(asset, ()):  # ETF -> member stocks
                rel[stock] = max(rel.get(stock, 0.0), 0.5 * strength)
    return {tk: round(v, 4) for tk, v in rel.items() if v >= MIN_RELEVANCE}


def market_relevance(text: str) -> tuple[bool, list[str]]:
    """(is_market_relevant, reasons). Conservative KEEP: any of the three
    pre-t0 text layers firing keeps the post; reasons are auditable."""
    reasons: list[str] = []
    if sector_scores(text):
        reasons.append("sector_keywords")
    ent = asset_relevance(text)
    if any(v >= 0.99 for v in ent.values()):
        reasons.append("entity_mention")
    reasons += [f"theme:{t}" for t in active_themes(text)]
    return bool(reasons), reasons


def combined_relevance(text: str) -> dict[str, float]:
    """max over the entity/sector layer and the theme layer, per ticker."""
    rel = dict(asset_relevance(text))
    for tk, v in theme_asset_relevance(text).items():
        rel[tk] = max(rel.get(tk, 0.0), v)
    return {tk: round(v, 4) for tk, v in rel.items() if v >= MIN_RELEVANCE}
