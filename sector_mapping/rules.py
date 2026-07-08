"""Rule-based tweet -> sector-ETF mapping (§6) — stricter hybrid v2.

Deterministic, no ML in the causal chain (§6). v1 was single-word SUBSTRING
matching, which mis-fired badly: "war" matched inside "trade war"/"toward",
"car" inside "caravan", "rate" inside "grateful". v2 fixes that with:

  * phrase-level, WORD-BOUNDARIED regex triggers (no substring bleed);
  * per-trigger weights (strong domain phrase > generic token);
  * negative VETOES (context filters that cancel a sector);
  * a minimum score to map at all (a lone generic token no longer maps).

Confidence = a sector's share of total matched weight (context-aware, not binary).
`matched_triggers()` exposes exactly which phrases fired, for the audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ETF -> ((regex, weight), ...). Weight: 3 = unambiguous domain phrase,
# 2 = solid domain token, 1 = generic/soft. All patterns are \b-anchored.
SECTOR_TRIGGERS: dict[str, tuple[tuple[str, int], ...]] = {
    "XLE": ((r"\boil\b", 2), (r"\bopec\b", 3), (r"\bdrill(?:ing)?\b", 2),
            (r"\bpipelines?\b", 2), (r"\bcrude\b", 2), (r"\bgasoline\b", 2),
            (r"\bpetroleum\b", 2), (r"\bnatural gas\b", 3), (r"\bfracking\b", 3),
            (r"\bkeystone\b", 2), (r"\bshale\b", 2), (r"\benergy\b", 1)),
    "SMH": ((r"\bsemiconductors?\b", 3), (r"\bmicrochips?\b", 3), (r"\bchips?\b", 2),
            (r"\bchip (?:makers?|industry)\b", 3)),
    "XLK": ((r"\btechnology\b", 2), (r"\bbig tech\b", 3), (r"\bartificial intelligence\b", 3),
            (r"\bsilicon valley\b", 3), (r"\bsoftware\b", 2),
            (r"\btech (?:companies|industry|giants)\b", 3)),
    "XLF": ((r"\bbanks?\b", 2), (r"\bfederal reserve\b", 3), (r"\bthe fed\b", 3),
            (r"\bjerome powell\b", 3), (r"\binterest rates?\b", 3), (r"\bwall street\b", 2),
            (r"\bdodd.frank\b", 2), (r"\bmortgages?\b", 2)),
    # v3: dropped "national guard" — domestic deployments (protests/disasters)
    # are not defense-industry demand news and mis-linked BA/LMT/RTX badly.
    "ITA": ((r"\bmilitary\b", 2), (r"\bdefense\b", 2), (r"\bdefence\b", 2),
            (r"\barmed forces\b", 3), (r"\btroops\b", 2), (r"\bsoldiers?\b", 2),
            (r"\bmissiles?\b", 2), (r"\bwarships?\b", 3),
            (r"\bpentagon\b", 3), (r"\bweapons?\b", 2), (r"\bfighter jets?\b", 3),
            (r"\baircraft carriers?\b", 3), (r"\bnavy\b", 2), (r"\bair force\b", 3),
            (r"\barmy\b", 2)),
    "XLI": ((r"\btariffs?\b", 3), (r"\btrade war\b", 3), (r"\bmanufacturing\b", 2),
            (r"\bfactor(?:y|ies)\b", 2), (r"\binfrastructure\b", 2), (r"\bindustrial\b", 1),
            (r"\breshoring\b", 3), (r"\bsupply chains?\b", 1), (r"\btrade deal\b", 2),
            (r"\bboeing\b", 2)),
    "XLB": ((r"\bsteel\b", 2), (r"\baluminum\b", 2), (r"\bcopper\b", 2), (r"\bmining\b", 2)),
    "XLV": ((r"\bdrug prices?\b", 3), (r"\bprescriptions?\b", 2), (r"\bpharma(?:ceuticals?)?\b", 2),
            (r"\bhealthcare\b", 2), (r"\bhealth care\b", 2), (r"\bmedicare\b", 2),
            (r"\bmedicaid\b", 2), (r"\bbig pharma\b", 3), (r"\bfda\b", 1)),
    "XLY": ((r"\bauto(?:mobile|makers?|motive)?s?\b", 2), (r"\bcars?\b", 2),
            (r"\bretail(?:ers?)?\b", 2), (r"\bconsumers?\b", 1), (r"\be.commerce\b", 2),
            (r"\bshopping\b", 1)),
    "XLP": ((r"\bgrocer(?:y|ies)\b", 2), (r"\bfood prices?\b", 3),
            (r"\bconsumer staples\b", 3), (r"\bsupermarkets?\b", 2)),
}

# Context vetoes: if a veto matches, the sector is cancelled for that tweet.
# Kills residual ambiguity even when a trigger technically fires.
SECTOR_VETOES: dict[str, tuple[str, ...]] = {
    # "war" isn't an ITA trigger anymore, but these phrases can still pull "army"/
    # "navy"-free false context; and a chip/car in a clearly off-topic frame.
    "ITA": (r"\btrade war\b", r"\bprice war\b", r"\bculture war\b", r"\bwar of words\b"),
    "XLY": (r"\bcaravans?\b",),   # immigration, not autos
    "SMH": (r"\bchip shot\b", r"\bblue chip\b"),
}

MIN_SCORE = 2  # a lone generic (weight-1) token no longer maps; kills noise


def _label(pattern: str) -> str:  # human-readable trigger for the audit log
    return re.sub(r"\\b|\(\?:|[\\()?]", "", pattern).replace("|", "/").strip()


_COMPILED: dict[str, tuple[tuple[re.Pattern[str], str, int], ...]] = {
    etf: tuple((re.compile(p, re.I), _label(p), w) for p, w in trigs)
    for etf, trigs in SECTOR_TRIGGERS.items()
}
_VETO: dict[str, tuple[re.Pattern[str], ...]] = {
    etf: tuple(re.compile(p, re.I) for p in pats) for etf, pats in SECTOR_VETOES.items()
}


@dataclass(frozen=True)
class Mapping:
    ticker: str | None  # None == maps to NONE, tweet excluded from eval (§6)
    score: int
    confidence: float  # share of total matched weight landing on this sector


def _score(text: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    scores: dict[str, int] = {}
    trig: dict[str, list[str]] = {}
    for etf, pats in _COMPILED.items():
        if any(v.search(text) for v in _VETO.get(etf, ())):
            continue
        s, labels = 0, []
        for rx, lab, w in pats:
            n = len(rx.findall(text))
            if n:
                s += n * w
                labels.append(lab)
        if s:
            scores[etf] = s
            trig[etf] = labels
    return scores, trig


def matched_triggers(text: str) -> dict[str, list[str]]:
    """Which trigger phrases fired per sector (for the mapping audit log)."""
    return _score(text)[1]


def map_tweet(text: str) -> Mapping:
    scores, _ = _score(text)
    total = sum(scores.values())
    if not scores:
        return Mapping(None, 0, 0.0)
    best = max(scores, key=lambda k: scores[k])
    if scores[best] < MIN_SCORE:
        return Mapping(None, 0, 0.0)
    return Mapping(best, scores[best], scores[best] / total)


def map_tweet_multi(text: str) -> list[Mapping]:
    """All sectors scoring >= MIN_SCORE, ranked by relevance (candidate set, §6)."""
    scores, _ = _score(text)
    total = sum(scores.values())
    if not total:
        return []
    hits = [(tk, s) for tk, s in scores.items() if s >= MIN_SCORE]
    hits.sort(key=lambda kv: kv[1], reverse=True)
    return [Mapping(tk, s, s / total) for tk, s in hits]


def sector_scores(text: str) -> dict[str, int]:
    """Per-sector weighted score (topic-flag source for core/features.py)."""
    return _score(text)[0]
