"""Rule-based tweet -> sector-ETF mapping (§6).

Deterministic keyword scoring. No ML mapper in V1 — an unvalidated classifier
in the causal chain would make "no signal" indistinguishable from "bad
mapping". Returns the argmax sector (one row per tweet, §6 primary) plus a
confidence, or NONE when nothing matches.
"""

from __future__ import annotations

from dataclasses import dataclass

# ETF -> lowercase keyword/phrase triggers. Kept small and legible on purpose.
SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "XLE": ("oil", "energy", "drill", "opec", "gas", "petroleum"),
    "SMH": ("chip", "chips", "semiconductor", "semiconductors"),
    "XLK": ("tech", "technology", "software"),
    "XLF": ("bank", "banks", "fed", "interest rate", "rates", "financial"),
    "ITA": ("military", "defense", "army", "war", "weapons", "troops"),
    "XLI": ("tariff", "tariffs", "manufacturing", "trade war", "industrial", "factory"),
    "XLB": ("steel", "aluminum", "materials", "mining"),
    "XLV": ("drug", "drugs", "pharma", "health", "healthcare", "medicare"),
    "XLY": ("car", "cars", "auto", "autos", "retail", "consumer"),
    "XLP": ("grocery", "groceries", "food", "staples"),
}


@dataclass(frozen=True)
class Mapping:
    ticker: str | None  # None == maps to NONE, tweet excluded from eval (§6)
    score: int
    confidence: float  # share of total keyword hits landing on the winner


def map_tweet(text: str) -> Mapping:
    tl = text.lower()
    scores = {tk: sum(tl.count(w) for w in ws) for tk, ws in SECTOR_KEYWORDS.items()}
    total = sum(scores.values())
    if total == 0:
        return Mapping(None, 0, 0.0)
    best = max(scores, key=lambda k: scores[k])
    return Mapping(best, scores[best], scores[best] / total)
