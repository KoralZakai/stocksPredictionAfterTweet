"""The structured signal an offline LLM extracts from tweet text (product fork).

Deliberately NARROW and DETERMINISTIC-TO-CONSUME: fixed enums + bounded scalars,
so the tree sees a stable, low-dimension feature vector (§8 — no raw embeddings).
No ticker/sector field: routing stays in the deterministic sector_mapping chain
(§6). Bump SCHEMA_VERSION on ANY field change so the cache re-extracts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

# Event taxonomy — a company/sector-agnostic description of WHAT the tweet does.
EVENT_TYPES: tuple[str, ...] = (
    "military_threat",   # threatens/announces force, strike, war
    "sanction",          # sanctions / export bans / blockade
    "tariff_trade",      # tariffs, trade war, trade deal
    "monetary_policy",   # Fed / rates / dollar
    "energy_policy",     # drilling, OPEC, oil supply
    "drug_pricing",      # pharma / healthcare pricing
    "regulation",        # antitrust, dodd-frank, sector rules
    "praise",            # positive stance toward a named entity
    "attack",            # negative stance toward a named entity (rhetorical)
    "macro_claim",       # economy/markets/jobs boast or warning
    "other",             # market-relevant but none of the above
    "none",              # not market-relevant
)
EventType = Literal[
    "military_threat", "sanction", "tariff_trade", "monetary_policy",
    "energy_policy", "drug_pricing", "regulation", "praise", "attack",
    "macro_claim", "other", "none",
]
Level = Literal["low", "medium", "high"]
Direction = Literal["bullish", "bearish", "neutral"]


class TweetSignal(BaseModel):
    """LLM's structured read of a single tweet. Text-derived only (pre-t0)."""

    event_type: EventType = Field(description="What the tweet primarily does.")
    direction_of_intent: Direction = Field(
        description="Expected directional pressure on the AFFECTED asset/sector "
        "implied by the tweet's content, ignoring current prices: bullish, "
        "bearish, or neutral. E.g. 'attack on Iran' -> oil bullish."
    )
    urgency: Level = Field(description="Immediacy of the implied action/threat.")
    magnitude: Level = Field(description="Scale of the implied market impact.")
    certainty: Level = Field(description="How committed/definite the statement is.")
    names_country: bool = Field(description="Names a specific country.")
    names_company: bool = Field(description="Names a specific company or exec.")


# Stable, sorted feature-column order so train and serve cannot disagree (§8).
_LEVELS = {"low": 0.0, "medium": 0.5, "high": 1.0}
_DIRS = {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}


def signal_to_features(signal: TweetSignal) -> dict[str, float]:
    """TweetSignal -> flat numeric columns prefixed `llm_` (merge into build_features)."""
    f: dict[str, float] = {f"llm_event_{e}": float(signal.event_type == e) for e in EVENT_TYPES}
    f["llm_intent"] = _DIRS[signal.direction_of_intent]
    f["llm_urgency"] = _LEVELS[signal.urgency]
    f["llm_magnitude"] = _LEVELS[signal.magnitude]
    f["llm_certainty"] = _LEVELS[signal.certainty]
    f["llm_names_country"] = float(signal.names_country)
    f["llm_names_company"] = float(signal.names_company)
    return f


def zero_features() -> dict[str, float]:
    """All-zero / neutral LLM columns — the honest fallback when no signal exists
    (no API key, cache miss on a live tweet). Same keys as signal_to_features so
    the feature vector layout never changes."""
    return signal_to_features(
        TweetSignal(
            event_type="none", direction_of_intent="neutral", urgency="low",
            magnitude="low", certainty="low", names_country=False, names_company=False,
        )
    )
