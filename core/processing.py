"""Helpers for turning a tweet into a structured, reviewable record.

This module is intentionally thin and deterministic. It reuses the repository's
existing sector mapping and point-in-time market-state concepts so the output is
consistent with the canonical decision pipeline.
"""

from __future__ import annotations

import re
from typing import Any

from data.sources.interfaces import DailyBar, Tweet
from sector_mapping.rules import map_tweet, map_tweet_multi


def _classify_sentiment(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("great", "strong", "good", "love", "amazing")):
        return "positive"
    if any(term in lowered for term in ("bad", "terrible", "hate", "worst", "fail")):
        return "negative"
    return "neutral"


def _classify_intent(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("announce", "announcement", "launch", "plan")):
        return "announcement"
    if any(term in lowered for term in ("support", "love", "great", "thank")):
        return "praise"
    if any(term in lowered for term in ("attack", "bad", "terrible", "stupid")):
        return "attack"
    if any(term in lowered for term in ("policy", "tax", "trade", "tariff")):
        return "policy"
    return "neutral"


def _extract_entities(text: str) -> dict[str, list[str]]:
    companies = [item for item in ("tesla", "apple", "microsoft", "amazon", "google", "ford", "gm") if item in text.lower()]
    executives = [item for item in ("elon musk", "jerome powell", "tim cook") if item in text.lower()]
    sectors = []
    for mapping in map_tweet_multi(text):
        sectors.append(mapping.ticker)
    themes = []
    if re.search(r"\benergy\b|\boil\b|\bdrill\b", text, re.I):
        themes.append("energy")
    if re.search(r"\btrade\b|\btariff\b", text, re.I):
        themes.append("trade")
    if re.search(r"\btech\b|\btechnology\b|\bai\b", text, re.I):
        themes.append("technology")
    return {"companies": companies, "executives": executives, "sectors": sectors, "themes": themes}


def _performance_metrics(bars: list[DailyBar]) -> dict[str, float]:
    if len(bars) < 2:
        return {"month_before_pct": 0.0, "week_after_pct": 0.0, "month_after_pct": 0.0}
    start = bars[0].close
    end = bars[-1].close
    return {
        "month_before_pct": (bars[-1].close / bars[0].close - 1.0) if bars[0].close else 0.0,
        "week_after_pct": (end / start - 1.0) if start else 0.0,
        "month_after_pct": (end / start - 1.0) if start else 0.0,
    }


def build_tweet_record(
    tweet: Tweet,
    bars: list[DailyBar],
    future_bars: list[DailyBar] | None = None,
) -> dict[str, Any]:
    """Create a reviewable, structured record for a tweet.

    The output is deterministic and uses the rule-based sector mapper already in
    the repository. It also includes mock performance metrics for human review.
    """

    mapping = map_tweet(tweet.text)
    entities = _extract_entities(tweet.text)
    sector_label = mapping.ticker or ""
    if sector_label and sector_label not in entities["sectors"]:
        entities["sectors"].append(sector_label)

    target = {
        "ticker": sector_label or "",
        "sector": sector_label or "",
        "industry": "",
        "reason_linked": f"rule-based sector mapping for {tweet.text}",
        "expected_direction": "FLAT",
        "performance": _performance_metrics(bars),
        "verdict_week": "NO_CALL",
    }

    return {
        "tweet_id": tweet.tweet_id,
        "timestamp": tweet.timestamp_utc.isoformat(),
        "text": tweet.text,
        "sentiment": _classify_sentiment(tweet.text),
        "intent": _classify_intent(tweet.text),
        "entities": entities,
        "confidence_score": round(mapping.confidence, 3) if mapping.ticker else 0.0,
        "targets": [target],
    }
