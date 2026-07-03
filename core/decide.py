"""The single pure decision function (§3.2, §5 core/decide.py).

decide(tweet, market_state) is the ONLY feature path in the system. The offline
batch runner, the future /predict endpoint, and the replay harness all route
through here — the no-skew test (tests/test_decide.py) asserts batch and
single-event calls produce byte-identical features.

Phase 0 has no trained model, so the honest output is ABSTAIN (conformal
abstention arrives in step 8). The prediction slot is a seam; the feature
computation is the real contract enforced now.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.features import build_features
from core.market_state import MarketState
from data.sources.interfaces import Tweet


@dataclass(frozen=True)
class Decision:
    ticker: str
    features: dict[str, float]
    direction: str  # UP / DOWN / NEUTRAL / ABSTAIN
    confidence: float
    abstain: bool


def decide(tweet: Tweet, state: MarketState) -> Decision:
    features = build_features(tweet, state)
    # No model yet -> abstain (step 8 attaches the GBT + conformal calibrator here).
    return Decision(state.ticker, features, "ABSTAIN", 0.0, True)


def decide_batch(
    tweets: Sequence[Tweet], states: Sequence[MarketState]
) -> list[Decision]:
    """Offline batch runner — a thin map over the SAME decide(). No second path."""
    return [decide(t, s) for t, s in zip(tweets, states, strict=True)]
