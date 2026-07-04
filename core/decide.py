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
from typing import Protocol

from core.features import build_features
from core.market_state import MarketState
from data.sources.interfaces import Tweet


class DirectionPredictor(Protocol):
    """Structural type a model bundle satisfies (models.Predictor). Kept as a
    Protocol so core/ never imports models/ (would be circular via dataset)."""

    def predict_direction(self, features: dict[str, float]) -> tuple[str, float, bool]:
        """-> (direction, confidence, abstain)."""
        ...


@dataclass(frozen=True)
class Decision:
    ticker: str
    features: dict[str, float]
    direction: str  # UP / DOWN / NEUTRAL / ABSTAIN
    confidence: float
    abstain: bool


def decide(
    tweet: Tweet, state: MarketState, predictor: DirectionPredictor | None = None
) -> Decision:
    features = build_features(tweet, state)
    if predictor is None:
        # No model -> abstain (the honest Phase-0 / cold-start output).
        return Decision(state.ticker, features, "ABSTAIN", 0.0, True)
    direction, confidence, abstain = predictor.predict_direction(features)
    return Decision(state.ticker, features, direction, confidence, abstain)


def decide_batch(
    tweets: Sequence[Tweet],
    states: Sequence[MarketState],
    predictor: DirectionPredictor | None = None,
) -> list[Decision]:
    """Offline batch runner — a thin map over the SAME decide(). No second path."""
    return [decide(t, s, predictor) for t, s in zip(tweets, states, strict=True)]
