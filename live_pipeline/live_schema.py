"""Structured schema for the EXPERIMENTAL live analysis route. NOT the shipped path.

Isolation: nothing here is imported by serving/app.py or the frozen tests. This is
a rich, qualitative LLM analysis surface — deliberately separate from the validated
beat-SPY decision engine.

INTEGRITY NOTES baked into the fields:
- There is NO calibrated per-tweet probability. `llm_conviction` is the model's
  UNCALIBRATED self-report, renamed from "confidence" on purpose: per-tweet
  confidence was tested and failed to generalize (Val AUC 0.593 -> Test 0.431).
  Never read it as P(correct).
- `long_term` predictions are flagged speculative: our evidence shows the edge is
  front-loaded and the multi-week "drift" was market beta, not signal.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

_CONVICTION_NOTE = ("Model's UNCALIBRATED self-reported conviction (0-1). NOT a "
                    "validated probability and NOT P(correct) — per-tweet confidence "
                    "was tested and failed to generalize (Val 0.593 -> Test 0.431).")


class AssetPrediction(BaseModel):
    asset_name: str = Field(description="Ticker/name of the stock, sector ETF (XLF, XLE), or index (SPY, QQQ)")
    asset_type: str = Field(description="'stock', 'sector', or 'index'")
    horizon: str = Field(description="'short_term' (1h-24h; the evidenced horizon) or 'long_term' (weeks-months)")
    direction: str = Field(description="'UP', 'DOWN', or 'NEUTRAL'")
    llm_conviction: float = Field(ge=0.0, le=1.0, description=_CONVICTION_NOTE)
    catalyst_reasoning: str = Field(description="Brief logical link from the statement to this asset")
    speculative: bool = Field(default=False,
                              description="True for long_term calls — beta-dominated, unvalidated by our study")


class TrumpStatementAnalysis(BaseModel):
    literal_translation: str = Field(description="What was literally said / claimed")
    veiled_meaning: str = Field(description="The implied geopolitical or economic agenda behind it")
    macro_economic_impact: str = Field(description="Broader impact: inflation, rates, tariffs, trade relations")
    predictions: list[AssetPrediction] = Field(default_factory=list,
                                               description="Per-asset impact predictions")


class LiveResponse(BaseModel):
    """The /live-predict envelope — carries the analysis plus loud honesty flags."""
    analysis: TrumpStatementAnalysis
    abstained: bool = Field(default=False, description="True when the post is not market-relevant")
    validated: bool = Field(default=False,
                            description="This experimental route's outputs are NOT validated on the sacred test")
    disclaimer: str = Field(
        default=("Experimental research output — NOT investment advice, NOT a validated model. "
                 "llm_conviction is uncalibrated. long_term calls are speculative (beta-dominated). "
                 "The validated pipeline is the shipped /predict endpoint (EOD beat-SPY)."))
