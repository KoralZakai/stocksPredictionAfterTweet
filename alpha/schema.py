"""Typed request/response contract for the /predict endpoint.

Split by PLANE to make the leakage firewall a TYPE-LEVEL guarantee, not just a
convention:
  - `RoutedDecision` (decision plane) is built from tweet text ALONE. It has no
    field that could hold a price, quote, or session state.
  - `MarketContext` (market plane) is a SEPARATE type, attached by the endpoint
    AFTER the decision. It can never be an input to `RoutedDecision`.

No per-tweet probability field exists anywhere here, by design: the meta-model
that would produce one was rejected on the sacred test (Val AUC 0.593 -> Test
0.431). We ship the raw call + a cohort base rate, never a per-tweet score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- request
@dataclass(frozen=True)
class PredictRequest:
    tweet_text: str
    t0_utc: str          # iso8601; the endpoint resolves the entry anchor from it
    author: str = ""


# ---------------------------------------------------------------- decision plane
@dataclass(frozen=True)
class Instrument:
    ticker: str
    direction: str       # "up" | "down"
    benchmark: str       # macro track: always "SPY"


@dataclass(frozen=True)
class CohortBaseRate:
    """Historical hit-rate of ALL calls of this type on the held-out chronological
    test set. NOT a probability for THIS tweet (see module docstring)."""
    value: float
    ci95: list[float]    # [low, high] Wilson interval
    n: int
    horizon: str
    note: str


@dataclass(frozen=True)
class RoutedDecision:
    """DECISION PLANE — derived from tweet text only. No market-derived field."""
    decision: str                    # "LONG" | "SHORT" | "ABSTAIN"
    instruments: list[Instrument]
    scenario: str
    reasoning: str
    abstain_reason: str = ""         # populated only when decision == "ABSTAIN"


# ---------------------------------------------------------------- market plane
@dataclass(frozen=True)
class Quote:
    ticker: str
    last: float
    benchmark_ticker: str
    benchmark_last: float


@dataclass(frozen=True)
class RealizedAlpha:
    ticker: str
    horizon: str
    instrument_ret: float
    spy_ret: float
    alpha: float
    beat: bool


@dataclass(frozen=True)
class MarketContext:
    """MARKET PLANE — attached AFTER the decision, never an input to it. Nullable:
    any market-plane failure yields None, and the decision is unaffected."""
    as_of_utc: str
    provider: str                    # "alpaca" | "yfinance" | "null"
    session_phase: str
    entry_anchor_utc: str            # first session open STRICTLY AFTER t0
    quotes: list[Quote] = field(default_factory=list)
    realized_alpha_since_t0: list[RealizedAlpha] | None = None
