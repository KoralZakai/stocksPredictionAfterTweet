"""Pre-registered configuration — every scientific choice lives here (§5).

Frozen on purpose: changing a value is a pre-registration change recorded in
git, not a runtime tweak. Import `SETTINGS` everywhere; never hardcode a
sector list, k, alpha, or horizon elsewhere.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Sector-ETF universe (the challenge's 10 tickers) + market benchmark (§Data).
    etfs: tuple[str, ...] = (
        "XLK", "XLE", "XLF", "XLI", "XLV", "XLP", "XLY", "XLB", "SMH", "ITA",
    )
    benchmark: str = "SPY"

    # Entry anchor (§3.3). "open" = leak-free primary; "close" = diagnostic only.
    entry_anchor: str = "open"

    # Drift horizons in sessions (§3.4): ret_1d/2d/3d -> close offsets 0,1,2 from s0.
    horizon_days: tuple[int, ...] = (1, 2, 3)

    # Label band (§3.5): thresholds = ±k * σ_backward, σ backward-only rolling vol.
    k: float = 0.5
    vol_window_sessions: int = 20

    # Evaluation (§3.6, §4).
    embargo_sessions: int = 3
    alpha: float = 0.05

    # Conformal abstention target coverage (§8), pre-registered.
    conformal_coverage: float = 0.9

    # Reproducibility (§12).
    seed: int = 0
    schema_version: int = 1

    @field_validator("embargo_sessions")
    @classmethod
    def _embargo_min(cls, v: int) -> int:
        if v < 3:
            raise ValueError("embargo must be >= 3 sessions (§3.6)")
        return v

    @field_validator("k", "alpha")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


SETTINGS = Settings()
