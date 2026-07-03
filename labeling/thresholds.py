"""Volatility-scaled label band (§3.5, §5 labeling/thresholds.py).

label = UP/DOWN/NEUTRAL by comparing a horizon return against ±k·σ_backward,
where σ_backward is the ticker's daily close-to-close vol computed ONLY from
sessions strictly before s0 (backward-only — no full-series statistics).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import date

from data.sources.interfaces import DailyBar

LABELS = ("UP", "DOWN", "NEUTRAL", "NA")


def backward_vol(bars: Sequence[DailyBar], s0_date: date, window: int) -> float | None:
    """Population stdev of daily returns over the `window` sessions before s0."""
    prior = sorted(
        (b for b in bars if b.session_date.date() < s0_date),
        key=lambda b: b.session_date,
    )
    closes = [b.close for b in prior[-(window + 1) :]]
    if len(closes) < 3:
        return None
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    return statistics.pstdev(rets)


def label(ret: float | None, sigma: float | None, k: float) -> str:
    if ret is None or sigma is None:
        return "NA"
    thr = k * sigma
    if ret > thr:
        return "UP"
    if ret < -thr:
        return "DOWN"
    return "NEUTRAL"


def class_balance(labels: Sequence[str]) -> dict[str, int]:
    """First-class diagnostic (§3.5): a degenerate split is a labeling failure."""
    return {c: sum(1 for x in labels if x == c) for c in LABELS}
