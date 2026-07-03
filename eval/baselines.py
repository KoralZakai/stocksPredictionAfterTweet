"""First-class baselines (§4). Nothing is a "signal" until it beats these.

- majority: always predict the training-majority class.
- market-follow: predict the sign of the prior 1-day return (momentum) — the
  cheap "the market was already moving" explanation a tweet must beat.
- permutation null: shuffle the labels to break any tweet<->outcome link and
  recompute the metric, giving that metric's distribution under H0.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence


def majority_class(y_train: Sequence[str]) -> str:
    return Counter(y_train).most_common(1)[0][0] if y_train else "NEUTRAL"


def market_follow(prior_1d_ret: Sequence[float]) -> list[str]:
    return ["UP" if r > 0 else "DOWN" if r < 0 else "NEUTRAL" for r in prior_1d_ret]


def permutation_null(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    n: int = 1000,
    seed: int = 0,
) -> list[float]:
    """Null distribution of `metric`: permute labels vs the fixed predictions."""
    rng = random.Random(seed)
    labels = list(y_true)
    out: list[float] = []
    for _ in range(n):
        rng.shuffle(labels)
        out.append(metric(labels, y_pred))
    return out
