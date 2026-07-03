"""Power / MDE gate (§4) — run BEFORE modeling.

Given N usable rows and the class priors, simulate against the permutation null
to find the minimum detectable effect (MDE) at the target power. If the MDE
exceeds any plausible tweet effect, the study is underpowered BY CONSTRUCTION —
and that is recorded as a primary finding, not hidden.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from eval.metrics import CLASSES, macro_f1
from eval.significance import permutation_pvalue


@dataclass(frozen=True)
class PowerResult:
    n: int
    mde: float | None  # smallest detectable accuracy lift, or None if none up to the grid max
    powered: bool
    target_power: float


def _sample(priors: Sequence[float], n: int, rng: random.Random) -> list[str]:
    return rng.choices(CLASSES, weights=priors, k=n)


def _power_at(
    n: int, priors: Sequence[float], lift: float, alpha: float,
    n_sims: int, n_perm: int, rng: random.Random,
) -> float:
    base = max(priors)
    p_correct = min(1.0, base + lift)
    sig = 0
    for _ in range(n_sims):
        y_true = _sample(priors, n, rng)
        y_pred = [
            t if rng.random() < p_correct else rng.choice(CLASSES) for t in y_true
        ]
        obs = macro_f1(y_true, y_pred)
        null = [macro_f1(rng.sample(y_true, n), y_pred) for _ in range(n_perm)]
        if permutation_pvalue(obs, null) <= alpha:
            sig += 1
    return sig / n_sims


def mde_gate(
    n: int,
    priors: Sequence[float],
    alpha: float,
    target_power: float = 0.8,
    n_sims: int = 60,
    n_perm: int = 200,
    seed: int = 0,
) -> PowerResult:
    rng = random.Random(seed)
    for lift in (i / 20 for i in range(1, 11)):  # 0.05 .. 0.50
        if _power_at(n, priors, lift, alpha, n_sims, n_perm, rng) >= target_power:
            return PowerResult(n, round(lift, 3), True, target_power)
    return PowerResult(n, None, False, target_power)
