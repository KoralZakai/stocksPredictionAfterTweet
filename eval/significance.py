"""Permutation p-values + Benjamini-Hochberg correction (§4).

Nothing is called significant unless it survives BH over the FULL registry
(§5 registry.py provides the denominator). The correct headline may be "nothing
survives" — that is a success, not a failure.
"""

from __future__ import annotations

from collections.abc import Sequence


def permutation_pvalue(observed: float, null_samples: Sequence[float]) -> float:
    """One-sided (larger = better). (1 + #{null >= observed}) / (1 + n) — never 0."""
    ge = sum(1 for x in null_samples if x >= observed)
    return (1 + ge) / (1 + len(null_samples))


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """BH-adjusted q-values, same order as input. Compare q <= alpha to reject."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):  # from largest p to smallest, enforce monotonicity
        i = order[rank]
        val = pvalues[i] * m / (rank + 1)
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return q
