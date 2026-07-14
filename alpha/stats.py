"""Small-sample statistics for the validation manifest — stdlib only (no scipy).

Three primitives, all pure and deterministic:
  wilson_ci     - 95% CI for a binomial proportion. Correct at small n, where the
                  normal approximation is not (n_test here is ~89 tweets).
  binom_p_greater - exact one-sided binomial tail P(X >= k | n, p0). The null for
                  a beat-SPY hit rate is p0 = 0.5 (relative alpha strips beta).
  benjamini_hochberg - BH step-up adjusted p-values across the test registry, so
                  "survives correction" means survives the WHOLE horizon set, not
                  a single cherry-picked cell.
"""

from __future__ import annotations

from math import comb, sqrt


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials. Returns (low, high).

    Empty n -> (0.0, 1.0): maximal uncertainty, never a divide-by-zero.
    """
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def binom_p_greater(k: int, n: int, p0: float = 0.5) -> float:
    """Exact one-sided p-value: P(X >= k | Binomial(n, p0)). Small-n honest.

    Used to test H1: beat-SPY hit rate > 50%. Returns 1.0 for n == 0.
    """
    if n <= 0:
        return 1.0
    k = max(0, min(k, n))
    tail = sum(comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(k, n + 1))
    return min(1.0, tail)


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH step-up adjusted p-values, returned in the INPUT order.

    adj_(i) = min_{j >= i} ( p_(j) * m / rank_(j) ), then clamped to <= 1 and made
    monotone. Empty input -> empty output.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])          # ascending by p
    adj = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):                              # largest p first
        idx = order[rank - 1]
        val = pvals[idx] * m / rank
        running_min = min(running_min, val)
        adj[idx] = min(1.0, running_min)
    return adj


def _demo() -> None:
    # Wilson: 60/100 -> ~[0.503, 0.691] (known reference values).
    lo, hi = wilson_ci(60, 100)
    assert 0.49 < lo < 0.51 and 0.68 < hi < 0.70, (lo, hi)
    # Wilson degenerate guards.
    assert wilson_ci(0, 0) == (0.0, 1.0)
    assert wilson_ci(5, 5)[1] == 1.0
    # Binomial: 50/50 fair coin -> exactly 0.5 mass at/above the mean+... P(X>=25|50,.5)
    assert abs(binom_p_greater(25, 50) - 0.5561375) < 1e-6, binom_p_greater(25, 50)
    assert binom_p_greater(50, 50) == 0.5**50
    assert binom_p_greater(0, 10) == 1.0
    # A clear signal: 58/89 heads is significant vs a fair coin.
    assert binom_p_greater(58, 89) < 0.005
    # BH: monotone, order-preserving, clamped.
    raw = [0.01, 0.02, 0.03, 0.04, 0.05]
    adj = benjamini_hochberg(raw)
    assert all(a <= 1.0 for a in adj)
    assert adj[0] <= adj[1] <= adj[2]                          # sorted input stays ordered
    # single p passes through (m/rank = 1).
    assert abs(benjamini_hochberg([0.03])[0] - 0.03) < 1e-12
    assert benjamini_hochberg([]) == []
    print("alpha.stats self-check OK")


if __name__ == "__main__":
    _demo()
