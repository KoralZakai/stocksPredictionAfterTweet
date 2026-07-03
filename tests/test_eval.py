from datetime import date

from eval.baselines import majority_class, market_follow, permutation_null
from eval.metrics import accuracy, macro_f1
from eval.power import mde_gate
from eval.registry import Registry
from eval.significance import benjamini_hochberg, permutation_pvalue
from eval.splits import purged_walk_forward


def test_metrics_perfect_and_chance() -> None:
    y = ["UP", "DOWN", "NEUTRAL", "UP"]
    assert macro_f1(y, y) == 1.0
    assert accuracy(y, y) == 1.0
    assert macro_f1(y, ["NEUTRAL"] * 4) < 1.0


def test_baselines() -> None:
    assert majority_class(["UP", "UP", "DOWN"]) == "UP"
    assert market_follow([0.01, -0.02, 0.0]) == ["UP", "DOWN", "NEUTRAL"]


def test_permutation_pvalue_bounds() -> None:
    null = [0.1] * 100
    assert permutation_pvalue(0.9, null) == 1 / 101  # obs beats all null
    assert permutation_pvalue(0.05, null) == 101 / 101  # obs below all null


def test_permutation_null_shape() -> None:
    y = ["UP", "DOWN", "NEUTRAL", "UP", "DOWN"]
    null = permutation_null(y, ["UP"] * 5, macro_f1, n=50, seed=1)
    assert len(null) == 50


def test_benjamini_hochberg_monotone() -> None:
    q = benjamini_hochberg([0.01, 0.02, 0.5])
    assert all(0.0 <= x <= 1.0 for x in q)
    assert q[0] <= q[2]  # smaller p -> smaller-or-equal q


def test_registry_counts() -> None:
    r = Registry()
    r.register("XLE", 1, "gbt", 0.5)
    r.register("XLE", 2, "gbt", 0.5)
    assert len(r) == 2


def test_purged_walk_forward_embargo_and_order() -> None:
    sessions = [date(2024, 2, d) for d in range(1, 28) if date(2024, 2, d).weekday() < 5]
    s0 = [sessions[i] for i in range(12)]
    ids = [str(i) for i in range(12)]
    folds = purged_walk_forward(s0, ids, sessions, n_splits=3, embargo=3)
    assert folds
    for train, test in folds:
        # every train row is strictly before every test row (walk-forward) ...
        assert max(s0[i] for i in train) < min(s0[j] for j in test)
        # ... and separated by more than the embargo in sessions.
        gap = sessions.index(min(s0[j] for j in test)) - sessions.index(max(s0[i] for i in train))
        assert gap > 3


def test_small_n_is_underpowered() -> None:
    # N=10 against a near-balanced prior cannot detect a modest effect (§4).
    res = mde_gate(10, [0.34, 0.33, 0.33], alpha=0.05, n_sims=20, n_perm=100)
    assert res.n == 10 and (res.mde is None or res.mde > 0.2)
