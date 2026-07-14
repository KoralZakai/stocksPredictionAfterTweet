"""Shared point-in-time feature vector for the multi-benchmark model (§3.2).

ONE pure function, `feature_vector`, called by BOTH the training job
(jobs/train_multibench.py) and the serving path — so train and serve cannot
build features differently (no train/serve skew). Every input is knowable strictly
before t0: the LLM stance (text-only), the entity match tier, the stock's index/
sector membership (static), and pre-event market context (vol + prior returns from
bars closed before t0). NO outcome/abnormal-return value is a feature — those are
the label.

Column order is fixed and sorted so the XGBoost matrix is stable across runs.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from config.settings import SETTINGS

SECTORS: tuple[str, ...] = SETTINGS.etfs           # 10 sector ETFs, fixed order
INDICES: tuple[str, ...] = ("SPY", "QQQ", "DIA")


def pre_context(prior_closes: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    """(pre_vol, pre_ret_1, pre_ret_3) from closes of sessions BEFORE t0.

    pre_vol = pstdev of daily returns over the pre-registered vol window;
    pre_ret_1/3 = cumulative return over the last 1 / 3 prior sessions. None when
    there is not enough history (honest, never fabricated).
    """
    c = list(prior_closes)
    w = SETTINGS.vol_window_sessions
    pre_vol: float | None = None
    if len(c) >= w + 1:
        rets = [c[i] / c[i - 1] - 1.0 for i in range(len(c) - w, len(c))]
        pre_vol = statistics.pstdev(rets)
    pre_ret_1 = c[-1] / c[-2] - 1.0 if len(c) >= 2 else None
    pre_ret_3 = c[-1] / c[-4] - 1.0 if len(c) >= 4 else None
    return pre_vol, pre_ret_1, pre_ret_3


def feature_vector(
    *,
    stance: str,
    match_tier: str,
    sectors: Sequence[str],
    indices: Sequence[str],
    pre_vol: float | None,
    pre_ret_1: float | None,
    pre_ret_3: float | None,
    weekday: int,
    after_hours: int,
    used_fallback: int,
) -> dict[str, float]:
    """Flat, stable-order feature dict. Missing market context -> 0.0 (neutral)."""
    f: dict[str, float] = {
        "stance_positive": float(stance == "positive"),
        "stance_negative": float(stance == "negative"),
        "tier_direct": float(match_tier == "direct"),
        "pre_vol": float(pre_vol or 0.0),
        "pre_ret_1": float(pre_ret_1 or 0.0),
        "pre_ret_3": float(pre_ret_3 or 0.0),
        "weekday": float(weekday),
        "after_hours": float(after_hours),
        "used_fallback": float(used_fallback),
    }
    sset, iset = set(sectors), set(indices)
    for etf in SECTORS:
        f[f"sec_{etf}"] = float(etf in sset)
    for idx in INDICES:
        f[f"idx_{idx}"] = float(idx in iset)
    return dict(sorted(f.items()))


FEATURE_ORDER: tuple[str, ...] = tuple(
    feature_vector(
        stance="neutral", match_tier="", sectors=(), indices=(),
        pre_vol=None, pre_ret_1=None, pre_ret_3=None,
        weekday=0, after_hours=0, used_fallback=0,
    ).keys()
)


def _demo() -> None:
    v = feature_vector(
        stance="positive", match_tier="direct", sectors=["SMH"], indices=["SPY", "QQQ"],
        pre_vol=0.02, pre_ret_1=0.01, pre_ret_3=-0.005, weekday=1, after_hours=1, used_fallback=0,
    )
    assert v["stance_positive"] == 1.0 and v["stance_negative"] == 0.0
    assert v["sec_SMH"] == 1.0 and v["idx_SPY"] == 1.0 and v["idx_DIA"] == 0.0
    assert list(v.keys()) == list(FEATURE_ORDER)          # stable order holds
    pv, r1, r3 = pre_context([100, 101, 102, 103, 104] * 5)
    assert pv is not None and r1 is not None
    print(f"multibench_features _demo OK: {len(FEATURE_ORDER)} features")


if __name__ == "__main__":
    _demo()
