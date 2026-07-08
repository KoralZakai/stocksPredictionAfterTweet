"""First baseline classifier: abnormal-return direction (UP/DOWN/NEUTRAL), 3d.

Honest scope: a FIRST baseline on a TIME-ordered split (train early, test late) —
NOT the rigorous verdict. The real evaluation (purged+embargoed walk-forward,
permutation null, BH correction, §4/§7) comes later and is what decides whether
anything survives. Here we just check: does a GBT beat always-majority at all?

Stocks + ETFs are pooled into one model (asset identity is NOT a feature — we
don't want it memorizing per-asset base rates; the model sees market-state +
topic + timing only).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/baseline_model.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from config.settings import SETTINGS

LAB = "data/real/labeled.csv"
TARGET = "lab_3"
CLASSES = ["DOWN", "NEUTRAL", "UP"]
FEATURES = (
    ["pre_r1", "pre_r3", "pre_r5", "pre_r10", "pre_r20", "pre_vol", "rel_spy5",
     "relevance", "trump_exposed", "weekday", "hour", "after_hours", "is_etf"]
    + [f"topic_{e}" for e in SETTINGS.etfs]
)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    f1s: dict[str, float] = {}
    for c in range(3):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[CLASSES[c]] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return float(np.mean(list(f1s.values()))), f1s


def main() -> None:
    df = pd.read_csv(LAB).dropna(subset=[TARGET]).sort_values("timestamp_utc")
    df = df[df["is_spy"] == 0]  # SPY abnormal is ~0 by construction
    y = df[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X = df[FEATURES].astype(float).to_numpy()

    cut = int(len(df) * 0.7)  # time-ordered split: earlier -> train, later -> test
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    split_date = str(df["timestamp_utc"].iloc[cut])[:10]
    print(f"rows: {len(df)}  train: {len(Xtr)}  test: {len(Xte)}  split at {split_date}")
    print(f"test label balance: "
          f"{ {CLASSES[i]: int((yte==i).sum()) for i in range(3)} }")

    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": 3, "max_depth": 4,
         "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "seed": SETTINGS.seed},
        xgb.DMatrix(Xtr, label=ytr), num_boost_round=120,
    )
    pred = booster.predict(xgb.DMatrix(Xte)).argmax(axis=1)

    maj = int(np.bincount(ytr).argmax())  # always-majority baseline
    base_pred = np.full_like(yte, maj)
    gbt_acc = float((pred == yte).mean())
    base_acc = float((base_pred == yte).mean())
    gbt_f1, gbt_per = macro_f1(yte, pred)
    base_f1, _ = macro_f1(yte, base_pred)

    print(f"\n{'model':16}{'accuracy':>10}{'macro-F1':>10}")
    print(f"{'always-majority':16}{base_acc:>10.3f}{base_f1:>10.3f}")
    print(f"{'GBT (text+market)':16}{gbt_acc:>10.3f}{gbt_f1:>10.3f}")
    print(f"per-class F1 (GBT): { {k: round(v,3) for k,v in gbt_per.items()} }")
    verdict = ("beats" if gbt_f1 > base_f1 else "does NOT beat")
    print(f"\nFIRST-BASELINE READ: GBT {verdict} always-majority on macro-F1.")
    print("This is NOT significance — purged CV + permutation null + BH decide that (§4/§7).")

    # ponytail: one runnable check — predictions are valid class indices
    assert set(np.unique(pred)).issubset({0, 1, 2})


if __name__ == "__main__":
    main()
