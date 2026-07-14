"""Nebius Job: train_multibench (§13) — train/test the multi-benchmark direction model.

Reads data/real/labeled_multibench.csv, builds the SHARED point-in-time feature
vector (models/multibench_features.py — same function serving uses, so no skew),
and fits an XGBoost classifier to the folded label at one horizon.

TRAIN/TEST is a TIME-ORDERED split (never random — rows overlap in time, §3.6):
the earliest `train_frac` of tweets train the model, the most recent tail is the
held-out TEST set the model has never seen — the honest "predict on new data"
check the user asked for. Reports test macro-F1 against the always-majority
baseline; a model that cannot beat majority on a weak/near-null signal is the
expected, valid outcome (§1).

Rows whose label is NA (missing bars / no sector / no vol) are dropped for that
horizon before splitting. Deterministic + run-id stamped.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe jobs/train_multibench.py \
      --data data/real/labeled_multibench.csv --out runs/real/multibench_model --horizon 3d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.multibench_features import FEATURE_ORDER, feature_vector

CLASSES = ("DOWN", "NEUTRAL", "UP")
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


def _row_features(r: pd.Series) -> dict[str, float]:
    return feature_vector(
        stance=str(r.get("stance", "")),
        match_tier=str(r.get("match_tier", "")),
        sectors=str(r.get("sectors_used", "") or "").split(),
        indices=str(r.get("indices_used", "") or "").split(),
        pre_vol=_num(r.get("pre_vol")),
        pre_ret_1=_num(r.get("pre_ret_1")),
        pre_ret_3=_num(r.get("pre_ret_3")),
        weekday=int(r.get("weekday", 0) or 0),
        after_hours=int(r.get("after_hours", 0) or 0),
        used_fallback=int(r.get("used_fallback", 0) or 0),
    )


def _num(v: object) -> float | None:
    return None if v is None or (isinstance(v, float) and v != v) else float(v)  # type: ignore[arg-type]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in range(len(CLASSES)):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def train(data: str, out: str, horizon: str, train_frac: float = 0.7) -> None:
    import xgboost as xgb

    df = pd.read_csv(data)
    label_col = f"label_{horizon}"
    if label_col not in df.columns:
        raise SystemExit(f"no column {label_col}; horizons: "
                         f"{[c[6:] for c in df.columns if c.startswith('label_')]}")

    df = df[df[label_col].isin(CLASSES)].copy()
    df = df.sort_values("tweet_date").reset_index(drop=True)
    if len(df) < 2 * len(CLASSES):
        raise SystemExit(f"only {len(df)} labeled rows at {horizon} — too few to split.")

    X = np.array([[fv[k] for k in FEATURE_ORDER]
                  for fv in (_row_features(r) for _, r in df.iterrows())], dtype=float)
    y = df[label_col].map(CLASS_IDX).to_numpy()

    cut = int(len(df) * train_frac)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=list(FEATURE_ORDER))
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": len(CLASSES), "max_depth": 3,
         "eta": 0.1, "seed": 0, "nthread": 1, "verbosity": 0},
        dtr, num_boost_round=50,
    )

    # majority baseline from TRAIN, applied to TEST (honest baseline).
    majority = int(np.bincount(ytr, minlength=len(CLASSES)).argmax())
    dte = xgb.DMatrix(Xte, feature_names=list(FEATURE_ORDER))
    pred = booster.predict(dte).argmax(axis=1) if len(Xte) else np.array([])
    base = np.full(len(yte), majority)

    f1_model = _macro_f1(yte, pred) if len(yte) else 0.0
    f1_base = _macro_f1(yte, base) if len(yte) else 0.0

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.json"))
    meta = {
        "horizon": horizon, "features": list(FEATURE_ORDER), "classes": list(CLASSES),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "test_macro_f1": round(f1_model, 4), "majority_macro_f1": round(f1_base, 4),
        "beats_majority": bool(f1_model > f1_base),
        "test_label_balance": {c: int((yte == i).sum()) for c, i in CLASS_IDX.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"trained on {len(Xtr)} / tested on {len(Xte)} rows @ {horizon} -> {out_dir}")
    print(f"  TEST macro-F1: model {f1_model:.3f}  vs  majority {f1_base:.3f}  "
          f"({'beats' if meta['beats_majority'] else 'does NOT beat'} baseline)")
    print(f"  test label balance: {meta['test_label_balance']}")
    print("  (weak/near-null signal -> not beating majority is the expected result, "
          "not a bug — §1.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/labeled_multibench.csv")
    ap.add_argument("--out", default="runs/real/multibench_model")
    ap.add_argument("--horizon", default="3d")
    ap.add_argument("--train-frac", type=float, default=0.7)
    a = ap.parse_args()
    train(a.data, a.out, a.horizon, a.train_frac)


if __name__ == "__main__":
    main()
