"""Combined dual-track predictor (Macro + Micro) over the EOD relative-alpha labels.

Pools the two shipped datasets, adds a `track` feature so the model knows the environment,
and predicts the track's relative-alpha label:
  * macro rows -> label_eod       (basket beat SPY at EOD)
  * micro rows -> label_eod_soft  (stock beat the blended index/sector/peer mean at EOD)

Features (all tweet-time, no outcome leakage): intensity, phase, weekend_flag, market_closed,
scenario (macro: coarse family; micro: "micro"), track.

Models: XGBoost GBT (gbtree) + a logistic baseline (gblinear) — same lib, no sklearn dep.
Protocol: tune on VALIDATION (balanced accuracy), pick the best config, then score the SACRED
TEST set exactly once. Balanced accuracy + AUC reported overall and per track, against the
majority base rate (macro must beat ~0.64, micro ~0.51).

Run: PYTHONPATH=. .venv/Scripts/python.exe jobs/train_dual_engine.py
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
import xgboost as xgb

_STOP = frozenset(("the", "and", "for", "that", "this", "with", "will", "are", "was",
                   "its", "his", "her", "you", "have", "has", "not", "but", "they"))


def _hash_text(texts: list[str], k: int = 64) -> np.ndarray:
    """Deterministic hashed bag-of-words (no sklearn). md5 -> bucket, so it's stable
    across runs (unlike Python's salted hash())."""
    mat = np.zeros((len(texts), k), dtype=float)
    for i, t in enumerate(texts):
        for tok in re.findall(r"[a-z]{3,}", str(t).lower()):
            if tok not in _STOP:
                b = int(hashlib.md5(tok.encode()).hexdigest(), 16) % k
                mat[i, b] += 1.0
    return mat

MACRO = "reports/macro_dataset.csv"
MICRO = "reports/micro_dataset.csv"
FEATURES_NUM = ["intensity", "weekend_flag", "market_closed"]
FEATURES_CAT = ["phase", "scenario", "track"]
SEED = 0


def scenario_family(s: str) -> str:
    s = str(s).lower()
    fam = [
        ("geopolitics", ("geopolit", "war", "peace", "conflict", "military", "defense",
                         "nato", "iran", "russia", "ukraine", "security", "diplomacy")),
        ("trade", ("trade", "tariff", "protection", "china")),
        ("fed_rates", ("fed", "rate", "inflation", "monetary", "dollar")),
        ("energy", ("energy", "oil", "drill")),
        ("domestic", ("politic", "election", "domestic", "immigration", "fiscal", "economy")),
    ]
    for name, kws in fam:
        if any(k in s for k in kws):
            return name
    return "other"


def _load() -> pd.DataFrame:
    ma = pd.read_csv(MACRO)
    ma = ma[ma["label_eod"].notna()].copy()
    ma["target"] = ma["label_eod"].astype(int)
    ma["track"] = "macro"
    ma["scenario"] = ma["scenario"].map(scenario_family)

    mi = pd.read_csv(MICRO)
    mi = mi[mi["label_eod_soft"].notna()].copy()
    mi["target"] = mi["label_eod_soft"].astype(int)
    mi["track"] = "micro"
    mi["scenario"] = "micro"     # micro has no macro-scenario; track already separates it

    for d in (ma, mi):
        d["text"] = (d.get("summary", "").fillna("") + " " + d.get("macro_link", "").fillna(""))
    cols = ["intensity", "phase", "weekend_flag", "market_closed", "scenario", "track",
            "text", "split", "target"]
    return pd.concat([ma[cols], mi[cols]], ignore_index=True)


def _encode(df: pd.DataFrame):
    """Returns (X_meta, X_full=meta+hashed-text, y, split, track)."""
    df = df.copy()
    med = df.loc[df.split == "train", "intensity"].median()
    df["intensity"] = df["intensity"].fillna(med)
    Xm = pd.get_dummies(df[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT, dtype=float)
    txt = _hash_text(df["text"].tolist())
    names = list(Xm.columns) + [f"txt_{i}" for i in range(txt.shape[1])]
    Xfull = np.hstack([Xm.values, txt])
    return (Xm.values, list(Xm.columns)), (Xfull, names), df["target"].astype(int), \
        df["split"], df["track"]


def balanced_accuracy(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    pred = (p >= thr).astype(int)
    recalls = []
    for c in (0, 1):
        m = y == c
        if m.sum():
            recalls.append((pred[m] == c).mean())
    return float(np.mean(recalls)) if recalls else float("nan")


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = p.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _fit_predict(cfg: dict, Xtr, ytr, Xeval, names) -> np.ndarray:
    dtr = xgb.DMatrix(Xtr, label=ytr.values, feature_names=names)
    de = xgb.DMatrix(Xeval, feature_names=names)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": SEED,
              "nthread": 1, **cfg["params"]}
    bst = xgb.train(params, dtr, num_boost_round=cfg["rounds"], verbose_eval=False)
    return bst.predict(de)


GRID = [
    {"name": "GBT d2 e0.1 r60", "rounds": 60, "params": {"booster": "gbtree", "max_depth": 2, "eta": 0.1}},
    {"name": "GBT d3 e0.1 r80", "rounds": 80, "params": {"booster": "gbtree", "max_depth": 3, "eta": 0.1}},
    {"name": "GBT d2 e0.3 r40", "rounds": 40, "params": {"booster": "gbtree", "max_depth": 2, "eta": 0.3}},
    {"name": "Logistic (gblinear) L2=1", "rounds": 80, "params": {"booster": "gblinear", "lambda": 1.0, "alpha": 0.0}},
    {"name": "Logistic (gblinear) L2=0", "rounds": 80, "params": {"booster": "gblinear", "lambda": 0.0, "alpha": 0.0}},
]


def _base_rate(y: np.ndarray) -> float:
    return max(y.mean(), 1 - y.mean()) if len(y) else float("nan")


def _per_track(name: str, y, p, track) -> None:
    for tk in ("macro", "micro"):
        m = track.values == tk
        if m.sum():
            print(f"      {tk:6} n={int(m.sum()):3}  bal_acc={balanced_accuracy(y[m], p[m]):.3f}  "
                  f"auc={auc(y[m], p[m]):.3f}  base={_base_rate(y[m]):.3f}")


def _run_featureset(name: str, Xmat, names, y, split, track) -> None:
    tr, va, te = split == "train", split == "val", split == "test"
    print("\n" + "#" * 72)
    print(f"# FEATURE SET: {name}   ({Xmat.shape[1]} features)")
    print("#" * 72)
    best = None
    for cfg in GRID:
        p_va = _fit_predict(cfg, Xmat[tr.values], y[tr], Xmat[va.values], names)
        ba, ac = balanced_accuracy(y[va].values, p_va), auc(y[va].values, p_va)
        print(f"  {cfg['name']:26}  val bal_acc={ba:.3f}  auc={ac:.3f}")
        if best is None or ba > best[1]:
            best = (cfg, ba)
    cfg = best[0]
    p_va = _fit_predict(cfg, Xmat[tr.values], y[tr], Xmat[va.values], names)
    print(f"  >>> best on val: {cfg['name']} (bal_acc={best[1]:.3f})")
    _per_track("val", y[va].values, p_va, track[va])

    trva = (tr | va).values
    p_te = _fit_predict(cfg, Xmat[trva], y[tr | va], Xmat[te.values], names)
    print(f"  SACRED TEST (once): bal_acc={balanced_accuracy(y[te].values, p_te):.3f}  "
          f"auc={auc(y[te].values, p_te):.3f}  base={_base_rate(y[te].values):.3f}")
    _per_track("test", y[te].values, p_te, track[te])


def main() -> None:
    df = _load()
    (Xm, nm), (Xf, nf), y, split, track = _encode(df)
    va = split == "val"
    print(f"pooled: {len(df)} rows  ({(split=='train').sum()} train / {va.sum()} val "
          f"/ {(split=='test').sum()} test)")
    print(f"base rates  val: overall={_base_rate(y[va].values):.3f}  "
          f"macro={_base_rate(y[va & (track=='macro')].values):.3f}  "
          f"micro={_base_rate(y[va & (track=='micro')].values):.3f}")
    print("Protocol: tune each feature set on VAL, score sacred TEST once. Compare the two "
          "to isolate the TEXT contribution (Option C).")
    _run_featureset("metadata only", Xm, nm, y, split, track)
    _run_featureset("metadata + LLM text (summary + macro_link)", Xf, nf, y, split, track)
    print("\n" + "=" * 72)
    print("Read: text ADDS value only if it lifts TEST auc above the metadata-only run "
          "AND above 0.5. Small n -> directional.")


if __name__ == "__main__":
    main()
