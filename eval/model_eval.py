"""Out-of-fold GBT evaluation via purged walk-forward CV (§7).

Trains the GBT on each fold's train rows and predicts the held-out test rows,
collecting out-of-fold predictions. `market_only=True` drops the tweet topic
features — the two runs together are the text-ablation: if market-only matches
with-text, the tweets add nothing and the claim is dead.
"""

from __future__ import annotations

from collections.abc import Sequence

from config.settings import SETTINGS, Settings
from dataset.build import Row
from eval.metrics import CLASSES, macro_f1
from eval.splits import purged_walk_forward
from models.gbt import GBTModel


def cv_out_of_fold(
    rows: Sequence[Row], horizon: int, market_only: bool, cfg: Settings = SETTINGS
) -> tuple[list[str], list[str]]:
    """Return (y_true, y_pred) pooled over walk-forward test folds. Argmax, not
    conformal — here we measure discriminative signal, not calibrated abstention."""
    s0 = [r.s0_date for r in rows]
    ids = [r.tweet_id for r in rows]
    # ponytail: session axis = the label dates present; embargo is then counted in
    # tweet-sessions, a monotone proxy for true trading sessions (which the dataset
    # artifact doesn't carry). Tighten by threading the price calendar if it matters.
    sessions = sorted(set(s0))
    folds = purged_walk_forward(s0, ids, sessions, n_splits=3, embargo=cfg.embargo_sessions)

    y_true: list[str] = []
    y_pred: list[str] = []
    for train_idx, test_idx in folds:
        model = GBTModel.fit([rows[i] for i in train_idx], horizon, market_only)
        if model is None:
            continue
        for i in test_idx:
            if rows[i].label[horizon] not in CLASSES:
                continue
            proba = model.predict_proba(rows[i].features)
            y_pred.append(max(proba, key=lambda c: proba[c]))
            y_true.append(rows[i].label[horizon])
    return y_true, y_pred


def cv_macro_f1(
    rows: Sequence[Row], horizon: int, market_only: bool, cfg: Settings = SETTINGS
) -> tuple[float, int, list[str], list[str]]:
    y_true, y_pred = cv_out_of_fold(rows, horizon, market_only, cfg)
    return macro_f1(y_true, y_pred), len(y_true), y_true, y_pred
