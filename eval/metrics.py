"""Classification metrics (§5 eval/metrics.py).

Macro-averaged so a majority-NEUTRAL split can't inflate the score — the
honest metric under class imbalance. Pure Python, deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence

CLASSES = ("UP", "DOWN", "NEUTRAL")


def _f1_for(y_true: Sequence[str], y_pred: Sequence[str], c: str) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p == c)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != c and p == c)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p != c)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def macro_f1(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str] = CLASSES
) -> float:
    if not y_true:
        return 0.0
    return sum(_f1_for(y_true, y_pred, c) for c in classes) / len(classes)


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    hits = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return hits / len(y_true)
