"""Conformal abstention (§8) — class-conditional (Mondrian) split conformal.

NOT softmax>k. We calibrate a per-class nonconformity quantile on held-out data,
then form a prediction SET at the pre-registered coverage. We emit a class only
when the set is a singleton; an empty or multi-class set is an abstention. This
gives a distribution-free coverage guarantee that a raw probability threshold
does not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from config.settings import SETTINGS
from eval.metrics import CLASSES


@dataclass(frozen=True)
class ConformalAbstainer:
    q: dict[str, float]  # per-class nonconformity threshold (Mondrian)
    classes: tuple[str, ...]

    @classmethod
    def calibrate(
        cls,
        cal_proba: Sequence[dict[str, float]],
        cal_true: Sequence[str],
        coverage: float = SETTINGS.conformal_coverage,
        classes: tuple[str, ...] = CLASSES,
    ) -> ConformalAbstainer:
        q: dict[str, float] = {}
        for c in classes:
            scores = sorted(
                1.0 - p[c] for p, y in zip(cal_proba, cal_true, strict=True) if y == c
            )
            if not scores:
                # No calibration data for c -> we can't vouch for its coverage, so
                # exclude it from every set (1-p <= -1 is never true). Conservative:
                # the true label may then fall out -> abstain, never a wrong emit.
                q[c] = -1.0
                continue
            n = len(scores)
            k = math.ceil((n + 1) * coverage)  # conformal quantile rank
            q[c] = scores[min(k, n) - 1]
        return cls(q, classes)

    def prediction_set(self, proba: dict[str, float]) -> set[str]:
        return {c for c in self.classes if (1.0 - proba[c]) <= self.q[c]}

    def decide(self, proba: dict[str, float]) -> tuple[str, float, bool]:
        """-> (direction, confidence, abstain). Singleton set = emit; else abstain."""
        s = self.prediction_set(proba)
        if len(s) == 1:
            c = next(iter(s))
            return c, proba[c], False
        return "ABSTAIN", max(proba.values()), True
