"""Gradient-boosted trees on STRUCTURED features (§8, primary model).

Native xgboost Booster API (no sklearn dependency). Feature vectors are built
from Row.features with a fixed, stored column order so train and serve cannot
disagree on layout. At ~150-300 rows this is the honest modeling ceiling; raw
embeddings are deliberately NOT fed here (§8).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from config.settings import SETTINGS
from dataset.build import Row
from eval.metrics import CLASSES


def _vector(features: dict[str, float], names: Sequence[str]) -> list[float]:
    return [features.get(n, 0.0) for n in names]


def feature_names(features: dict[str, float], market_only: bool = False) -> list[str]:
    """Stable column order. market_only drops the tweet topic one-hots (§7 ablation)."""
    names = sorted(features)
    return [n for n in names if not n.startswith("topic_")] if market_only else names


@dataclass(frozen=True)
class GBTModel:
    booster: xgb.Booster
    names: list[str]
    classes: tuple[str, ...]

    @classmethod
    def fit(
        cls, rows: Sequence[Row], horizon: int, market_only: bool = False
    ) -> GBTModel | None:
        usable = [r for r in rows if r.label[horizon] in CLASSES]
        if len(usable) < len(CLASSES):  # need at least one row per class to be meaningful
            return None
        names = feature_names(usable[0].features, market_only)
        x = np.array([_vector(r.features, names) for r in usable], dtype=np.float32)
        y = np.array([CLASSES.index(r.label[horizon]) for r in usable], dtype=np.int32)

        params = {
            "objective": "multi:softprob",
            "num_class": len(CLASSES),
            "max_depth": 3,
            "eta": 0.1,
            "seed": SETTINGS.seed,
            "nthread": 1,  # deterministic on small data
            "verbosity": 0,
        }
        booster = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=50)
        return cls(booster, names, CLASSES)

    def predict_proba(self, features: dict[str, float]) -> dict[str, float]:
        x = np.array([_vector(features, self.names)], dtype=np.float32)
        probs = self.booster.predict(xgb.DMatrix(x))[0]
        return {c: float(p) for c, p in zip(self.classes, probs, strict=True)}

    def save(self, out_dir: str | Path) -> None:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / "gbt.json"))
        (d / "meta.json").write_text(
            json.dumps({"names": self.names, "classes": list(self.classes)}), encoding="utf-8"
        )

    @classmethod
    def load(cls, in_dir: str | Path) -> GBTModel:
        d = Path(in_dir)
        booster = xgb.Booster()
        booster.load_model(str(d / "gbt.json"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return cls(booster, meta["names"], tuple(meta["classes"]))
