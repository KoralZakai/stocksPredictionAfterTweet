"""Predictor bundle: GBT + conformal abstainer (§8).

Satisfies core.decide.DirectionPredictor structurally, so `decide(tweet, state,
predictor)` routes model predictions through the ONE decision path — the batch
runner and the /predict endpoint load the same bundle, no skew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from models.abstain import ConformalAbstainer
from models.gbt import GBTModel


@dataclass(frozen=True)
class Predictor:
    model: GBTModel
    conformal: ConformalAbstainer

    def predict_direction(self, features: dict[str, float]) -> tuple[str, float, bool]:
        return self.conformal.decide(self.model.predict_proba(features))

    def save(self, out_dir: str | Path) -> None:
        d = Path(out_dir)
        self.model.save(d)
        (d / "conformal.json").write_text(
            json.dumps({"q": self.conformal.q, "classes": list(self.conformal.classes)}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, in_dir: str | Path) -> Predictor:
        d = Path(in_dir)
        meta = json.loads((d / "conformal.json").read_text(encoding="utf-8"))
        conformal = ConformalAbstainer(meta["q"], tuple(meta["classes"]))
        return cls(GBTModel.load(d), conformal)
