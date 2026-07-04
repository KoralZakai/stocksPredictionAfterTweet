"""Nebius Job: training (§13, build step 8).

Fit the GBT on structured features and calibrate the conformal abstainer on a
held-out, time-ordered tail (earlier rows train, later rows calibrate — never
random, §3.6). Persists the Predictor bundle the endpoint loads.

Run:  python jobs/training.py --dataset runs/mvp/dataset.json --out runs/mvp/model --horizon 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import SETTINGS
from dataset.build import Row, rows_from_json
from eval.metrics import CLASSES
from models.abstain import ConformalAbstainer
from models.gbt import GBTModel
from models.predictor import Predictor


def train(rows: list[Row], horizon: int, cal_frac: float = 0.3) -> Predictor | None:
    usable = sorted(
        (r for r in rows if r.label[horizon] in CLASSES), key=lambda r: r.t0
    )
    if len(usable) < 2 * len(CLASSES):
        return None  # too few to both fit and calibrate honestly
    cut = int(len(usable) * (1 - cal_frac))
    train_rows, cal_rows = usable[:cut], usable[cut:]

    model = GBTModel.fit(train_rows, horizon)
    if model is None or not cal_rows:
        return None
    conformal = ConformalAbstainer.calibrate(
        [model.predict_proba(r.features) for r in cal_rows],
        [r.label[horizon] for r in cal_rows],
        coverage=SETTINGS.conformal_coverage,
    )
    return Predictor(model, conformal)


def run(dataset_path: str, out_dir: str, horizon: int) -> None:
    rows = rows_from_json(Path(dataset_path).read_text(encoding="utf-8"))
    predictor = train(rows, horizon)
    if predictor is None:
        print(f"insufficient data to train horizon {horizon} "
              f"({len(rows)} rows) — no model written (honest at small N).")
        return
    predictor.save(out_dir)
    print(f"trained + calibrated horizon {horizon} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=3)
    a = ap.parse_args()
    run(a.dataset, a.out, a.horizon)


if __name__ == "__main__":
    main()
