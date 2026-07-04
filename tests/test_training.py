from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dataset.build import Row
from eval.metrics import CLASSES
from jobs.training import train
from models.predictor import Predictor

UTC = timezone.utc


def _row(i: int, prior: float, lbl: str) -> Row:
    feats = {"topic_XLE": 1.0, "prior_1d_ret": prior, "prior_3d_ret": prior,
             "backward_vol": 0.01, "spy_trend": 1.0, "weekday": 1.0}
    t0 = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
    return Row(str(i), "XLE", t0, date(2024, 2, 6), 1.0, feats,
               {1: prior, 2: prior, 3: prior}, {1: lbl, 2: lbl, 3: lbl})


def _dataset() -> list[Row]:
    out = []
    for i in range(30):
        prior, lbl = [(0.05, "UP"), (-0.05, "DOWN"), (0.0, "NEUTRAL")][i % 3]
        out.append(_row(i, prior, lbl))
    return out


def test_train_save_load_predict(tmp_path: Path) -> None:
    predictor = train(_dataset(), horizon=1)
    assert predictor is not None

    predictor.save(tmp_path)
    loaded = Predictor.load(tmp_path)

    direction, conf, abstain = loaded.predict_direction(
        {"topic_XLE": 1.0, "prior_1d_ret": 0.05, "prior_3d_ret": 0.05,
         "backward_vol": 0.01, "spy_trend": 1.0, "weekday": 1.0}
    )
    assert direction in (*CLASSES, "ABSTAIN") and isinstance(abstain, bool)
    assert 0.0 <= conf <= 1.0


def test_train_returns_none_on_tiny_data() -> None:
    assert train(_dataset()[:3], horizon=1) is None  # too few to fit + calibrate
