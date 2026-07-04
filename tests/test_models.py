from datetime import date, datetime, timezone

from dataset.build import Row
from models.abstain import ConformalAbstainer
from models.gbt import GBTModel, feature_names

UTC = timezone.utc


def _row(rid: str, prior: float, lbl: str) -> Row:
    # prior_1d_ret carries the signal; label follows its sign so GBT can learn it.
    feats = {
        "topic_XLE": 1.0, "prior_1d_ret": prior, "prior_3d_ret": prior,
        "backward_vol": 0.01, "spy_trend": 1.0, "weekday": 1.0,
    }
    return Row(rid, "XLE", datetime(2024, 2, 5, tzinfo=UTC), date(2024, 2, 6),
               1.0, feats, {1: prior, 2: prior, 3: prior}, {1: lbl, 2: lbl, 3: lbl})


def _dataset() -> list[Row]:
    rows = []
    for i in range(30):
        if i % 3 == 0:
            rows.append(_row(str(i), 0.05, "UP"))
        elif i % 3 == 1:
            rows.append(_row(str(i), -0.05, "DOWN"))
        else:
            rows.append(_row(str(i), 0.0, "NEUTRAL"))
    return rows


def test_gbt_learns_separable_signal() -> None:
    model = GBTModel.fit(_dataset(), horizon=1)
    assert model is not None
    up = model.predict_proba({"topic_XLE": 1.0, "prior_1d_ret": 0.05, "prior_3d_ret": 0.05,
                              "backward_vol": 0.01, "spy_trend": 1.0, "weekday": 1.0})
    assert max(up, key=lambda c: up[c]) == "UP"


def test_market_only_drops_topic_features() -> None:
    names = feature_names(_dataset()[0].features, market_only=True)
    assert not any(n.startswith("topic_") for n in names)
    assert "prior_1d_ret" in names


def test_conformal_singleton_vs_abstain() -> None:
    # Confident, well-calibrated -> emit; flat probs -> abstain.
    cal_proba = [{"UP": 0.9, "DOWN": 0.05, "NEUTRAL": 0.05} for _ in range(20)]
    conf = ConformalAbstainer.calibrate(cal_proba, ["UP"] * 20, coverage=0.9)
    d_conf, _, ab_conf = conf.decide({"UP": 0.95, "DOWN": 0.03, "NEUTRAL": 0.02})
    d_flat, _, ab_flat = conf.decide({"UP": 0.34, "DOWN": 0.33, "NEUTRAL": 0.33})
    assert d_conf == "UP" and ab_conf is False
    assert ab_flat is True and d_flat == "ABSTAIN"
