from datetime import date, datetime, timedelta, timezone

from dataset.build import Row
from eval.model_eval import cv_macro_f1
from eval.report import ablation_gaps, format_report, run_report

UTC = timezone.utc


def _rows(n: int = 40) -> list[Row]:
    out = []
    for i in range(n):
        lbl = ("UP", "DOWN", "NEUTRAL")[i % 3]
        prior = {"UP": 0.05, "DOWN": -0.05, "NEUTRAL": 0.0}[lbl]
        d = date(2024, 1, 1) + timedelta(days=i)  # distinct session per row
        feats = {"topic_XLE": 1.0, "prior_1d_ret": prior, "prior_3d_ret": prior,
                 "backward_vol": 0.01, "spy_trend": 1.0, "weekday": 1.0}
        out.append(Row(str(i), "XLE", datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
                       d, 1.0, feats, {1: prior, 2: prior, 3: prior},
                       {1: lbl, 2: lbl, 3: lbl}))
    return out


def test_cv_produces_out_of_fold_predictions() -> None:
    f1, n, y_true, y_pred = cv_macro_f1(_rows(), horizon=1, market_only=False)
    assert n > 0 and len(y_true) == len(y_pred) == n
    assert 0.0 <= f1 <= 1.0


def test_report_includes_gbt_and_ablation() -> None:
    report, registry = run_report(_rows(), n_perm=50)
    models = {r.model for r in report}
    assert {"gbt_text", "gbt_market_only"} <= models
    assert len(registry) == len(report)
    assert ablation_gaps(report)  # per-horizon text-vs-market gap computed
    assert "Text-ablation" in format_report(report)
