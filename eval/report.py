"""Signal-or-null report (§7, §9) — the actual deliverable / reporting_job.

For each horizon: evaluate a predictor's macro-F1 against the majority baseline
and a permutation null, register the test, BH-correct across the whole
registry, and attach the power-gate verdict. The celebrated headline may be
"nothing survives correction" — a rigorous null result is a full success.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from config.settings import SETTINGS, Settings
from dataset.build import Row
from eval.baselines import majority_class, market_follow, permutation_null
from eval.metrics import CLASSES, macro_f1
from eval.model_eval import cv_out_of_fold
from eval.power import PowerResult, mde_gate
from eval.registry import Registry
from eval.significance import benjamini_hochberg, permutation_pvalue

DEFAULT_MODELS = ("market_follow", "gbt_text", "gbt_market_only")


@dataclass(frozen=True)
class ReportRow:
    horizon: int
    model: str
    n: int
    macro_f1: float
    majority_f1: float
    p_value: float
    q_value: float  # BH-corrected
    significant: bool
    power: PowerResult


def _predictions(
    rows: Sequence[Row], horizon: int, model: str, cfg: Settings
) -> tuple[list[str], list[str]] | None:
    """(y_true, y_pred) for a model at a horizon, or None if not evaluable here."""
    if model == "market_follow":
        usable = [r for r in rows if r.label[horizon] in CLASSES]
        if not usable:
            return None
        y = [r.label[horizon] for r in usable]
        return y, market_follow([r.features["prior_1d_ret"] for r in usable])
    if model in ("gbt_text", "gbt_market_only"):
        y, pred = cv_out_of_fold(rows, horizon, model.endswith("market_only"), cfg)
        return (y, pred) if y else None  # None when CV yields no out-of-fold rows (tiny N)
    raise ValueError(f"unknown model {model}")


def run_report(
    rows: Sequence[Row],
    models: Sequence[str] = DEFAULT_MODELS,
    n_perm: int = 1000,
    cfg: Settings = SETTINGS,
) -> tuple[list[ReportRow], Registry]:
    registry = Registry()
    raw: list[tuple[int, str, int, float, float, float, PowerResult]] = []

    for h in cfg.horizon_days:
        for model in models:
            yp = _predictions(rows, h, model, cfg)
            if yp is None:
                continue
            y, pred = yp
            f1 = macro_f1(y, pred)
            maj = macro_f1(y, [majority_class(y)] * len(y))
            null = permutation_null(y, pred, macro_f1, n=n_perm, seed=cfg.seed)
            p = permutation_pvalue(f1, null)
            priors = [y.count(c) / len(y) for c in CLASSES]
            power = mde_gate(len(y), priors, cfg.alpha)
            registry.register("ALL", h, model, cfg.k)  # BH denominator = registry size
            raw.append((h, model, len(y), f1, maj, p, power))

    qs = benjamini_hochberg([r[5] for r in raw])
    return (
        [
            ReportRow(h, model, n, f1, maj, p, q, q <= cfg.alpha, power)
            for (h, model, n, f1, maj, p, power), q in zip(raw, qs, strict=True)
        ],
        registry,
    )


def ablation_gaps(rows: Sequence[ReportRow]) -> dict[int, float]:
    """Per-horizon (gbt_text macro-F1 - gbt_market_only). ~0 => tweets add nothing (§7)."""
    by = {(r.horizon, r.model): r.macro_f1 for r in rows}
    return {
        h: by[(h, "gbt_text")] - by[(h, "gbt_market_only")]
        for h in {r.horizon for r in rows}
        if (h, "gbt_text") in by and (h, "gbt_market_only") in by
    }


def format_report(rows: Sequence[ReportRow]) -> str:
    lines = [
        f"{'horizon':<8}{'model':<17}{'n':>4}{'macroF1':>9}{'majF1':>8}{'p':>8}"
        f"{'q(BH)':>8}{'signif':>8}{'MDE':>7}  power",
        "-" * 88,
    ]
    for r in sorted(rows, key=lambda r: (r.horizon, r.model)):
        mde = "none" if r.power.mde is None else f"{r.power.mde:.2f}"
        verdict = "powered" if r.power.powered else "UNDERPOWERED"
        lines.append(
            f"ret_{r.horizon}d{'':<3}{r.model:<17}{r.n:>4}{r.macro_f1:>9.3f}"
            f"{r.majority_f1:>8.3f}{r.p_value:>8.3f}{r.q_value:>8.3f}"
            f"{'YES' if r.significant else 'no':>8}{mde:>7}  {verdict}"
        )

    gaps = ablation_gaps(rows)
    if gaps:
        lines.append("")
        lines.append("Text-ablation (gbt_text - gbt_market_only macro-F1; ~0 => tweets add nothing):")
        for h, g in sorted(gaps.items()):
            lines.append(f"  ret_{h}d: {g:+.3f}")

    signif = [r for r in rows if r.significant]
    lines.append("")
    lines.append(
        f"HEADLINE: {len(signif)} of {len(rows)} cells survive BH correction "
        f"(alpha={SETTINGS.alpha}). "
        + ("A null result here is a valid, expected outcome." if not signif else "")
    )
    return "\n".join(lines)
