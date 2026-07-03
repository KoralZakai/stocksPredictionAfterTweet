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
from eval.power import PowerResult, mde_gate
from eval.registry import Registry
from eval.significance import benjamini_hochberg, permutation_pvalue


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


def run_report(
    rows: Sequence[Row],
    model: str = "market_follow",
    n_perm: int = 1000,
    cfg: Settings = SETTINGS,
) -> tuple[list[ReportRow], Registry]:
    registry = Registry()
    raw: list[tuple[int, int, float, float, float, PowerResult]] = []

    for h in cfg.horizon_days:
        usable = [r for r in rows if r.label[h] in CLASSES]
        if not usable:
            continue
        y = [r.label[h] for r in usable]
        pred = market_follow([r.features["prior_1d_ret"] for r in usable])

        f1 = macro_f1(y, pred)
        maj = macro_f1(y, [majority_class(y)] * len(y))
        null = permutation_null(y, pred, macro_f1, n=n_perm, seed=cfg.seed)
        p = permutation_pvalue(f1, null)

        priors = [y.count(c) / len(y) for c in CLASSES]
        power = mde_gate(len(y), priors, cfg.alpha)

        registry.register("ALL", h, model, cfg.k)  # BH denominator = registry size
        raw.append((h, len(y), f1, maj, p, power))

    qs = benjamini_hochberg([r[4] for r in raw])
    return (
        [
            ReportRow(h, model, n, f1, maj, p, q, q <= cfg.alpha, power)
            for (h, n, f1, maj, p, power), q in zip(raw, qs, strict=True)
        ],
        registry,
    )


def format_report(rows: Sequence[ReportRow]) -> str:
    lines = [
        f"{'horizon':<8}{'n':>4}{'macroF1':>9}{'majF1':>8}{'p':>8}{'q(BH)':>8}"
        f"{'signif':>8}{'MDE':>8}  power",
        "-" * 72,
    ]
    for r in rows:
        mde = "none" if r.power.mde is None else f"{r.power.mde:.2f}"
        verdict = "powered" if r.power.powered else "UNDERPOWERED"
        lines.append(
            f"ret_{r.horizon}d{'':<3}{r.n:>4}{r.macro_f1:>9.3f}{r.majority_f1:>8.3f}"
            f"{r.p_value:>8.3f}{r.q_value:>8.3f}{'YES' if r.significant else 'no':>8}"
            f"{mde:>8}  {verdict}"
        )
    signif = [r for r in rows if r.significant]
    lines.append("")
    lines.append(
        f"HEADLINE: {len(signif)} of {len(rows)} cells survive BH correction "
        f"(alpha={SETTINGS.alpha}). "
        + ("A null result here is a valid, expected outcome." if not signif else "")
    )
    return "\n".join(lines)
