"""DSPy prompt-optimization experiment. EXPERIMENTAL — see experiments/dspy/README.

Guardrails (also enforced here): the optimizer sees the TRAIN SPLIT ONLY; val is
iteration feedback; test is scored once, last, via --final. The metric reuses the
shipped engine (`alpha.benchmark.relative_hit`, beat-SPY). Nothing here edits the
frozen `alpha/classify.py` prompt or the validation manifest.

`dspy` is imported inside main() so this module (and the --selfcheck split-safety
check) run on a box without dspy installed.

Run:
    python experiments/dspy/train_dspy.py --selfcheck                      # no LLM
    python experiments/dspy/train_dspy.py --run-name lfs-baseline-iter-01  # spends Nebius
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from alpha.benchmark import relative_hit
from alpha.env import env, load_dotenv
from scripts.nebius_macro_backtest import RESULTS, _assign_splits

ROOT = Path(__file__).resolve().parents[2]
RESULTS_TSV = Path(__file__).resolve().parent / "results.tsv"
_TSV_COLS = ["run_name", "optimizer", "model", "n_train", "n_val", "n_test",
             "train_hit", "val_hit", "test_hit", "compile_s", "note"]


def _load_split(split: str) -> list[dict[str, Any]]:
    """Rows for one split, with cached EOD returns kept for the metric (no refetch)."""
    data: list[dict[str, Any]] = json.loads((ROOT / RESULTS).read_text())
    _assign_splits(data)                      # same chronological 60/20/20 as the shipped run
    rows = [r for r in data if r.get("split") == split]
    assert all(r.get("split") == split for r in rows), "split leak"
    return rows


def _cached_eod(row: dict[str, Any]) -> dict[str, float]:
    """{ticker: EOD return} + SPY, from the cached backtest — free, no market call."""
    out: dict[str, float] = {}
    spy = (row.get("spy_returns") or {}).get("EOD")
    for ins in row.get("instruments", []):
        rets = (ins.get("returns") or {})
        if "EOD" in rets:
            out[str(ins["ticker"]).upper()] = rets["EOD"]
    if spy is not None:
        out["SPY"] = spy
    return out


def beat_spy_hit(pred_instruments: list[dict[str, Any]], cached_eod: dict[str, float]) -> float | None:
    """Tweet-level beat-SPY hit at EOD (majority vote of scoreable instruments).

    THE METRIC BEING OPTIMIZED — and, by design, the meta-model trap: maximizing
    this on train is exactly what failed the sacred test. Returns None if unscoreable.
    """
    spy = cached_eod.get("SPY")
    if spy is None:
        return None
    hits: list[bool] = []
    for ins in pred_instruments:
        tk = str(ins.get("ticker", "")).upper()
        if tk == "SPY" or tk not in cached_eod:
            continue                          # ponytail: only score tickers we have cached returns for
        h = relative_hit(str(ins.get("predicted_direction", "")).lower(), cached_eod[tk], spy)
        if h is not None:
            hits.append(h)
    if not hits:
        return None
    return float(sum(hits) / len(hits) >= 0.5)


def _mean_hit(rows: list[dict[str, Any]], predict: Any) -> float:
    """Mean tweet-level beat-SPY hit over rows, using `predict(text) -> instruments`."""
    scored = []
    for r in rows:
        pred = predict(r.get("text", ""))
        h = beat_spy_hit(pred, _cached_eod(r))
        if h is not None:
            scored.append(h)
    return sum(scored) / len(scored) if scored else 0.0


def _append_tsv(record: dict[str, Any]) -> None:
    new = not RESULTS_TSV.exists()
    with RESULTS_TSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_TSV_COLS, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in _TSV_COLS})


def main() -> None:
    ap = argparse.ArgumentParser(description="DSPy prompt-optimization experiment (train-only)")
    ap.add_argument("--run-name", default="lfs-baseline-iter-01")
    ap.add_argument("--optimizer", default="labeled_fewshot",
                    choices=["labeled_fewshot", "bootstrap_fewshot"])
    ap.add_argument("--model", default=env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL",
                                           default="meta-llama/Llama-3.3-70B-Instruct"))
    ap.add_argument("--final", action="store_true",
                    help="ALSO score the sacred TEST split (do this once, at the very end)")
    a = ap.parse_args()
    load_dotenv()

    import dspy   # heavy import: only needed for a real run

    api_key = env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    base = env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL",
               default="https://api.studio.nebius.ai/v1")
    if not api_key:
        raise SystemExit("No NEBIUS_API_KEY (.env). Needed for a real run; use --selfcheck otherwise.")
    dspy.configure(lm=dspy.LM(f"openai/{a.model}", api_base=base, api_key=api_key, temperature=0.0))

    class TweetToInstruments(dspy.Signature):
        """Read a political social-media post and name the US-listed instruments its
        CONTENT (not any market data) should move, with a direction. JSON only."""
        tweet_text: str = dspy.InputField()
        instruments_json: str = dspy.OutputField(
            desc='JSON list of {"ticker","predicted_direction"} with direction in {up,down}')

    program = dspy.Predict(TweetToInstruments)

    train, val = _load_split("train"), _load_split("val")
    trainset = [dspy.Example(tweet_text=r.get("text", ""),
                             instruments_json="").with_inputs("tweet_text") for r in train]

    def metric(example: Any, pred: Any, trace: Any = None) -> float:
        row = next((r for r in train if r.get("text", "") == example.tweet_text), None)
        if row is None:
            return 0.0
        try:
            instruments = json.loads(pred.instruments_json)
        except Exception:
            return 0.0
        h = beat_spy_hit(instruments if isinstance(instruments, list) else [], _cached_eod(row))
        return float(h or 0.0)

    t0 = time.time()
    if a.optimizer == "labeled_fewshot":
        compiled = dspy.LabeledFewShot(k=8).compile(program, trainset=trainset)
    else:
        compiled = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=8).compile(
            program, trainset=trainset)
    compile_s = round(time.time() - t0, 1)

    def compiled_predict(text: str) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(compiled(tweet_text=text).instruments_json)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    record: dict[str, Any] = {
        "run_name": a.run_name, "optimizer": a.optimizer, "model": a.model,
        "n_train": len(train), "n_val": len(val), "n_test": 0,
        "train_hit": round(_mean_hit(train, compiled_predict), 4),
        "val_hit": round(_mean_hit(val, compiled_predict), 4),
        "compile_s": compile_s,
        "note": "train-only optimize; val=feedback",
    }
    if a.final:
        test = _load_split("test")          # THE SACRED SPLIT — scored once, here.
        record["n_test"] = len(test)
        record["test_hit"] = round(_mean_hit(test, compiled_predict), 4)
        record["note"] = "FINAL: sacred test scored once"
    _append_tsv(record)
    print(f"[dspy] {a.run_name}: train={record['train_hit']} val={record['val_hit']} "
          f"test={record.get('test_hit', 'n/a')} -> {RESULTS_TSV.name}")
    print("[dspy] REMINDER: better val does NOT mean better test. Expect the meta-model null.")


def _demo() -> None:
    """CPU self-check: split-safety + metric shape, no LLM."""
    train = _load_split("train")
    assert all(r["split"] == "train" for r in train)
    assert len(train) == 266, f"expected 266 train rows, got {len(train)}"
    sample = train[0]
    cached = _cached_eod(sample)
    h = beat_spy_hit([{"ticker": t, "predicted_direction": "up"} for t in cached if t != "SPY"], cached)
    assert h in (0.0, 1.0, None)
    print(f"train_dspy self-check OK (train={len(train)}, metric sample={h})")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _demo()
    else:
        main()
