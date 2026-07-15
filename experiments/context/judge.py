"""LLM-as-a-Judge — OFFLINE post-hoc reasoning-quality diagnostic. EXPERIMENTAL.

Runs strictly on the already-produced val outputs (results_b.json). It does NOT
touch inference, the prompt hash, or the shipped path. It scores the QUALITY of
each system's written rationale on a 1-5 rubric — a measure of PLAUSIBILITY, not
predictive correctness.

KNOWN CONFOUND (stated loudly, and repeated in the report): System B was handed
the macro context and System A was not, so B's rationale will mention macro facts
and score higher on 'macro_alignment' BY CONSTRUCTION. A higher judge score is
therefore evidence of more macro-flavoured prose, NOT of a better prediction.
Judge scores never override the beat-SPY numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from alpha.env import env, load_dotenv

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_b.json"
OUT = HERE / "judge_scores.json"

_RUBRIC = (
    "You are a strict evaluation judge. Score a trading thesis's REASONING QUALITY "
    "(not whether it turned out right) on three 1-5 rubrics:\n"
    "  macro_alignment: did it properly weigh the given macro context?\n"
    "  causal_logic:    is the tweet->asset connection logical?\n"
    "  risk_awareness:  does it acknowledge uncertainty / downside?\n"
    'Return ONLY JSON: {"macro_alignment":n,"causal_logic":n,"risk_awareness":n,'
    '"critique":"one sentence"}. n are integers 1-5.'
)


def _judge_one(rationale: str, macro_context: str, *, base_url: str, api_key: str,
               model: str) -> dict[str, Any]:
    ctx = macro_context or "(none provided to this system)"
    user = f"MACRO CONTEXT:\n{ctx}\n\nTHESIS/REASONING:\n{rationale or '(empty)'}"
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "system", "content": _RUBRIC},
                           {"role": "user", "content": user}]},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Nebius {resp.status_code}: {resp.text[:200]}")
    c = resp.json()["choices"][0]["message"]["content"].strip()
    if c.startswith("```"):
        c = c.split("```", 2)[1].lstrip("json").strip() if "```" in c[3:] else c.strip("`")
    parsed: dict[str, Any] = json.loads(c[c.find("{"):c.rfind("}") + 1])
    return parsed


def _mean(scores: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(s[key]) for s in scores if isinstance(s.get(key), (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline LLM-judge on val rationales (A vs B)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL",
                                           default="meta-llama/Llama-3.3-70B-Instruct"))
    a = ap.parse_args()
    load_dotenv()
    api_key = env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    base = env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL",
               default="https://api.studio.nebius.ai/v1")
    if not api_key:
        raise SystemExit("No NEBIUS_API_KEY (.env).")
    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} not found — run run_context.py first.")

    rows = json.loads(RESULTS.read_text())
    if a.limit:
        rows = rows[:a.limit]
    a_scores: list[dict[str, Any]] = []
    b_scores: list[dict[str, Any]] = []
    for i, r in enumerate(rows, 1):
        try:
            js_a = _judge_one(r.get("a_rationale", ""), "", base_url=base, api_key=api_key, model=a.model)
            js_b = _judge_one(r.get("b_rationale", ""), r.get("macro_context", ""),
                              base_url=base, api_key=api_key, model=a.model)
        except Exception as exc:
            print(f"  [{i}/{len(rows)}] judge failed: {exc}")
            continue
        a_scores.append(js_a)
        b_scores.append(js_b)
        print(f"  [{i}/{len(rows)}] A={js_a.get('causal_logic')} B={js_b.get('causal_logic')} (causal)")

    keys = ["macro_alignment", "causal_logic", "risk_awareness"]
    summary = {
        "note": ("Reasoning-quality diagnostic only. 'macro_alignment' is biased toward B "
                 "by construction (B was given the macro context, A was not). Does NOT "
                 "measure predictive correctness — see beat-SPY numbers for that."),
        "system_a_avg": {k: _mean(a_scores, k) for k in keys},
        "system_b_avg": {k: _mean(b_scores, k) for k in keys},
        "n": len(a_scores),
    }
    OUT.write_text(json.dumps({"summary": summary, "a": a_scores, "b": b_scores}, indent=2),
                   encoding="utf-8")
    print(f"\n[judge] A avg: {summary['system_a_avg']}")
    print(f"[judge] B avg: {summary['system_b_avg']}")
    print(f"[judge] -> {OUT.name}  (qualitative only; macro_alignment confounded)")


if __name__ == "__main__":
    main()
