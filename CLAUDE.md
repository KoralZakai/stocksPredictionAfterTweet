CLAUDE.md — Political-Tweet -> Sector-ETF Signal Pipeline (V1, offline, daily bars; Nebius-serverless-deployed)
Drop this at repo root. Claude Code loads it as standing context. Treat the Hard Correctness Invariants and Non-Goals as non-negotiable; they override convenience.
Deployment target: the pipeline runs as **Nebius Serverless AI Jobs** (batch) plus one **Nebius Serverless AI Endpoint** (`/predict`), per the Serverless Builders Challenge. The serverless layer is orchestration + packaging ONLY — it wraps the pure modules below and adds no second feature path (§3.2). See §13.
1. Objective & scientific frame
A rigor exercise in ML-systems design under weak/near-null signal, not a trading system. Success = correctness, labeling rigor, calibrated abstention, and an evaluation that cannot fool itself. Predictive accuracy is NOT a success metric. A rigorous null result is a full success.
Scope of the measurement: with daily OHLCV we measure 1–3 trading-day drift-association between a tweet and its mapped sector ETF. We do NOT claim to measure the causal reaction — that is priced in seconds and is invisible to daily bars. Any finding is an association at the drift horizon, guilty until it survives §4.
Honest prior: signal is weak-to-absent. Build the system to detect that truthfully.
Formal hypotheses (pre-registered):

* H0: Tweets from selected political figures have no measurable directional effect on sector-ETF returns beyond noise, over 1–3 trading days.
* H1: Certain categories of tweets have statistically significant directional impact over 1–3 trading days. (Compound -> multiple-comparison correction, §4.)
2. Non-goals for V1

* Daily OHLCV only. No intraday/tick data, no reaction-window claims.
* No real-time ingestion, streaming, or X-API polling. (Serverless *batch* jobs and a stateless replay endpoint ARE in scope — see §13 — but nothing polls a live feed. The `/predict` endpoint is caller-fed: the request carries the tweet text + `t0`, and features come from the historical price store as-of `t0`. A genuinely live `t0` with no stored bars `< t0` returns a clean abstain, not a guess.)
* No trading simulation, PnL, or transaction-cost modeling.
* No LLM fine-tuning (no LoRA/QLoRA).
* No ML-based sector mapping in V1 (rule-based + human-verified only; §6).
* Do not tune toward a positive result.
3. Hard correctness invariants (this IS the spec — enforce with CI tests)

1. Point-in-time. Every feature for a tweet at `t0` uses ONLY data with timestamp `< t0`. A test scans feature inputs and fails on any input `>= t0`.
2. No train/serve skew. Exactly one pure function `decide(tweet, market_state_as_of(t0)) -> Decision`, used by the offline batch jobs (§13), the `/predict` serverless endpoint (§13), and (later) the replay harness — ALL THREE. No second feature path; `jobs/` and `serving/` are thin wrappers that only marshal I/O and call this function. A test runs a sample through the batch path and through a single-event/endpoint call and asserts identical features.
3. Entry anchor (pre-registered, config). `s0` = first regular session whose open is strictly after `t0` (off-hours / weekend / holiday tweets resolve cleanly to the next session — no leakage, no NaN needed).
   * `entry` = `open(s0)` — leak-free, primary.
   * Optional diagnostic variant `close_entry` = first session close after t0 (for a pre-close tweet, the same-day close, which already embeds the day's move). OFF by default; used only to quantify same-day contamination (§7).
4. Drift horizons. Cumulative return from `entry` to session close:
   * `ret_1d` = close(s0) / entry - 1
   * `ret_2d` = close(s0+1) / entry - 1
   * `ret_3d` = close(s0+2) / entry - 1 A test asserts every horizon's close timestamp `> t0`.
5. Labels UP/DOWN/NEUTRAL via a volatility-scaled band: thresholds = `±k · σ_backward`, where `σ_backward` is the sector's backward-only rolling daily return vol and `k` is pre-registered (start `k=0.5`). Report the resulting class balance per sector/horizon as a first-class diagnostic — a degenerate split (all NEUTRAL, or no NEUTRAL) is a labeling failure, not a model problem. No full-series statistics anywhere.
6. Non-i.i.d. rows. Overlapping `ret_2d/3d` windows correlate rows across nearby tweets. Evaluation uses purged + embargoed walk-forward CV, embargo `>= 3 sessions`, purge by tweet id. No random KFold anywhere.
3b. Failure-mode detectors (each is a concrete test/diagnostic)

* Label leakage — assert no label input predates entry; thresholds backward-only.
* Temporal leakage — the point-in-time scanner (1) + a "future-shuffled features" canary that must destroy measured performance.
* Regime-shift sensitivity — evaluate split by pre/post-election and crisis windows; report metric stability.
* Overfit-to-noise — the transformer canary (§8) + permutation null (§4).
* Spurious macro correlation — the text-ablation (§7) is the primary guard.
* Sector misclassification — human-verified mapping on MVP-10; log confidence.
4. Statistical design

* Power / MDE gate (run BEFORE modeling). Per horizon: given N usable rows, 3 classes, corrected α, compute the minimum detectable effect at 80% power (via simulation against the permutation null). Emit a verdict: powered to detect a plausible effect? If MDE exceeds any plausible effect, declare underpowered by construction and record it as a primary finding.
* Baselines are first-class: always-majority, market-follow, and a permutation null (shuffle tweet timestamps / block-bootstrap) giving each metric's null.
* Significance: effect size AND permutation p-value, Benjamini-Hochberg corrected over the full test registry (§5: 3 horizons × sectors × models). Nothing is reported significant unless registered before running.
* Headline output: does anything survive correction? The correct, celebrated answer may be "no."
5. Architecture (modular, deterministic, seam-ready)

```
config/            # pydantic; ALL pre-registered choices: sectors, horizons, k,
                   #   entry anchor, decision rule, alpha
data/
  sources/         # TweetSource + PriceSource(daily) abstract interfaces;
                   #   LOCAL parquet/CSV adapters only (no paid API hardcoded)
  storage/         # DuckDB + parquet, schema-versioned; fail fast on tz-naive
                   #   timestamps or duplicate (author, timestamp)
  audit/           # datasheet + bias + timestamp-precision diagnostics (§9)
core/
  calendar.py      # trading-day calendar: sessions, holidays, half-days; resolves
                   #   s0 = first open strictly after t0
  market_state.py  # market_state_as_of(t0): point-in-time view; asof joins ONLY
                   #   (pandas merge_asof backward / DuckDB ASOF JOIN)
  features.py      # STRUCTURED features by default: topic one-hots, entity flags,
                   #   prior 1d/3d return, backward-only vol, SPY-trend regime,
                   #   weekday. Raw embeddings ONLY behind a flag.
  decide.py        # the single pure decide(...) used everywhere
sector_mapping/
  rules.py         # rule-based keyword + entity -> sector(s) (§6)
  weights.py       # probabilistic multi-sector weights + ambiguity handling
labeling/
  windows.py       # entry anchor + ret_1d/2d/3d (strict t>t0)
  thresholds.py    # vol-scaled band (±k·σ_backward) + class-balance report
eval/
  splits.py        # purged + embargoed walk-forward, purge by tweet id
  baselines.py     # majority, market-follow, PERMUTATION NULL
  ablation.py      # §7 — text-ablation is PRIMARY
  metrics.py       # macro-F1, precision/recall, calibration, per-sector, per-horizon
  significance.py  # block-bootstrap + permutation p-values; Benjamini-Hochberg
  power.py         # §4 MDE / power gate, per horizon
  registry.py      # every (sector,horizon,model,threshold) tested is logged;
                   #   BH denominator = registry size; no unregistered significance
models/
  gbt.py           # gradient-boosted trees on STRUCTURED features (primary)
  transformer.py   # small text classifier — OVERFIT CANARY (§8)
  abstain.py       # conformal prediction (class-conditional/Mondrian), NOT softmax>k
interpret/
  shap_gbt.py      # SHAP for GBT — SUGGESTIVE only (high-variance at small N)
  attention.py     # illustrative only; attention is NOT a faithful explanation
reports/           # signal-or-null table: effect size + BH-corrected p, per cell
jobs/              # Nebius Serverless AI Jobs — thin CLI entrypoints, NO feature
                   #   logic (§13). Each reads/writes DuckDB+parquet artifacts,
                   #   is deterministic, and stamps outputs with the run id.
  data_ingestion.py        #  -> data/sources + data/storage
  labeling.py              #  -> core/calendar + market_state + labeling/*
  feature_engineering.py   #  -> sector_mapping/* + core/features
  dataset_build.py         #  -> assemble modeling frame (features+labels+weights)
  training.py              #  -> models/gbt (+ transformer canary, optional)
  evaluation.py            #  -> eval/* (splits, baselines, significance, power)
  reporting.py             #  -> reports/ + interpret/  (signal-or-null report)
serving/
  endpoint.py      # Nebius Serverless AI Endpoint: POST /predict -> calls the
                   #   SAME decide() + models/abstain. No live polling (§2).
deploy/            # Dockerfile, Nebius job/endpoint manifests, run scripts
tests/

```

6. Sector mapping (V1 = rule-based + human-verified)

* A tweet maps to: one primary sector, multiple sectors (spillover), or NONE.
* Deterministic keyword + entity rules. No ML mapper in V1 — an unvalidated classifier in the causal chain makes "no signal" indistinguishable from "bad mapping."
* Ambiguity: probabilistic weights; never force single-sector under high uncertainty.
* Eval math (pre-registered):
   * Primary: argmax single sector -> one row per tweet -> clean purging.
   * Secondary: weighted multi-sector via sample weights, folds purged by tweet id.
* Log per-mapping confidence; surface low-confidence mappings to the §9 audit.
7. Evaluation protocol (the actual deliverable)

* Pre-register everything in `config/` before running.
* For each registered `(sector, horizon, model)`: metric vs (a) majority, (b) market-follow, (c) permutation null.
* Ablations — text-ablation is PRIMARY, not a side check:
   * `model_with_text` vs `model_market_only` — the central scientific test. If market-only matches full, apparent performance is macro, not tweets -> claim dead.
   * remove market features -> measure impact.
   * shuffle tweet timestamps -> must collapse to null.
   * random sector assignment -> null benchmark.
* If `close_entry` diagnostic is enabled: the `ret_1d` gap between `open` and `close` entry measures same-day contamination.
* Report per horizon (`ret_1d/2d/3d`); effect size + BH-corrected permutation p.
8. Modeling notes

* Primary = GBT on structured features. At ~150–300 rows this is the honest ceiling. Do NOT feed raw 384/768-dim embeddings into GBT by default.
* Transformer = deliberate overfit canary. Its job is adversarial: confirm that purged CV + permutation null + BH correction correctly REFUSE to reward it. If the eval crowns the transformer, the eval is broken — not the model.
* Abstention = conformal prediction with pre-registered target coverage. Report abstention rate and empirical coverage. Abstain is a first-class output.
9. Data realism, bias audit & datasheet (`data/audit/`)
Dataset is NOT assumed representative of a full causal universe. Datasheet documents and, where possible, diagnoses:

* Tweet-timestamp precision — need `t0` with correct timezone and enough resolution to place it relative to the session (before open / intraday / after close), so `s0` is unambiguous. Assert it.
* Selection bias from incomplete X/API coverage.
* Non-uniform temporal coverage — plot tweets/time; flag gaps.
* Temporal clustering (elections, crises) — detect/quantify; ties to §3.6 overlap and the regime-shift evaluation.
* Macro confounding — a threat to validity you document, not something N≈1000 observational data can deconfound. No causal identification is claimed.
* Survivorship bias — deleted tweets are gone and may be systematically the moved-then-retracted ones. State explicitly.
* Retweets / edits / deletions — normalize and flag; exclude where they distort t0.
Mostly documented and tested-for, not fixed. Say so plainly.
10. Phased roadmap (build Phase 0 -> gate -> Phase 1)

* Phase 0 — MVP (10 tweets). Plumbing correctness ONLY. Per tweet: text + timestamp, rule-based sector(s), resolve `s0`, compute `ret_1d/2d/3d`, vol-scaled labels, manually inspect alignment. Do NOT inspect outcomes for signal — 10 outcomes carry zero evidential weight. NOT for training.
* Decision gate. Proceed ONLY if: point-in-time & no-skew tests pass, `s0` resolution correct across before-open/intraday/after-close/weekend cases, labeling coherent with sane class balance, sector mapping stable. Gate on correctness invariants, never on "results look promising."
* Phase 1 (main). Full dataset + power gate + features + eval harness + GBT + conformal abstain + ablations + report -> ship the signal-or-null table. Package each stage as a Nebius Serverless AI Job (§13) and stand up the `/predict` endpoint wrapping `decide()` + conformal abstain. The endpoint ships regardless of whether signal is found — a well-calibrated model that mostly *abstains* on a null signal is a correct, submittable deliverable.
* Phase 2 (only if Phase 1 shows stable, BH-corrected signal). Replay harness feeding historical events through the SAME `decide()` in timestamp order.
* Phase 3. Streaming infra — not designed until Phase 2 validates.
11. Build order (separate reviewable commits)

1. config + schemas + storage + source interfaces w/ local adapters + tests.
2. `calendar.py` (sessions, holidays, half-days, `s0` resolution) + tests.
3. `market_state_as_of` + asof-join test + point-in-time scanner test.
4. labeling (entry anchor, `ret_1d/2d/3d`, vol-band thresholds, class-balance report) + `close > t0` test.
5. features + `decide()` + no-skew equivalence test.
6. sector_mapping rules + human-verified MVP-10 harness (Phase 0 gate).
7. splits + baselines + permutation null + registry + significance + BH + power gate.
8. GBT + conformal abstain; then transformer overfit canary; then ablations.
9. interpretability + report generator -> signal-or-null table.
10. serverless packaging: wrap steps 1–9 as `jobs/*` entrypoints, add `deploy/Dockerfile` + Nebius manifests, then `serving/endpoint.py` (`/predict`) over the same `decide()`. Do NOT start this until the no-skew test (step 5) is green.
Start with step 1. Before implementing collectors, show me the two source interfaces and the design of the point-in-time + no-skew tests. Ask before adding any dependency beyond the standard scientific stack.
12. Coding standards

* Python 3.11+, `uv`, `ruff` + `mypy --strict`, `pytest`; CI runs the invariant tests (point-in-time, no-skew, close>t0, asof-only).
* pandas or polars (pick one, justify in README); DuckDB for asof joins.
* All timestamps tz-aware UTC internally; `calendar.py` owns market-local session logic. Fully deterministic: seed everything, snapshot + hash datasets, stamp every reported number with a reproducible run id.
* Every module has tests; the correctness-invariant tests are mandatory in CI.
* One `deploy/Dockerfile` (CPU-only; classical-ML track) builds the single image both the jobs and the endpoint run from — same code, same deps, so batch and serve cannot drift. Pin deps via `uv.lock`.

13. Serverless deployment (Nebius) — the challenge substrate
The pipeline is deployed on Nebius Serverless AI: seven batch **Jobs** run the offline research pipeline, one **Endpoint** exposes `/predict`. The rule that makes this rigorous and not just plumbing: **the serverless layer contains zero science.** Every job is a thin CLI that reads input artifacts, calls the pure modules in §5, and writes output artifacts. The endpoint is a thin handler that calls `decide()`. The no-skew test (§3.2) is what guarantees this.

* Job DAG (each a separate Nebius AI Job, chained by artifact I/O in a shared DuckDB+parquet store; CPU-only):
   1. `data_ingestion` — load tweets + daily OHLCV (XLK,XLE,XLF,XLI,XLV,XLP,XLY,XLB,SMH,ITA) via the §5 local adapters into schema-versioned storage. Fail fast on tz-naive timestamps / duplicate (author, timestamp).
   2. `labeling` — resolve `s0`, compute `ret_1d/2d/3d`, apply the vol-scaled band, emit the class-balance diagnostic (§3.4–5).
   3. `feature_engineering` — sector mapping (§6) + structured features (§5 `features.py`); embeddings only behind the flag.
   4. `dataset_build` — assemble the final modeling frame (features + labels + purge keys + sample weights). Hash + run-id stamp it.
   5. `training` — fit GBT primary (+ transformer canary, optional); persist model + conformal calibrator.
   6. `evaluation` — purged/embargoed walk-forward CV, baselines, permutation null, power gate, BH correction over the registry (§4).
   7. `reporting` — render the signal-or-null table (effect size + BH-corrected p per cell) + interpretability. This is THE deliverable.
* `/predict` Endpoint (`serving/endpoint.py`): request = `{tweet_text, timestamp}`. Handler builds `market_state_as_of(t0)` from the stored bars (point-in-time, §3.1), calls `decide()`, returns `{sector, direction ∈ UP/DOWN/NEUTRAL, confidence, abstain}`. Abstention is conformal (§8), not softmax-thresholded, and is mandatory below the calibrated coverage target. If no bars exist `< t0` (a live/future `t0`), abstain — do not fabricate features.
* Reproducibility (a judged criterion): README documents hardware config (CPU), expected outputs, approximate runtime + cost per job. Every run is deterministic and run-id stamped. Proof-of-execution (job logs, endpoint URL) captured for submission.

14. Challenge deliverables (Nebius Serverless Builders Challenge, due 2026-07-15)

* Public repo: modular `jobs/` + `serving/` + `deploy/Dockerfile` + Nebius manifests, reproducible per README, OSI license (MIT), no committed secrets/private data (public datasets only).
* README: setup, hardware config, expected outputs, approximate runtime/cost.
* Technical blog post (≥600 words, tagged `#NebiusServerlessChallenge`): problem, architecture, evaluation methodology, and **why a null result is a valid, expected outcome** — the honest framing IS the story.
* Optional: 3–10 min video walkthrough; `/predict` demo.
