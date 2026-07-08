# Copilot instructions for the Trump-tweet → stock/sector signal project

You are working in an offline research and inference pipeline for linking Donald J. Trump tweets to stock and sector movement.

Your objective is to produce reliable, reproducible results that follow the repository’s scientific and engineering constraints. Do not optimize for a positive result. Prefer rigorous, well-documented, auditable outputs over flashy predictions.

## Core mission

For each tweet, you must:
1. Collect or ingest high-quality tweet and market data.
2. Explain the tweet in financial context.
3. Classify the tweet’s entities, sentiment, intent, and themes.
4. Link the tweet to one or more relevant tickers and sector ETFs.
5. Measure price movement before and after the tweet.
6. Prepare training data for a decision-tree baseline.
7. Prepare transformer-ready structured data for future modeling.
8. Generate human-readable mock outputs for review.
9. Support inference for new incoming tweets.

All outputs must be structured, consistent, and machine-readable.

## Hard constraints

Follow these rules at all times:
- Preserve point-in-time correctness. Any feature or label derived for a tweet at time $t_0$ must use only data with timestamp strictly less than $t_0$.
- Preserve the single-feature-path rule: the primary decision logic must flow through the repository’s canonical decision function, not through a second ad hoc path.
- Do not introduce leakage by using future bars, future labels, or post-event data in feature generation.
- Keep the pipeline deterministic and reproducible.
- Prefer structured, explainable features over opaque embeddings unless explicitly requested.
- Treat abstention and uncertainty as first-class outputs when appropriate.

## Repository-specific expectations

This repository already contains a modular pipeline with these core concepts:
- Tweet and market data ingestion under the data layer.
- Point-in-time market state construction.
- Rule-based sector mapping.
- Structured feature generation.
- A canonical decision function for batch and serving paths.
- Evaluation, labeling, and optional model components.

When implementing or modifying code:
- Reuse the existing architecture instead of creating a parallel path.
- Keep jobs and serving code thin wrappers around the core logic.
- Avoid adding model logic directly into endpoint or job modules.
- Keep changes testable and aligned with the existing module boundaries.

## Required behavior for tweet processing

For every tweet, produce a normalized record with:
- tweet id
- timestamp
- text
- sentiment
- intent
- entities: companies, executives, sectors, themes
- confidence score
- mapped targets with ticker, sector, industry, and reason for linkage
- expected direction and performance metrics

Use this JSON shape as the minimum schema:

```json
{
  "tweet_id": "",
  "timestamp": "",
  "text": "",
  "sentiment": "",
  "intent": "",
  "entities": {
    "companies": [],
    "executives": [],
    "sectors": [],
    "themes": []
  },
  "targets": [
    {
      "ticker": "",
      "sector": "",
      "industry": "",
      "reason_linked": "",
      "expected_direction": "",
      "performance": {
        "month_before_pct": 0.0,
        "week_after_pct": 0.0,
        "month_after_pct": 0.0
      },
      "verdict_week": ""
    }
  ]
}
```

## Modeling expectations

### Phase 1: decision-tree baseline
Use a simple, interpretable baseline that consumes:
- sentiment
- intent
- sector
- entity type
- historical movement patterns
- volatility
- tweet length
- keyword strength

Target label:
- expected movement direction: UP, DOWN, or FLAT

### Phase 2: transformer-ready preparation
Prepare the dataset so future transformer work can use:
- raw tweet text
- enriched metadata
- sector context
- historical market context
- multi-ticker labels

Do not overcomplicate phase 1. Build a clean, testable baseline first.

## Human-in-the-loop mock outputs

For each processed item, generate mock outputs suitable for human review before training:
- sector impact summary
- ticker impact summary
- verdicts
- confidence values
- explanation for each linkage

These should be structured enough to validate the mapping and labeling logic before real training is attempted.

## Inference endpoint expectations

When a new tweet arrives, the system must:
1. Parse the tweet.
2. Classify sentiment and entities.
3. Map to relevant tickers and sectors.
4. Run the model or decision logic.
5. Return a prediction payload in a strict JSON schema.

Use this output shape:

```json
{
  "tweet_id": "",
  "predictions": [
    {
      "ticker": "",
      "expected_move": "",
      "confidence": 0.0,
      "reasoning": ""
    }
  ]
}
```

## Implementation priorities

When deciding what to build next, prefer the following order:
1. Data collection and normalization.
2. Tweet/entity classification and mapping.
3. Point-in-time market feature construction.
4. Decision-tree baseline training and evaluation.
5. Human-review mock outputs.
6. Inference endpoint support.

## Quality bar

Before considering work complete:
- Verify that point-in-time logic is respected.
- Verify that outputs are structured and consistent.
- Verify that the system can process a tweet end to end.
- Verify that any new code is covered by tests where appropriate.

If a requirement conflicts with the repository’s scientific constraints, preserve the repository’s correctness invariants and document the tradeoff clearly.
