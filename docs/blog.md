# I Built a Machine to Prove Trump's Tweets Move Markets. It Kept Proving Me Wrong — Seven Times.

*#NebiusServerlessChallenge — code: https://github.com/KoralZakai/stocksPredictionAfterTweet*

Everyone on a trading desk knows the story. *He tweeted about Intel, and the stock ran for
months.* *He posts about Iran, oil spikes.* The anecdotes are vivid, specific, and everybody
has one.

I spent this project trying to measure that. The answer is **no** — political tweets carry
no measurable short-term directional signal for the assets they name. That result isn't
interesting on its own; a coin flip is a coin flip. What's interesting is **how many times
my own pipeline nearly told me otherwise**, and what it took to catch it.

Seven times I built something that produced a beautiful, publishable, *completely false*
finding. Every one of them looked exactly like the discovery the anecdotes promise. This
post is about those seven, because they are the actual engineering content.

---

## The setup

**Data.** 78,130 public posts; 8,317 in the study window (2025-01-01 → 2026-07-06). Daily
OHLCV bars for 62 tickers, committed to the repo. Public data only — a stranger can re-run
every number without a single private key.

**The model.** `meta-llama/Llama-3.3-70B-Instruct` on **Nebius AI Studio**, zero-shot,
reading *only the tweet text*: what is this post about, which instruments does it touch, and
which way does each one go. 476 tweets, 1,296 instrument calls. No fine-tuning — the question
is whether the *text* carries signal, and a fine-tune would smuggle the outcome into the
answer.

**The architecture.** Nebius Serverless AI Jobs run the batch research pipeline; a Serverless
AI Endpoint serves `/predict`. One CPU-only image builds both, so batch and serve cannot
drift. The rule that makes this rigorous rather than plumbing: **the serverless layer
contains zero science.** Every job is a thin CLI that calls pure modules and writes
artifacts. The endpoint is a thin handler that calls the same `decide()`.

```
Jobs:      data_ingestion → labeling → features → dataset → training → evaluation → reporting
Endpoint:  POST /predict ──── the same decide() ────┘
Feedback:  /predict → prediction_log.jsonl → jobs/feedback.py → replication report
```

The endpoint **refuses to boot** if its live classification prompt hash or its corpus
sha256 don't match the manifest the batch job produced. The Job produces the numbers; the
Endpoint may only cite them.

---

## The seven artifacts

### 1. The tie-break that counted ties as wins

`_tweet_hit` aggregates a tweet's instrument basket into one verdict. The tie rule:

```python
return int(sum(hs) / len(hs) >= 0.5)   # 1-of-2 hits → 0.5 >= 0.5 → a WIN
```

**49% of baskets hold exactly two instruments**, where "majority" is undefined. Every 50/50
split scored as a full win. Headline hit rate: **0.618, p=0.017**. I had a finding.

A tie carries no directional verdict — it's unscoreable, not a win. Excluding ties: **0.507**.
Scoring them as losses is the mirror bug (0.393). Two independent unbiased estimators agree
at ~50%. The entire "signal" was the tie rule.

### 2. The permutation null that couldn't say no

The event study flagged **22 of 72 cells** as significant. Gorgeous. Also arithmetic: he
posts on **83% of all trading sessions**, so sampling a null pool of size ≈ the population
*without replacement* collapsed the null's variance and floored every p-value. Bootstrap with
replacement + dedupe to unique event-days: **22 → 0**.

### 3. The same-bar burst

The intraday study returned **9 solid cells, p≈0.0018** — "war tweets spike oil volume 79%".
He posts in bursts: five tweets in ten minutes. With one price bar per hour, all five pointed
at the *same* move. One event counted five times as five pieces of evidence.

I only caught it because the "biggest movers" list had a tweet about a **renovated Palm Room**
at the top. Count each hour once: **9 → 0**.

### 4. A hardcoded market clock

`13.5` for the US open. NYSE opens 13:30 UTC in EDT but **14:30 UTC in EST** — so every tweet
posted 13:30–14:30 UTC from November to mid-March was anchored to the wrong session.

### 5. `DJT` — his own signature

Trump signs posts "President DJT". The ticker rule read that as a mention of his listed
company: **754 of 762 "DJT mentions" were the sign-off.**

### 6. Negated-return math

`-E(-X)` to get a backward return. Valid for small moves; on a stock that had *doubled* it
reported +44.8% for a +105.3% run-up.

### 7. "intel" is not Intel

The one that stings. In a political corpus "intel" overwhelmingly means *intelligence*:
**51 of 55 matches (93%)** were spy-agency tweets — `leaking intel`, `Danish intel`, `Obama
Intel Chief` — and they were sitting in the CORPORATE cohort of a registered study.

But the **mirror half** is the lesson. My dashboard kept its *own* private Intel regex that
only accepted company context *after* the word. So it scored these as **not Intel**:

- `The CEO of INTEL is highly CONFLICTED and must resign`
- `I met with Mr. Lip-Bu Tan, of Intel`
- `I PAID ZERO FOR INTEL, IT IS WORTH APPROXIMATELY 11 BILLION DOLLARS`

Those are exactly the mentions that **contradict my own headline exhibit**. Intel was
*falling* (−14.3%) when he demanded the CEO resign — then rose **+18.1%**. My exhibit claimed
"he reflects news rather than generating it," and my regex had quietly deleted four
counterexamples out of eight.

One rule dropped 93% true positives. The other dropped the inconvenient cases. **A rule that
keeps what fits the thesis is the anecdote generator I was writing about — running on my own
exhibit.** Fixed, rescored: CORPORATE 43 → 13 events, 72 → 63 cells, verdict **unchanged: 0
survive**.

---

## The results

Every number below is explorable in the interactive dashboard —
[**Myth-Busting Quantitative Terminal**](https://koralzakai.github.io/stocksPredictionAfterTweet/reports/dashboard.html)
— including the wrong-day control and the per-tweet reasoning behind each call.

| test | result |
|---|---|
| EOD directional (tweet-level) | **0.522**, n=67, p=0.40 → `shipped_horizons: []` |
| Beat-SPY (instrument-level, 1,296 calls) | **48.1% / 47.3% / 48.6% / 49.2%** at EOD/3d/1w/1mo |
| Event study, 63 pre-registered cells | **0 survive** BH (min p_bh 0.76) |
| Intraday shock, 30 cells, MDE 0.1–0.3% | **0 survive** |
| Meta-model ("when can we trust it?") | Val AUC 0.593 → **Test 0.431** |

Two tells explain the coin flip. **73% of the model's calls are "UP"** — it isn't reaching a
direction, it mostly says up, and markets drift up anyway. And its most-named instrument is
**SPY itself** (304 calls), which cannot beat itself.

Then the control that settles it. The model's named assets *do* move more than the ones it
ignores (1.52% vs 1.12% at one day). So I scored each tweet's picks **on a different tweet's
day**: **+0.40pp real vs +0.38pp wrong-day**. The entire advantage survives on a day the
tweet had nothing to do with. It isn't knowledge — it's that oil and the fear gauge are
jumpier than staples and bonds *every day of the year*.

**Why the anecdotes feel true:** reverse causality (Intel had already doubled), mean reversion
(big oil moves recover the same with no tweet), and saturation (he posts on 83% of sessions,
so every big move has a coincident tweet to blame).

## Closing the loop, honestly

`/predict` appends every served call to a bucket-mounted log; a Job waits for each horizon to
**close**, labels it, and scores it against the same pre-registered registry.

The obvious version of this — retrain on live calls, ship when p < 0.05 — is the fastest way
to destroy everything above. Callers choose which tweets to send, so that sample is
*selected*, not sampled; and re-scoring a growing log until it passes reaches significance
with probability 1 on pure noise. So live calls are a **replication set, never training
data**, and the correction is applied over `registry × n_looks` — the bar **rises** every run.
You cannot wait your way to a finding.

## What shipped

An endpoint that **abstains on every tweet** — because nothing survived correction, and a
calibrated refusal is the correct output. It quotes no accuracy and no confidence score,
because nothing I measured earned one.

Serverless fits research better than I expected: every stage is deterministic, containerized,
hash-stamped, independently re-runnable, CPU-only, and scales to zero. But the real lesson is
cheaper than infrastructure. Six of my seven artifacts were caught by a *control*, not by
inspection — and the seventh was caught because a tweet about a **renovated Palm Room**
topped the leaderboard.

Build the thing that can say no. Then check whether it's saying no for the right reason.
