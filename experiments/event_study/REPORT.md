# Event study: does Trump move markets? (pre-registered, discovery set = 476)

**Question.** Directional prediction is a confirmed null. This study asks the
weaker, sharper question: do high-impact Trump posts produce *bigger* abnormal
moves (|CAR|), *systematic drift* (signed CAR), or *abnormal volume* in their
mapped assets than random days do?

**Design.** Outcome-blind cohorts from text-only tags (GEO_SHOCK n=184 unique
event-days, CORPORATE n=43, NOISE control n=89) × fixed asset map × windows
{1,3,5} sessions × families {|CAR|, signed, volume} = **72 scored cells**, all
enumerated in `registry.json` **before** scoring. Market model per event: OLS
beta vs SPY on 120 sessions ending 6 before s0 (SPY itself: Brown-Warner
mean-adjusted). Entry anchor = first open strictly after t0. Permutation null =
same statistic on random session-days, drawn **with replacement**; one BH pass
over all 72 cells. Seed 20260715, N_perm=1000, fully offline (committed daily bars).

## Result: **nothing survives. 0 of 72 cells.** (min p_bh = 0.83)

| closest cells (all fail) | n | observed | null | p_raw | p_bh |
|---|---|---|---|---|---|
| GEO_SHOCK / GLD w=5 signed | 184 | −0.0070 | −0.0029 | 0.043 | 0.831 |
| GEO_SHOCK / ITA w=3 signed | 184 | −0.0036 | −0.0018 | 0.074 | 0.831 |
| GEO_SHOCK / USO w=1 volume | 184 | 1.99× | 1.71× | 0.106 | 0.831 |

Even the suggestive-looking USO volume (~2× estimation baseline after oil
tweets) fails: the null pool *also* runs 1.71× — elevated volume is the 2025-26
oil regime, not the tweets.

## Why the anecdotes feel true anyway: two artifacts we caught mid-study

1. **A degenerate null flagged 22/72 cells as "significant" on the first run.**
   Geo-tweets are so frequent that event-days = 83% of ALL trading sessions;
   sampling n≈pool without replacement collapsed the null's variance (finite-
   population effect) and floored every p. Fixed: dedupe to unique event-days
   (667 same-day duplicates removed) + bootstrap null with replacement →
   survivors went 22 → 0. *The first run's output looked exactly like the
   discovery the anecdotes promise. It was arithmetic.*
2. **The calendar is saturated — there is no counterfactual.** With Trump
   posting geo/macro content on most trading days, "did the market move after a
   Trump tweet?" degenerates to "did the market move?" — at daily resolution the
   treatment has almost no control group. This is the identification problem in
   one line, and it is why isolated anecdotes cannot be tested at this resolution.

## The case studies, surfaced by data (top |CAR| day-1)

| s0 | cohort/asset | CAR 1d | tweet |
|---|---|---|---|
| 2026-03-09 | GEO_SHOCK/USO | **−13.2%** | "There will be no deal with Iran except UNCONDITIONAL SURRENDER!…" |
| 2026-02-17 | CORPORATE/DJT | −10.2% | "Happy President's Day! Prices and Inflation are Way Down…" |
| 2025-06-23 | GEO_SHOCK/USO | −8.8% | "I am very happy to report that I have arranged… [Iran ceasefire]" |
| 2025-08-14 | CORPORATE/INTC | **+8.0%** | tweet that day was about **Comey** — unrelated to Intel |

Hormuz-type oil moves on Iran days are real and large — *individually*. But the
INTC row is the tell: an 8% Intel move "after a Trump tweet" whose tweet had
nothing to do with Intel. When someone posts every day, every big market move
has a same-day tweet. **The anecdotes are survivorship + coincidence-by-
saturation; the population effect, measured against an honest null, is zero.**

## Exhibit A: the Intel tweet is REVERSE CAUSALITY

The most-cited anecdote — *"he tweeted about Intel and the stock rose for a long
time"* — is the sequence run backwards. There are exactly **4 Intel-the-company
mentions** in 8,317 posts (2025-01-01 → 2026-07-06). The famous one:

> `2026-04-29` — *"**Intel Stock continues to rise.** I'm very proud of that Company
> in that **I am responsible** for making the United S…"*

**INTC before that tweet:**

| window before | INTC | SPY | INTC excess |
|---|---|---|---|
| prior 5d | +29.5% | +0.1% | **+29.4%** |
| prior 21d | **+105.2%** | +12.6% | **+92.6%** |
| prior 42d | +85.3% | +4.0% | +81.3% |

**INTC after it:** 1d **−3.0%**, 21d **−2.4%**.

Intel had already **doubled** — then he tweeted about it, claiming credit — then
nothing. He is a *reflector* of news, not a generator of it: he posts about
winners after they win, and memory encodes the correlation as a sequence. (The one
Intel tweet followed by a rise, `2026-01-08` +7.6% 1d, reports a CEO meeting and a
product launch — the news was the news; n=1.)

## Exhibit B: the "fear then recovery" arc is mean reversion, not Trump

USO abnormal-return paths, Iran/Hormuz/oil tweets vs controls:

| | 1d | 3d | 5d | 10d | 21d | 42d |
|---|---|---|---|---|---|---|
| top-5 "strongest" tweets | −5.34% | −3.99% | +0.01% | +4.03% | +8.03% | +12.71% |
| all 64 tweet-days | −0.21% | +0.36% | +0.50% | +2.48% | +5.39% | +4.52% |
| random days (n=120) | −0.11% | +0.09% | +0.21% | +0.33% | +1.87% | +5.07% |
| **big move, NO tweet (n=4)** | **−3.06%** | −0.61% | +4.39% | +8.40% | +11.85% | **+16.72%** |

The dip-then-recovery shape is **real** — and it recovers *harder* on big-move days
with no tweet at all. It is what volatility does. All 64 tweet-days are
indistinguishable from random days. (Control n=4 is small; it agrees with the
64-vs-random comparison, which is flat.)

## Exhibit C: the anecdote list builds itself from coincidence

Sorting oil events by biggest mover surfaced, among the "top 5 strongest oil tweets":

- *"The **Iran National Soccer Team** is welcome to The World Cup…"* → USO **+17.1%** @42d
- *"The **Radical Left Democrats**, a Party that has completely lost its…"* → USO **+45.4%** @42d

Neither is about oil. With posts on 83% of sessions, sorting on the outcome
manufactures a greatest-hits reel out of pure coincidence. This is the anecdote
generator, visible in one table.

## What would actually answer this
Intraday resolution (minutes around t0, not daily bars) with tweets outside
market hours as natural controls — the only design that restores a
counterfactual. Requires licensed intraday data (Alpaca path); out of scope here.

## Verification
`tests/test_event_study.py`: point-in-time estimation-window guard, strict s0>t0
anchor, overlap flagging, outcome-blind cohort test, shuffled-dates canary (which
caught the SPY-vs-SPY degeneracy during development). Registry check: scored
cells ≡ registered cells. Replication set: the expanded Nebius Job corpus rescores
this exact registry (`--results` flag) when it lands.
