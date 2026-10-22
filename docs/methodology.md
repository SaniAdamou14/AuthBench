# Methodology

Full rationale lives in [`AuthBench_Specification.md`](../AuthBench_Specification.md).
This document summarizes the protocol actually implemented in `src/authbench/`.

## Pipeline

```
ingest -> parse/clean -> label -> split -> features -> models -> evaluate -> report
```

Each stage is a pure function over Polars `LazyFrame`s (see `authbench.models.base.AnomalyScorer`
for the model contract). No stage re-reads raw data more than once (NFR-01), and no stage after
`split` can reach the test partition except through `authbench.split.temporal.get_test_split` —
enforced by `tests/unit/test_no_test_leakage.py`.

## Temporal split (US-107)

Boundaries live in `conf/split/temporal.yaml`, expressed in 0-indexed `day = time // 86400`:

| Partition | Days (0-indexed) | Events | Positives | Campaigns |
|---|---|---:|---:|---:|
| Train | 0–7 | 133,093,586 | 50 | 11 |
| Validation | 8–10 | 51,687,105 | 266 | 45 |
| Test | 11–13 | 54,690,768 | 282 | 52 |

Each partition also reads `history_warmup_days: 2` days before its first day,
featurised but never written, so its first events have a past. This is the
evaluated configuration (`reports/lanl/wider_window_results/`). The first LANL
run used one day per partition (train 5, validation 8, test 12) without warm-up,
because an 8 GB machine could not hold more; it is kept in `reports/lanl/RUN.md`.

**Day-by-day computation (since 22 October 2026).** `pipeline.build_features`
featurises one day at a time and carries what later days need: pair, user-source,
source-destination and user-auth-type counts with their last-seen times (F3), each
user's circular sums, last time and last destination (F4), and the previous day
for the 24-hour windows (F2). The carried sums enter each cumulative sum as its
first term, so the features equal those of the whole window computed at once, bit
for bit (`tests/unit/test_incremental.py`, `tests/integration/test_build_store.py`).
Partition bounds may be given in days (inclusive) or hours (half-open), and
`split.history` selects `window` (each partition from its own first day minus the
warm-up) or `cumulative` (one pass from day 0, every event compared with all
earlier ones). `features.universe` restricts the events to all, remote LogOn
(source differs from destination), or remote LogOn to a user account; it is part
of the feature-store hash because it changes the history each event is compared
against. `conf/split/harmonised.yaml` combines hour bounds, cumulative history and
the remote LogOn universe to reproduce the protocol of Larroche (2026); it has no
validation partition, so M1 keeps uniform rule weights there.

**Why not the spec's split.** The spec's own split (train 1–30, validation 31–40,
test 41–58, i.e. 0–29 / 30–39 / 40–57 here) cannot be run at all: the red team stops at
day 29, so it puts every one of the 749 red-team events in train and leaves validation and
test with none. Every ranking metric would be undefined and M1's weight calibration would
have nothing to maximise. `conf/dataset/lanl.yaml` records `redteam_active_days` for exactly
this reason, and `tests/unit/test_split_has_positives.py` refuses a split that repeats the
mistake.

The constraints that a LANL split must satisfy (campaigns in every partition, temporal
order, red-team activity only on days 1 to 29, and a partition size the machine can hold)
are set out in `conf/split/temporal.yaml`. Day 11 holds no red-team event, so the test
positives are on days 12 and 13. Each `split_summary.json` records, per partition, the
events, positives, campaigns and the source hosts of the positives.

`authbench.split.temporal.verify_temporal_order` raises `LeakageError` unless
`max(train.time) < min(val.time)` and `max(val.time) < min(test.time)`.

## Causal features (US-108)

All windowed features (F2 history, F3 novelty, F5 graph) go through
`authbench.features.causal`, which provides two primitives:

- `causal_rolling_count` / `causal_rolling_sum` — Polars `rolling_*_by(..., closed="left")`,
  i.e. the window `[t - window, t)` that **excludes** the current event's own timestamp.
- `causal_prior_events` — a self-join for statistics a running sum can't express (distinct
  counts, entropy), restricted to strictly-prior events of the same entity.

`tests/unit/test_causality.py` builds synthetic events whose feature value would change if the
future were visible and asserts the computed value is the causal one.

## Novelty relations (F3)

F3 describes three relations, each by a first-occurrence flag and a rarity
`-log((k + 1) / (n + 1))`: the user-destination pair (`pair_*`, n = all events), the
user-source pair (`user_src_*`, n = the user's events) and the source-destination edge
(`src_dst_*`, n = the source's events). k counts earlier occurrences in (time, event_id)
order; n counts events on days before the event's day, so equally novel events of one day
tie instead of being ordered by their time. The user-source relation is the "known source
pc" signal of Garchery and Granitzer (2020); M2c scores it alone and M1's rule R8 flags its
first occurrences. The published LANL runs predate the source relations and the day anchor.

## Fitted quantities (US-107, US-109, US-112)

Two things in the feature pipeline are learned rather than computed row-locally, and both are
fitted on the **training split only**, then applied unchanged to validation and test:

| Quantity | Fitted by | Applied by |
|---|---|---|
| Night window | `features.temporal.calibrate_night_window` | `compute_f4` |
| F1 category frequencies | `features.event.fit_frequency_encoding` | `compute_f1` |

Fitting either one per split is a leak that produces no error and no warning — it lets an
event's encoding depend on events that came after it, and gives the same `auth_type` a
different numeric value in train and in test. `compute_f1` therefore takes the fitted
encoding as a required argument rather than an optional one; a protocol that depends on the
caller remembering a keyword is not a protocol.
`tests/unit/test_frequency_encoding.py` is the regression guard.

## Two training regimes (section 6.2)

**Not implemented yet.** The key exists in `conf/split/*.yaml` and nothing reads it. When it
lands:

- **R1 (unsupervised)**: red-team events inside the training window are left in — the realistic
  situation of a defender who doesn't know they're already compromised.
- **R2 (semi-supervised)**: red-team events are removed from training.

Both are to be reported; most published work silently uses R2 only, which inflates results.

## Evaluation (US-125 to US-128)

`authbench.evaluate.summary.evaluate_model` assembles every reported metric, and both the
`train_eval` stage and `authbench demo` go through it — so the two cannot drift into
reporting different things. Metrics stay in two registers, kept under separate keys in
`metrics_summary.json` / `demo_results.json`:

**Operational** — what a SOC at this budget actually gets:

- **AUC-PR**, the headline ranking metric, with a campaign-stratified CI.
- **Recall at alert budget** (`evaluate.budget`) — top-k scored events **per day**, never
  top-k over the whole period. Reported per event and per campaign.
- **Tie-break, stated rather than inherited.** "The top 10 events of the day" is not a
  question a model with a coarse score answers on its own: M0b always-fail puts 2,619
  equally-scored demo events in competition for 8 places. Equal scores are ordered by
  `(time, event_id)` — arrival order, label-free and total (`ALERT_ORDER_TIE_BREAK`).
  Before this was explicit, ties fell to the frame's row order, which after `compute_f4`
  is `(src_user, time)`; the published floor row was the failures of the alphabetically
  earliest users. `campaign_recall_bracket` now reports what the tie could cost (exact)
  and buy (achievable), and `tie_exposure` the slots and contenders behind the width. A
  zero-width bracket certifies that the tie-break decided nothing, which is every model
  whose score is continuous.
- **Budget for first detection** (`budget_for_first_detection`) — the smallest daily
  budget at which the model catches any campaign. Recall at a fixed budget saturates at
  zero exactly in the regime this benchmark studies, ranking every failed model as equally
  far from working; this does not. A campaign is detected at budget `k` iff one of its
  events reaches rank `k` in its own day, so the answer is the minimum rank over campaign
  events — one pass, exact, no sweep over candidate budgets.
- **Time-to-detection** (`evaluate.campaign`) — campaigns never detected at a given budget
  are counted in `n_never_detected`, never dropped from the denominator and never folded in
  as a delay of zero.

**Literature-comparable** — reported so results can be placed next to published ones, never
to rank models:

- **ROC-AUC** (`evaluate.metrics.roc_auc`), which always emits `ROC_AUC_WARNING`.
- **Global precision@k** — the top-k over the whole period, as the literature reports it,
  which is *not* the per-day operating point above.
- **Recall at fixed FPR**.

The gap between the registers is the project's thesis, and both runs show it. On the demo
sample M3a reaches ROC-AUC 0.798 and catches nothing at 10, 50 or 100 alerts/day. On the
LANL wider window no model detects a campaign at any budget up to 500, and the ordering by
ROC-AUC does not track the ordering by budget for first detection (Spearman −0.10 over the
five non-trivial models; AUC-PR: 0.70). The best ROC-AUC (M2a, 0.941) is third
operationally and the worst (M1, 0.563) first. With five models these coefficients are
descriptive: "decoupled", not "anti-correlated", is the claim the data support.

## Uncertainty and pairwise comparisons (US-128)

`stats_tests.paired_campaign_bootstrap` runs **one** campaign-stratified resampling pass and
evaluates every model on each resample. That single pass yields both the per-model CIs and
every pairwise difference, which means the intervals and the comparisons come from the same
sampling distribution, and each difference is paired: campaign-composition noise cancels
between two models rather than being counted twice.

`eval.bootstrap.stratify_by` accepts only `campaign`; any other value is rejected rather than
silently ignored.

Three artifacts of the method can manufacture a verdict that is indistinguishable from a real
one — two of them a spurious "no difference", the third a spurious clean sweep. All are
handled explicitly:

| Floor | Cause | Handling |
|---|---|---|
| Degenerate resamples | A ranking metric over zero positives is undefined, but scikit-learn returns 0.0, so every model ties — and ties count in both tails. With 1 campaign in the test split ~37% of resamples are degenerate, flooring p near 0.74. | Discarded and redrawn; `n_degenerate_discarded` reported. A frame with no positives at all raises. |
| Bootstrap resolution vs. Holm | A percentile bootstrap cannot report p below `2/(R+1)`; Holm's strictest threshold is `alpha/n_pairs`. The 8-model catalog needs `R >= 1119`, so the original `n_resamples: 1000` could never yield a significant result. | `minimum_resamples_for_family` computes the bound; `comparisons()` warns when it is not met. `conf/eval/default.yaml` sets 2000. |
| A single campaign | The sample size of a campaign-stratified bootstrap is the number of campaigns. With one, every resample re-weights the same attack: no pairwise difference changes sign, the tail count is 0, and every p-value lands on `2/(R+1)` — *below* Holm's threshold, so **every** comparison reads "outperforms". The demo produced 21/21 this way. | `MIN_CAMPAIGNS_FOR_SIGNIFICANCE = 2`. Below it `comparisons()` logs the reason and reports every pair as not significant. Point estimates and intervals are unchanged; only the verdict is withheld. |

**How a resample is drawn (since 22 October 2026).** A resample draws `n_blocks` blocks
with replacement from the campaigns and the benign singletons: how many land on campaigns is
binomial, and the campaign multiplicities are then multinomial and uniform. AUC-PR depends on
the benign multiplicities only through the benign weight at or above each distinct positive
score, so each model's benign draws are taken as one multinomial over its benign bins
(`stats_tests.benign_bins`: the benign events between and level with consecutive positive
scores). That has exactly the distribution the per-event draw induces, at a cost of a few
hundred numbers per resample instead of one per benign event (about 4e8 on LANL). The campaign
multiplicities, the benign total and the redraws are shared by every model, and each model's
benign split restarts one generator from the same seed, so two models with the same bins (a
model and itself) receive the same draws.

What this changes: the per-model intervals have exactly the old distribution; the comparisons
stay paired exactly in the campaign component, which carries the variance at this positive
rate. Measured on identical scores (the demo split, 20,000 resamples, both schemes): point
estimates equal to six decimals, interval bounds to four, the same significance verdict on 28
of 28 pairs, raw p-values within 0.006. A test checks that a resample's AUC-PR equals
scikit-learn's weighted average precision for per-event weights consistent with the bins. The
published LANL runs used the per-event draw (`stats_tests.CampaignBlocks`, removed with it).

## Every model is row-local, to the last bit

The test split is scored one day at a time, and the positives on their own
(`pipeline.train_eval.score_split`), so an event must score the same alone, in any chunk and
in the whole split. A tolerance would hide the failure that matters: a one-ULP difference moves
an event across a tie and changes its rank. `tests/unit/test_row_locality.py` asserts bit
equality for every model, and every scored day re-checks it on the positives.

Five models failed this before 22 October 2026, and the published LANL runs used them as they
were:

- **M3b ECOD was transductive.** PyOD's `ECOD.decision_function` recomputes the empirical
  distributions from the training sample concatenated with the matrix it scores, so the same
  row scored 10.64 alone and 10.81 among four hundred others, and M3b's scores depended on the
  test distribution. It is now fitted: per-feature empirical CDFs and skewness signs from the
  training sample, and an event scored as PyOD would score it if it were the only event scored,
  with tail counts `(count + 1) / (n + 1)` (Li et al., 2022). ECOD was excluded from both LANL
  runs for memory, so no published LANL number changes.
- **M0a drew by position.** A seeded generator over the frame scored the k-th row, not an
  event. It now hashes `event_id` with the seed (SplitMix64), uniform on [0, 1).
- **M2b PCA, M4a and M4b depended on the batch** through BLAS and torch matrix products, whose
  summation order follows the shape of the whole matrix: one ULP for PCA, up to 1.9e-6 for the
  VAE. They now score in `models.rowwise` arithmetic (one input column at a time, separate
  multiply and add), the autoencoders as their trained network evaluated in float64. M4c, still
  batched in torch, declares itself not row-local and is refused by the day-by-day scorer.

The earlier chunked-scoring test compared scores with a tolerance of 1e-12, which is why the
PCA difference went unnoticed.

**M1 was transductive in the published runs.** Until 22 October 2026 it divided each rule's
rank by the size of the frame being scored, so an event's M1 score depended on the other test
events, across the three test days. M1 now scores each rule on an empirical CDF fitted on the
last training day (`normalization_days`, seeded sample of at most 2,000,000 events), and R6 and
R7 are stored features, so M1 is row-local. The LANL results in `reports/lanl/` predate the
change; the next run uses the fitted normalisation.

## Evaluation from sufficient statistics

Every reported metric depends on the scores only through the positives
(`evaluate.statistics.SplitAccumulator`, since 22 October 2026): each positive's rank in its
day's queue and under the two bracketing tie-breaks, the benign events of the whole split above
and level with each distinct positive score, and, per day, the tie at each budget and the first
events in queue order. These are accumulated one day at a time, so evaluating a split needs the
memory of its largest day (measured: 872 bytes per event of the day, `docs/scaling.md`). Each
metric is tested equal to its whole-frame definition (`evaluate.budget`, `evaluate.campaign`,
`evaluate.ledger`, scikit-learn) on tie-heavy frames.

Three definitions changed with it, each for a reason a test now pins:

- **Global precision@k** breaks ties in queue order (time, then `event_id`), as the daily
  budget does. It used the frame's row order, the artifact `evaluate.budget` documents.
- **Recall at a fixed FPR** reads every threshold. `roc_curve`'s default drops collinear ROC
  points, and one of them can be the answer: for labels (0, 0, 0, 1, 1) and scores
  (0, 1, 2, 0, 1) at FPR 2/3 the true recall is 0.5 and the old function returned 0.
- **Score resolution** reports exact quantities computable by day: the share of
  (positive, benign) pairs on the same score, which is the part of ROC-AUC credited one half,
  and per day the fewest distinct scores and the largest tie as a share of the day. The global
  distinct-value count it replaced would need every score of the split in memory.

## Labels

M1 is the only model that uses labels: its weights maximise AUC-PR on the validation
partition (200 Dirichlet draws, seeded), and `RulesScorer.requires_labels` is `True`. Every
other model is unsupervised, with default hyperparameters, and never reads the validation
partition. This favours M1 and is stated as such in the report; an equal tuning budget for
every model is planned.

## Scale

The protocol is written against 1.05 billion auth events. Five changes brought a full run from
~161 GB of disk and ~439 GB of RAM down to ~81 GB and ~49 GB — a day-window conversion, a
feature store projected to the columns anything reads, bounded-sample fitting, day-chunked
scoring, and a rank-ordered AUC-PR bootstrap. None of them moves a measured number:
`reports/demo/` regenerates byte for byte after all five. Since 22 October 2026 features and
evaluation both run one day at a time, so memory follows the largest day rather than the run.

[`scaling.md`](scaling.md) has the measured per-stage budget, the recommended 14-day LANL
slice, and what each change bought. `authbench preflight` prints the budget for the machine it
runs on and exits non-zero when the run does not fit, naming the event count that would.

`eval.pairwise_test.method` accepted `permutation`, a per-event paired test, until 22 October
2026. It assumed the two models' scores exchangeable event by event, which campaign structure
violates, and it needed every event's score in memory; it was removed with the per-event
evaluation.

`render_comparison_sentence` can only emit "outperforms" on the `significant` branch, and
reflects the reported interval through zero whenever it swaps the pair into
`better - worse` order.
