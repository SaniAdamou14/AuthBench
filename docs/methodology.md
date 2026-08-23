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

| Partition | Days (0-indexed) | Spec's 1-indexed equivalent |
|---|---|---|
| Train | 0–29 | 1–30 |
| Validation | 30–39 | 31–40 |
| Test | 40–57 | 41–58 |

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
- **Time-to-detection** (`evaluate.campaign`) — campaigns never detected at a given budget
  are counted in `n_never_detected`, never dropped from the denominator and never folded in
  as a delay of zero.

**Literature-comparable** — reported so results can be placed next to published ones, never
to rank models:

- **ROC-AUC** (`evaluate.metrics.roc_auc`), which always emits `ROC_AUC_WARNING`.
- **Global precision@k** — the top-k over the whole period, as the literature reports it,
  which is *not* the per-day operating point above.
- **Recall at fixed FPR**.

The gap between the registers is the project's whole thesis, and the demo sample already
shows it: M2b reaches ROC-AUC 0.98 while catching nothing at a budget of 10 or 50 alerts/day.

## Uncertainty and pairwise comparisons (US-128)

`stats_tests.paired_campaign_bootstrap` runs **one** campaign-stratified resampling pass and
evaluates every model on each resample. That single pass yields both the per-model CIs and
every pairwise difference, which means the intervals and the comparisons come from the same
sampling distribution, each difference is paired (campaign-composition noise cancels between
two models rather than being counted twice), and the cost is `O(n_resamples x n_models)`
instead of `O(n_permutations x n_pairs)`.

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

The resampling itself is `stats_tests.CampaignBlocks`: campaigns as explicit index arrays,
every benign event as one entry of a single array. Each draw picks `n_blocks` blocks with
replacement — a binomial split between the campaign set and the singleton set, then uniform
within each — which is exactly the uniform draw over the combined list, at a cost that does
not involve materializing one Python object per benign event. `build_campaign_blocks` keeps
the explicit form as the readable reference the tests check against.

## One model is transductive, and it is not the one you would guess

`M3b_ecod` is not a pure "fit here, apply there" model. PyOD's
`ECOD.decision_function` recomputes the empirical cumulative distribution from the matrix it
is handed, so an event's score depends on which other events are scored alongside it: the same
row scores 10.64 alone and 10.81 among four hundred others.

No label crosses the train/test boundary, so this is not leakage in the sense US-107 guards
against, and it is how ECOD is used in the literature. But it does mean M3b's test scores
depend on the *test* distribution, and a benchmark claiming a strictly temporal protocol owes
its reader that sentence rather than leaving it in a library's source.

It was found by `tests/unit/test_chunked_scoring.py`, which exists to assert the opposite —
that scoring a split piecewise gives the same answer as scoring it whole. `M3a_iforest`,
`M3b_hbos`, `M2b_pca_reconstruction` and `M0b_always_fail` pass that test; ECOD does not, and
`AnomalyScorer.scores_row_locally` records which is which. So does M1, for a different and
intended reason: R6 and R7 look backwards across the split by design.

## Scale

The protocol is written against 1.05 billion auth events. Five changes brought a full run from
~161 GB of disk and ~439 GB of RAM down to ~81 GB and ~49 GB — a day-window conversion, a
feature store projected to the columns anything reads, bounded-sample fitting, day-chunked
scoring, and a rank-ordered AUC-PR bootstrap. None of them moves a measured number:
`reports/demo/` regenerates byte for byte after all five.

[`scaling.md`](scaling.md) has the measured per-stage budget, the recommended 14-day LANL
slice, and what each change bought. `authbench preflight` prints the budget for the machine it
runs on and exits non-zero when the run does not fit, naming the event count that would.

`eval.pairwise_test.method` also accepts `permutation` — the exact per-event paired test
(`compare_models`). It assumes the two models' scores are exchangeable event by event, which
is precisely the independence campaign structure violates, and it costs
`n_permutations x n_pairs` metric evaluations. Kept available for small splits; not the
default.

`render_comparison_sentence` can only emit "outperforms" on the `significant` branch, and
reflects the reported interval through zero whenever it swaps the pair into
`better - worse` order.
