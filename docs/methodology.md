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

Computed and written into `reports/tables/` by the `train_eval` stage and by `authbench demo`:

- **AUC-PR**, the headline ranking metric.
- **Recall at alert budget** (`authbench.evaluate.budget`) — top-k scored events **per day**,
  never top-k over the whole period. Reported per event and per campaign.
- **Confidence intervals** (`authbench.evaluate.stats_tests.bootstrap_ci`) resample at the
  **campaign** level (each campaign is one block), not the event level.

Implemented and unit-tested, but not yet called by either pipeline — the functions are
usable from a notebook, they simply do not appear in the generated tables:

- **ROC-AUC** (`evaluate.metrics.roc_auc`), which always emits `ROC_AUC_WARNING`;
  **precision@k**; **recall at fixed FPR**.
- **Time-to-detection** (`authbench.evaluate.campaign`) — campaigns never detected at a given
  budget are counted in `n_never_detected`, never dropped from the denominator.
- **Pairwise comparisons** (`evaluate.stats_tests.compare_models`) — paired permutation test
  with Holm-Bonferroni correction; `render_comparison_sentence` can only emit "outperforms"
  on the `significant` branch.
