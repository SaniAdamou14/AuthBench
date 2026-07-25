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

## Two training regimes (section 6.2)

- **R1 (unsupervised)**: red-team events inside the training window are left in — the realistic
  situation of a defender who doesn't know they're already compromised.
- **R2 (semi-supervised)**: red-team events are removed from training.

Both are reported; most published work silently uses R2 only, which inflates results.

## Evaluation (US-125 to US-128)

- **AUC-PR** is the headline ranking metric; **ROC-AUC** is reported but always paired with
  `authbench.evaluate.metrics.ROC_AUC_WARNING`.
- **Recall at alert budget** (`authbench.evaluate.budget`) takes the top-k scored events
  **per day**, never top-k over the whole period.
- **Time-to-detection** (`authbench.evaluate.campaign`) — campaigns never detected at a given
  budget are counted in `n_never_detected`, never dropped from the denominator.
- **Confidence intervals** (`authbench.evaluate.stats_tests.bootstrap_ci`) resample at the
  **campaign** level (each campaign is one block), not the event level.
- **Pairwise comparisons** use a paired permutation test with Holm-Bonferroni correction;
  `render_comparison_sentence` can only emit "outperforms" on the `significant` branch.
