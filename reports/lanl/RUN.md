# LANL run — 23 August 2026

The project's first results on real data. Committed output, like
[`reports/demo/`](../demo/RUN.md), but this one **is** a finding: the attacks
were carried out by a red team that had never heard of any model in the
catalog.

Read the limitation section before quoting any number. It is not boilerplate —
it decides which of the two available readings of a 0% recall is correct.

## What was run

| | |
|---|---|
| Source | LANL `auth.txt.gz`, SHA-256 `9c6b0cc2…f672`, 7,626,505,158 bytes |
| Converted | days 0–13 → 239,471,459 events (`--days 0:13`) |
| Split | **train day 5 · validation day 8 · test day 12** — one day each, not contiguous |
| Feature store | 33 columns, F1–F4, `history_warmup_days: 0` |
| Models | 8 in the catalog, **7 evaluated** (see exclusions) |
| Bootstrap | 2,000 campaign-stratified resamples, Holm-Bonferroni over 21 pairs |

| split | events | positives | campaigns | positive rate |
|---|---:|---:|---:|---:|
| train | 18,557,382 | 21 | 4 | 1.13 × 10⁻⁶ |
| validation | 19,374,688 | 261 | 45 | 1.35 × 10⁻⁵ |
| test | 19,870,848 | **207** | **39** | 1.04 × 10⁻⁵ |

The split is one day per partition and skips the days between, because three
constraints hold at once and LANL does not admit a tidy answer to all three:
the red team stops at day 29 and is absent from days 0, 3, 4, 10 and 11 inside
the converted window; temporal order must hold (5 < 8 < 12); and no partition
may exceed ~20M events or the feature build exhausts an 8 GB machine. A gap is
not a leak — nothing from day 6 or 9 reaches any model — and testing on day 12
after training on day 5 is a *harder* generalisation test than the next day
would be.

## Results

| Model | AUC-PR [95% CI] | ROC-AUC | Camp. @10 | @50 | @100 | @500 |
|---|---|---:|---:|---:|---:|---:|
| M0a random *(floor)* | 0.00001 [0.00000, 0.00002] | 0.500 | 0% | 0% | 0% | 0% |
| M0b always-fail *(floor)* † | 0.00001 [0.00000, 0.00002] | 0.496 | 0% | 0% | 0% | 0% |
| **M2a pair rarity** | **0.00065** [0.00023, 0.00131] | **0.942** | 0% | 0% | 0% | 0% |
| M2b PCA reconstruction | 0.00005 [0.00002, 0.00011] | 0.893 | 0% | 0% | 0% | 0% |
| M3a Isolation Forest | 0.00004 [0.00001, 0.00010] | 0.872 | 0% | 0% | 0% | 0% |
| M3b HBOS | 0.00005 [0.00002, 0.00012] | 0.875 | 0% | 0% | 0% | 0% |
| M1 rules | 0.00010 [0.00002, 0.00071] | 0.547 | 0% | 0% | 0% | **5.1%** |

**Not one of the seven detects a single campaign out of 39 at 10, 50 or 100
alerts per day.** One catches two campaigns at 500 alerts/day, a budget no SOC
staffs for.

13 of the 21 pairwise comparisons are significant after Holm-Bonferroni. M2a
beats both floors, PCA, Isolation Forest and HBOS. M2a versus M1 is *not*
significant (Holm-adjusted p = 0.828; raw p = 0.138), and neither is M1 versus
the random floor (Holm-adjusted p = 0.176; raw p = 0.022).

**Twelve of those thirteen are bounds, not measurements.** No resample out of
2,000 put the difference on the other side of zero, so their p-value is the
bootstrap's own resolution floor `2/(R+1)` and they are reported as
`Holm-adjusted p ≤ 0.0210` against a threshold of 0.05 — true, with a factor
of 2.4 of margin, and not a number the run resolved. The single exception is
M0b always-fail versus M1 rules at Holm-adjusted `p = 0.0450`, the one pairwise
verdict here whose p-value the bootstrap actually measured. Raising
`eval.bootstrap.n_resamples` is the only thing that buys resolution;
`at_resolution_floor` in `tables/pairwise_comparisons.json` marks every affected
pair.

> **Note on the M0b row, September 2026.** M0b always-fail scores every event
> 0 or 1, so on a 19.9M-event test day its alert set at any budget is decided
> entirely by how equally-scored events are ordered. This run used the ordering
> that was in force at the time — the frame's own row order, which after
> `features.temporal.compute_f4` is `(src_user, time)`, i.e. alphabetical by
> user. **M0b's zeros are therefore a property of that ordering, not of the
> floor**, and should be read as undetermined rather than as 0%. On the demo
> sample the same fix moved M0b from 0% to 75% at budget 500, with a tie
> bracket of `[0, 1]`. `evaluate.budget` now breaks ties on `(time, event_id)`
> and publishes `campaign_recall_tie_bracket` and `tie_exposure` next to every
> recall; this snapshot predates all three and cannot be re-run cheaply. Every
> other row is unaffected — their scores are continuous and leave the tie-break
> nothing to decide — and nothing here touches AUC-PR, ROC-AUC, the intervals
> or the pairwise comparisons, none of which depend on the alert ordering.

> **Note on this table, September 2026.** `tables/pairwise_comparisons.json`
> was re-emitted in the current schema — `p_value_raw`,
> `p_value_holm_adjusted` and `at_resolution_floor` in place of a single
> `p_value_corrected` that held the *uncorrected* value. Nothing was re-run and
> nothing could have been: the adjusted p-value is a function of the 21 raw
> p-values already in the file, and the floor flag is `p == 2/(R+1)`. Every
> `significant` verdict came out identical, which the migration asserted rather
> than assumed. The point estimates, intervals and recall figures below are the
> originals, untouched.

## The two registers are anti-correlated, not merely different

This is the benchmark's reason to exist, and here it is on real attacks:

- **M2a scores ROC-AUC 0.942 and detects nothing** at any operational budget.
  0.94 is a number that passes without comment in a paper.
- **M1 has the lowest ROC-AUC of the five non-trivial models, 0.547 — barely
  above the 0.50 the two floors score by construction — and is the only model
  that catches anything at all.**

Ranking the non-trivial models by the literature-comparable metric would put
the model that detects nothing first and the only model that detects something
last.

## The demo's M1 result was circular; this one is not

On the synthetic sample M1 reached 100% campaign recall at a budget of 10. The
generator emitted campaigns as A→B→C chains and M1's rule R7 tests for A→B→C
chains — the same predicate on both sides, worth 113× separation and 0.52 of
M1's fitted weight (see [`../demo/RUN.md`](../demo/RUN.md)).

Against a red team that never heard of R7: **5.1% at budget 500, 0% everywhere
else.** The circularity diagnosis was correct, and this run is the control that
proves it.

## Exclusions

`tables/skipped_models.json` records what was not evaluated and why.

**M3b_ecod** — `MemoryError`, 3.89 GiB. ECOD is transductive: PyOD's
`decision_function` concatenates the fitted sample onto the frame being scored
and argsorts all 21 columns of the result, here 24,870,848 rows. This is a
property of the model, not of the harness, and it also means ECOD's scores
depend on the test distribution (see [`../../docs/methodology.md`](../../docs/methodology.md)).
Its row-local sibling HBOS represents M3b.

## Limitation: one day of history, and what that costs

**Each partition is a single day, with no warm-up.** F2's 24-hour windows and
F3's cumulative pair statistics therefore have at most one day of past inside
their own partition.

This matters because it makes a 0% recall ambiguous between two readings:

1. the models genuinely detect nothing at an operational budget; or
2. one day is not enough history to establish what counts as *new*, and the
   novelty features — which this benchmark argues carry the lateral-movement
   signal — are being measured on a past that barely exists.

The size of the effect is measured, not guessed. On the demo sample, adding two
warm-up days halves `pair_is_new` (2,623 → 1,287), raises mean
`pair_global_rarity` from 7.00 to 7.73, and raises mean `src_user_1d_n_events`
from 7.85 to 10.33 — over an identical set of written events. **Without
history, half the pairs called "never seen before" were only unseen because
there was no past to have seen them in.**

`history_warmup_days` exists and works; the machine cannot afford it. One
warm-up day makes each split process 38M rows instead of 19M and died after
writing the first split; two makes it 57M and died before writing anything.
[`../../docs/scaling.md`](../../docs/scaling.md) sets out what lifting this
needs.

**What the limitation does not touch:** the anti-correlation between ROC-AUC
and campaign recall above. That is a property of the operating point and of the
positive rate, not of how much history a feature had.

## Reproducing

```bash
authbench data download --dataset lanl            # verifies SHA-256, downloads nothing
authbench data to-parquet data/raw/auth.txt.gz --out-dir data/interim/auth --days 0:13
POLARS_MAX_THREADS=6 python -m authbench.pipeline.build_features
python -m authbench.pipeline.train_eval
```

`POLARS_MAX_THREADS` is load-bearing on a small machine: Polars allocates
per-thread buffers, and the default of one per logical core is what turns a
working build into a paging one. The bootstrap is separately parallel over
models (`runtime.bootstrap_workers`) — 4h49 on one core became roughly 1h15 on
four, with bit-identical results.

Timings on the reference machine (8 GB RAM, 6 cores): conversion 2.5 min,
feature build 18 min, evaluation 5h20 before the bootstrap was parallelised.
