# Scaling: what a real LANL run costs, and how to make it fit

Run `authbench preflight` (or `make preflight`) for these numbers computed on
*your* machine, and `authbench preflight --events N` to price a slice. This
document explains where the numbers come from and what was done to bring them
down.

## Where the numbers come from

Every per-event size is **measured**, not guessed, and the two that matter most
are now measured on LANL itself rather than extrapolated from the demo:

| Quantity | Measured | Basis |
|---|---|---|
| Raw LANL-shaped text | 68.7 B/event | `data/demo/auth_demo.txt` |
| Interim ZSTD Parquet (17 typed columns) | **9.7 B/event** | 2.31 GB for 239,471,459 real events |
| Feature store, F1–F4, ZSTD Parquet (33 columns) | **35.5 B/event** | 3 real splits, 48.9M events |
| Design matrix in RAM (21 float64 columns) | 168 B/event | `features.MODEL_FEATURE_COLUMNS` |
| Scored frame + one rank order, 8 models | 100 B/event | `evaluate.summary.EVAL_COLUMNS` |

The demo-derived estimates they replaced were 60% and 57% too high: LANL's much
higher cardinality compresses better than the penalty assumed.

Two adjustments, both stated rather than hidden:

- **×1.4 cardinality penalty.** The demo has fewer distinct users and machines
  than LANL's 12,425 and 17,684, and dictionary-encoded Parquet compresses
  better the fewer distinct values it holds. Every Parquet estimate is inflated
  by this factor, so the figure errs high.
- **Gzip ratio 0.08**, assumed. LANL publishes no file size, and nobody can
  compute the ratio without the file. `authbench data download` replaces the
  assumption with the server's `Content-Length` before writing a byte, and
  refuses to start if the volume cannot hold it.

## What changed, and what it bought

| | Before | After |
|---|---:|---:|
| Full LANL, disk | ~161 GB | **~81 GB** |
| Full LANL, peak RAM | ~439 GB | **~49 GB** |
| Fits in 29 GB free | ~100 M events | **~310 M events** |
| Feature build, demo sample | 9.3 s | **1.4 s** |
| Bootstrap, per resample, 20 M rows, model with signal | 5.2 s | **0.002 s** |

Five changes, none of which moves a single measured number — `reports/demo/`
regenerates byte for byte after all of them, which is the check that says so:

1. **`--days FIRST:LAST` on `to-parquet`.** Converts a window instead of all 58
   days, and — because `auth.txt` is written in time order — stops reading once
   it is past the window, so a two-week slice reads about a quarter of the file.
   Rows outside the window are counted separately from dropped rows, and
   `verify_row_count` *refuses* to reconcile a partial run against the published
   total rather than failing a check nobody could pass.
2. **The feature store persists only what is read.** 33 columns instead of 90.
   The other 57 were produced by eight `causal_prior_events` self-joins whose
   output no model, rule or report ever consumed — and a self-join over a 7-day
   window at LANL's events-per-entity is the single most expensive thing the
   pipeline could do. `features.DEFAULT_DIVERSITY_PAIRS` names the one pair that
   is consumed (M1's rule R2).
3. **Vector-space models fit on a bounded sample.** A covariance, an empirical
   CDF and a histogram do not need 550 million rows; M3a already subsamples to
   256 internally. `runtime.fit_sample_size` is a published parameter, not a
   hidden constant, and **every event is still scored** — only the fit is
   bounded.
4. **Row-local models are scored one day at a time.** Peak memory becomes a day
   rather than a split. `AnomalyScorer.scores_row_locally` marks which models
   this is valid for, and `tests/unit/test_chunked_scoring.py` asserts the
   chunked and whole-split answers are identical per model. That test found a
   real one: **ECOD is transductive** — PyOD recomputes the empirical CDF from
   whatever matrix it is handed, so the same row scores differently depending on
   its company. It is excluded from the chunked path and the property is
   documented in [`methodology.md`](methodology.md).
5. **AUC-PR bootstraps from a pre-sorted rank order.** Sorting the same scores
   once instead of once per resample per model, and dropping every row ranked
   below the last positive — exact, since average precision adds no term where
   recall no longer increases. On 20 M rows: 2,616× faster for a model that
   ranks the attacks high, 3.9× for a random floor that has nothing to exploit.
   `tests/unit/test_fast_auc_pr.py` pins it to scikit-learn's weighted average
   precision to machine precision, ties included.

**Correction, 22 October 2026.** Items 4 and 5 are superseded. Every model is now
row-local to the last bit, ECOD included (`methodology.md`), and the chunked-scoring
test that used a 1e-12 tolerance became `tests/unit/test_row_locality.py`, which
asserts bit equality. The bootstrap no longer holds the scored split: evaluation
reduces each day to sufficient statistics (`evaluate.statistics`) and draws each
model's benign weights as a multinomial over a few hundred bins, so 10,000
resamples of nine models take under a second (measured on a real LANL day).

## Full LANL — 1,051,430,459 auth events

| Stage | Disk added | Peak RAM |
|---|---:|---:|
| `download` (auth.txt.gz + redteam.txt.gz) | ~5.8 GB | 0.1 GB |
| `to_parquet` → `data/interim/auth` | ~16.6 GB | ~2 GB |
| `label_split_features` → `data/processed/features` | ~58.4 GB | ~6 GB |
| `train_eval` (design matrix) | ~0.1 GB | ~2.5 GB |
| `train_eval` (bootstrap) | — | **~49 GB** |
| **Total** | **~81 GB** | **~49 GB** |

Disk is cumulative: no stage deletes its predecessor's output, because DVC needs
the inputs of every stage it might re-run. The remaining ceiling is the scored
frame the bootstrap holds — the whole test split, slim — which is what
`max_events_for_memory` inverts.

**Correction, 22 October 2026: memory follows the largest day.** Features and
evaluation now run one day at a time, so the table above no longer describes the
pipeline. Both peaks were measured on LANL on the 8 GB reference machine:

| Stage | Per event of the largest day | Measured on |
|---|---:|---|
| `label_split_features`, day by day | 713 bytes | remote LogOn days 0-2 (5.8M to 6.6M events), 4.70 GB peak |
| `train_eval`, score and evaluate by day | 872 bytes | day 12, remote LogOn to users (2.6M events), nine models, 3.65 GB peak |

The largest LANL day of days 0-13 holds 1.216 times the mean (20.8M against
17.1M events). For the full dataset `authbench preflight` now gives about 55 GB of
disk and 19 GB of RAM (the evaluation of a 22M-event day); restricted to remote
LogOn events (37.5% of LANL), about 7 GB. `max_day_events_for_memory` replaces
`max_events_for_memory`, and `preflight --days` states the span of `--events`.
Featuring every event of one day exhausted the 8 GB machine, so the all-events
run needs the rented one; the remote LogOn protocol fits here.

## The real run, as actually performed

Days 0–13 convert comfortably — 2.31 GB of Parquet, 2.5 minutes. **The split is
what the machine constrains, not the conversion.** On 8 GB of RAM the feature
build handles about 20 million events per partition and no more, which is one
LANL day:

```yaml
# conf/split/temporal.yaml — one day each, deliberately not contiguous
train_days: [5, 5]
val_days:   [8, 8]
test_days:  [12, 12]
```

Days 0, 3, 4, 10 and 11 carry no red-team activity inside the converted window,
so a contiguous split cannot put campaigns in all three partitions *and* stay
under the size ceiling. These bounds give 4 / 45 / 39 campaigns. On 16 GB the
constraint disappears and `train_days: [0, 7]` with `history_warmup_days: 2`
becomes affordable — which is the single upgrade that would most improve the
result.

**Label every number it produces with the window it came from.** A 14-day slice
of LANL is a real result about 14 days of LANL; it is not the 58-day figure, and
the red-team campaigns it contains are whichever ones fall in those days. Check
that count before trusting any interval — below two campaigns in the test split,
`stats_tests.MIN_CAMPAIGNS_FOR_SIGNIFICANCE` withholds every significance
verdict, and it is right to.

The download is the whole file either way: LANL serves one gzip stream per file
and no range of days within it. Once `to-parquet` has succeeded you can delete
`data/raw/auth.txt.gz` (keep `redteam.txt.gz`, the labeling stage reads it) and
recover about 5.8 GB — at the cost of re-downloading if you later want a
different window.

## What the published LANL run actually cost, and what it could not buy

Measured on the reference machine — **8 GB of RAM, 6 cores** — for the run in
[`../reports/lanl/RUN.md`](../reports/lanl/RUN.md):

| Stage | Wall time | Note |
|---|---:|---|
| `download` | ~5 h | 183 kB/s from `csr.lanl.gov`, resumed three times |
| `to-parquet --days 0:13` | 2.5 min | 239M rows, stopped early at day 14, 2.31 GB |
| `build_features` | 18 min | 3 splits x ~19M rows, `POLARS_MAX_THREADS=6` |
| `train_eval` | 5 h 20 | of which **4 h 49 in the bootstrap, on one core** |

Two knobs turned out to be load-bearing on a machine this size, and neither is
discoverable from the code:

- **`POLARS_MAX_THREADS`.** Polars allocates per-thread buffers and defaults to
  one thread per logical core. At 12 threads the feature build paged for 36
  minutes without writing a file; at 6 it finished in 18 minutes. This is the
  single most effective memory lever in the pipeline and it is an environment
  variable, not a config key.
- **`runtime.bootstrap_workers`.** The models are independent, so the bootstrap
  parallelises exactly — 4 h 49 on one core becomes roughly 1 h 15 on four, with
  bit-identical results (`tests/unit/test_fast_auc_pr.py`). Removed on 22 October
  2026, when the bootstrap moved to sufficient statistics and stopped needing it.

### What did not fit, and why it matters

`history_warmup_days` lets each partition read days *before* itself, compute
features over them, and write only its own — the fix for a one-day partition
having no past. It works, and it is set to **0** in the published run:

| warm-up | rows processed per split | outcome |
|---|---:|---|
| 0 | 19M | completes, 18 min |
| 1 | 38M | wrote `train.parquet`, then `MemoryError` on `val` |
| 2 | 57M | `MemoryError` before writing anything |

Three failures against the same wall is a hardware ceiling, not a bug to fix.
The obvious next optimisation — chunking the interval sweep by day, the way the
1-hour window permits — **would not be enough**: F3's `pair_global_rarity`
sorts the whole partition by pair key and F4 accumulates per-user circular
means, and both are *expanding*, not windowed. Chunking them requires carrying
state across chunks, which is a real piece of work for an uncertain payoff at
this memory budget.

**Lifting the limitation needs one of:** a machine with more RAM (16 GB makes
warm-up and multi-day partitions immediately affordable — the budget table
above already prices it), or stateful chunked featurisation for F3/F4.

**Correction, 22 October 2026: the stateful path now exists.** `features/incremental.py`
carries F3 counts and F4 sums across days and is tested equal, bit for bit, to the
whole-window computation; `build_features` uses it for every partition and writes one
file per day. Measured on the real days 0-2 on the 8 GB reference machine (peak RSS
sampled every 0.2 s, labels stubbed because `redteam.txt.gz` was not on disk):

| universe | events per day | peak RSS after 3 days | time per day |
|---|---:|---:|---:|
| remote LogOn | 5.8M to 6.6M | 4.70 GB | 29 to 51 s |
| all events | about 18M | allocation failure on day 0 | |

The carried state grows with the number of distinct pairs, so the peak rises slowly
with the length of the run (3.86, 4.20, 4.70 GB over days 0-2). The all-events
universe still needs a larger machine; the remote LogOn universe of the harmonised
protocol fits on this one.

## Pricing any other slice

`authbench.ingest.budget.lanl_budget(n_events=...)` takes an event count, so the
same table answers "what would ten days cost?". `authbench preflight --events N`
prints it, and exits non-zero when it does not fit — with the count that would.
