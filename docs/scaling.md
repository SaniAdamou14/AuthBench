# Scaling: what a full LANL run costs, and what actually fits

Run `authbench preflight` (or `make preflight`) for these numbers computed on
*your* machine. This document explains where they come from and what to do
when they do not fit — which, on any laptop, they do not.

## Where the numbers come from

Every per-event size below is **measured on the demo sample**, not guessed.
`authbench demo` writes a 90-column ZSTD feature store; dividing by its event
count gives the constants in `authbench.ingest.budget`:

| Quantity | Measured | Basis |
|---|---|---|
| Raw LANL-shaped text | 68.7 B/event | `data/demo/auth_demo.txt` |
| Interim ZSTD Parquet (17 typed columns) | 11.3 B/event | `data/interim/auth` |
| Feature store, F1–F4, ZSTD Parquet (90 columns) | 94.1 B/event | `data/processed/features` |
| Design matrix in RAM (21 float64 columns) | 168 B/event | `features.MODEL_FEATURE_COLUMNS` |

Two adjustments, both stated rather than hidden:

- **×1.4 cardinality penalty.** The demo has fewer distinct users and machines
  than LANL's 12,425 and 17,684, and dictionary-encoded Parquet compresses
  better the fewer distinct values it holds. Every Parquet estimate is
  inflated by this factor, so the figure errs high.
- **Gzip ratio 0.08**, assumed. LANL publishes no file size, and nobody can
  compute the ratio without the file. `authbench data download` replaces the
  assumption with the server's `Content-Length` before writing a byte, and
  refuses to start if the volume cannot hold it.

## Full LANL — 1,051,430,459 auth events

| Stage | Disk added | Peak RAM |
|---|---:|---:|
| `download` (auth.txt.gz + redteam.txt.gz) | ~5.8 GB | 0.1 GB |
| `to_parquet` → `data/interim/auth` | ~16.6 GB | ~2 GB |
| `label_split_features` → `data/processed/features` | ~138.5 GB | ~8.6 GB |
| `train_eval` (design matrix) | ~0.1 GB | **~439 GB** |
| **Total** | **~161 GB** | **~439 GB** |

Disk is cumulative: no stage deletes its predecessor's output, because DVC
needs the inputs of every stage it might re-run.

`train_eval`'s 439 GB is not a typo. `pl.read_parquet` loads each split whole,
and PCA / ECOD / HBOS each copy the design matrix: 830M train+test events ×
21 float64 columns × ~3 copies.

**The download is the cheap part.** Fetching LANL costs about 6 GB. Turning it
into a benchmark costs 161 GB of disk and, as currently written, more RAM than
a workstation has.

## What fits on 21 GB free / 16 GB RAM

- **Disk** caps the run at roughly **100 million events** (~9.5% of LANL).
- **Memory** caps it at roughly **39 million events** (~3.7%, about 2.2 days).

Memory binds first, and 2.2 days cannot carry a 30/10/18-day temporal split
with red-team campaigns in all three partitions. So on this machine, **the
protocol described in the README cannot be run on the full dataset as the code
stands.** That is a fact about the code and the hardware, not a reason to
quietly report a smaller number as though it were the published one.

## The three honest ways forward

### 1. Give it the hardware

161 GB of free disk and, for `train_eval` as written, several hundred GB of
RAM. An external drive covers the disk — every path is a CLI flag or a
`conf/paths` value, so `data/` can live anywhere:

```bash
authbench data download --out-dir E:/authbench/raw
authbench data to-parquet E:/authbench/raw/auth.txt.gz --out-dir E:/authbench/interim/auth
```

The memory figure is not solved by a bigger disk. It needs option 3.

### 2. Run a documented subset, and say so

A contiguous slice of days, with the split rescaled and **every number in the
report labelled with the slice it came from**. Cheap to do, and defensible as
long as the label never comes off. Its cost is real: fewer campaigns in the
test split means fewer independent observations for the campaign-stratified
bootstrap, and below two campaigns significance stops being estimable at all
(`stats_tests.MIN_CAMPAIGNS_FOR_SIGNIFICANCE` — this is why `authbench demo`
reports 0/21 significant rather than 21/21).

```bash
authbench preflight --events 39000000     # check the slice fits first
```

### 3. Make `train_eval` out-of-core

The one that produces a real full-dataset result. Four changes, in order of
how much they buy:

1. **Fit on a bounded sample, score everything.** M3a already subsamples
   internally (`max_samples=256`); M2b/M3b do not need 550M rows to estimate a
   covariance or an ECDF either. Fitting on a few million rows and scoring the
   full split is standard, and it turns the 439 GB into a bounded constant.
2. **Score in day-partitioned chunks.** Scoring is a map: one day at a time,
   keeping only the slim evaluation frame.
3. **Keep only what the metrics need.** Alert-budget metrics need the top-K
   scores per day (K = the largest budget) plus every malicious event.
   Time-to-detection needs the same. Neither needs 326M benign scores resident.
4. **Compute AUC-PR from a score histogram.** It is exactly the count of
   negatives ranked above each positive; a fine histogram of negative scores
   plus all positive scores reproduces it to well under the width of its own
   confidence interval, in constant memory.

The campaign-stratified bootstrap survives this: it already runs on the slim
frame, and `stats_tests.CampaignBlocks` keeps benign events as one index array
rather than 300 million one-element ones.

## Making the estimate honest for your own data

`authbench.ingest.budget.lanl_budget(n_events=...)` takes an event count, so
the same table answers "what would a 10-day slice cost?". `authbench preflight
--events N` prints it.
