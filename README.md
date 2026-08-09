# AuthBench

**How much does deep learning actually buy you in authentication-log anomaly
detection?** A leak-free, alert-budget-constrained benchmark on the LANL
Comprehensive Multi-Source Cyber-Security Events dataset.

[![CI](https://github.com/SaniAdamou14/AuthBench/actions/workflows/ci.yml/badge.svg)](https://github.com/SaniAdamou14/AuthBench/actions/workflows/ci.yml)

> A SOC cannot triage more than a few dozen alerts per day per analyst. Most
> published anomaly-detection results are reported at operating points no
> analyst could ever use. AuthBench does not propose a new model — it
> measures, under a strictly temporal, leak-free protocol and a realistic
> daily alert budget, how much of the published deep-learning advantage over
> well-built heuristics actually survives.

**No LANL results yet.** The pipeline runs end to end, but only on the demo
sample so far. Everything below that shows numbers shows *demo* numbers, and
they are not a finding about anything. See
[State of the project](#state-of-the-project).

## Try it in under a minute

No 12 GB download required. `data/demo/` ships a small synthetic sample
(versioned in git) shaped like LANL — same 9-column schema, same `?` null
sentinel, machine accounts, per-user home machines — and carrying three
complete red-team lateral-movement campaigns.

```bash
make install     # uv venv + uv pip install -e ".[classical,dev]"
make demo        # ~40 s
```

Or without `make`:

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[classical,dev]" --python .venv
.venv/bin/authbench demo          # .venv\Scripts\authbench demo on Windows
```

This runs ingest → clean → label → temporal split → features F1–F4 →
M0/M1/M2/M3 → evaluation, and writes to `reports/`:

| Artifact | What it is |
|---|---|
| `reports/tables/demo_results.json` | every metric, per model |
| `reports/tables/pairwise_comparisons.json` | pairwise AUC-PR tests, Holm-Bonferroni corrected |
| `reports/tables/campaign_summary.csv` | one row per red-team campaign |
| `reports/figures/campaign_recall_vs_budget.png` | the headline figure |
| `reports/data_quality.json` | null rates, cardinalities, counted drops |

Everything under `reports/tables/` and `reports/figures/` is pipeline output
and is gitignored — regenerated, never hand-edited.

### What the demo sample is

| | |
|---|---|
| Auth events | 37,826 |
| Period | 14 days |
| Red-team events | 9, in 3 campaigns |
| Positive rate | 2.4 × 10⁻⁴ (≈340× LANL's — it is a smoke test, not a scale model) |
| Split | train days 0–7, val 8–10, test 11–13 |

One campaign lands in each partition, on purpose. A validation window with no
positives silently reduces M1's weight calibration to a single arbitrary
random draw — `tests/unit/test_demo_sample_integrity.py` is the guard, and
`authbench data generate-demo` regenerates the sample.

## The real dataset

```bash
export AUTHBENCH_LANL_EMAIL=you@example.org   # required, see below
authbench data download --dataset lanl
authbench data to-parquet data/raw/auth.txt.gz --out-dir data/interim/auth \
  --expected-rows 1051430459
dvc repro
```

LANL serves `cyber1` from behind a click-through data-use form rather than a
static URL, so `authbench data download` submits an email plus a usage
statement to obtain a token (`ingest.download.fetch_lanl_fence_token`) and
builds the file URLs from it. The address is read from `--email` or
`AUTHBENCH_LANL_EMAIL` and is never written into `conf/`.

LANL publishes no per-file checksums. The first successful download prints
the SHA-256 it computed; record it under `sha256:` in
`conf/dataset/lanl.yaml` and every later run verifies against it and aborts
on mismatch.

| | |
|---|---|
| Source | LANL Comprehensive, Multi-Source Cyber-Security Events (Kent, 2015) |
| Period | 58 consecutive days |
| Auth events | 1,051,430,459 |
| Red-team events | 749 (737 unique after dedup) |
| Positive rate | ≈ 7.1 × 10⁻⁷ |

## Architecture

```
LANL auth.txt.gz + redteam.txt.gz
  -> ingest   (token + resumable download + SHA-256 verify)
  -> parse    (typed schema, machine accounts, explicit nulls, counted drops)
  -> Parquet, partitioned by day, ZSTD
  -> label    (exact quadruplet join, campaign grouping)
  -> split    (temporal train/val/test, leak-free by construction)
  -> features (F1 event, F2 history, F3 novelty, F4 temporal, F5 graph, F6 sequence)
  -> feature store (Parquet, versioned by config hash)
  -> models   (M0 floors, M1 rules, M2 stats, M3 classical, M4 deep, M5 graph)
  -> evaluate (AUC-PR, recall@budget, campaign recall, TTD, bootstrap CIs)
```

Two things are *fitted*, and both are fitted on the training split alone: the
night window (`features.temporal.calibrate_night_window`) and the F1 category
frequency tables (`features.event.fit_frequency_encoding`). Refitting either
per split is a leak that no downstream check catches — it just makes the
numbers better. `tests/unit/test_frequency_encoding.py` holds that line.

Full design rationale: [`AuthBench_Specification.md`](AuthBench_Specification.md).
Implemented protocol: [`docs/methodology.md`](docs/methodology.md).

## Saying "outperforms" only when it is earned

Confidence intervals and pairwise tests come from **one** campaign-stratified
resampling pass shared by every model, so the intervals and the comparisons
sit on the same sampling distribution and each difference is paired.

Two floors sit under this and both produce results that look exactly like a
genuine "no difference" finding while having nothing to do with the models:

- **Degenerate resamples.** A ranking metric over zero positives is
  undefined, but scikit-learn returns 0.0 — so on such a resample every model
  ties, and those ties land in both tails of a two-sided test. With one
  campaign in the test split, ~37% of resamples are degenerate and every
  p-value is floored near 0.74. They are discarded and redrawn;
  `n_degenerate_discarded` is reported, because a high count is itself a
  finding about the split.
- **Bootstrap resolution vs. Holm.** A percentile bootstrap cannot report a
  p-value below `2/(R+1)`, and Holm's strictest threshold is `alpha/n_pairs`.
  For the 8-model catalog that needs `R ≥ 1119`; the config's original 1000
  could never have produced a single significant result.
  `stats_tests.minimum_resamples_for_family` computes the bound and warns
  when it is not met.

## State of the project

Being explicit about this is part of the point of the benchmark.

**Runs end to end today**

- Ingest, parse, label, temporal split with a mechanical leakage guard.
- Features F1 (event), F2 (history), F3 (novelty), F4 (temporal).
- Models M0a random, M0b always-fail, M1 rules (7 rules, weights calibrated
  on validation), M2a pair-rarity, M2b PCA reconstruction, M3a Isolation
  Forest, M3b ECOD/HBOS.
- Evaluation, in two registers the report keeps separate:
  - *operational* — event and campaign recall at daily alert budgets,
    time-to-detection per campaign, AUC-PR with campaign-stratified CIs;
  - *literature-comparable* — ROC-AUC, global precision@k, recall at a fixed
    FPR, each reported for placement against published numbers, not ranking.
- Pairwise AUC-PR comparisons across the whole model family, Holm-Bonferroni
  corrected. The report generator can only emit the word "outperforms" on the
  significant branch (`render_comparison_sentence`).

**Implemented and unit-tested, but not yet wired into the reported tables**

- SHAP alert cards (`explain/`) — needs the `explain` extra.
- F5 graph / F6 sequence features and the M4 deep / M5a graph models — these
  need the `deep` / `graph` extras and are not part of the default DVC stage.

**Not implemented**

- The R1 (unsupervised) / R2 (semi-supervised) training regimes. The key
  exists in `conf/split/*.yaml`; nothing reads it yet.
- M5b GNN link prediction — raises `NotImplementedError` on purpose rather
  than shipping a decorative implementation.
- `reports/paper/`. There is deliberately no `report` stage in `dvc.yaml`: a
  stage depending on a `main.tex` that does not exist breaks `dvc repro` at
  the last step for everyone.

## Development

```bash
make lint        # ruff check + ruff format --check, on src tests scripts
make typecheck   # mypy --strict on src
make test        # pytest with coverage
make demo        # the full pipeline on the demo sample
```

CI runs all four on every push and PR, plus a coverage gate of 85% on
`evaluate/` and `features/`, and uploads `reports/` as an artifact.

Requires Python ≥3.11, <3.13. Optional extras: `classical` (PyOD), `deep`
(torch), `graph` (networkx/node2vec), `explain` (shap), `dashboard`
(streamlit), `tracking` (mlflow/dvc), `dev`, `all`.

## Limits

- A single real dataset (LANL, 2015, one enterprise network) — transferability
  to modern cloud/MFA environments is not established.
- Red-team labels are a partial ground truth: an unflagged event may still be
  a true compromise never caught at the time.
- Campaign grouping has no canonical definition; the gap threshold is a
  published, sensitivity-tested parameter rather than a hidden constant.
- No adaptive adversary: LANL's red team was not evading these specific models.

Full discussion: [`docs/limitations.md`](docs/limitations.md).
Scope of use: [`docs/ethics.md`](docs/ethics.md).

## Citing the dataset

> Kent, A. D. (2015). *Comprehensive, Multi-Source Cyber-Security Events*.
> Los Alamos National Laboratory. https://doi.org/10.17021/1179829

## License

Code: Apache 2.0 (see `LICENSE`). LANL and CERT data are not redistributed —
only the download scripts are, under the terms of each dataset's own license
(see [`docs/ethics.md`](docs/ethics.md)).
