# LANL run, wider window with warm-up (15 September 2026)

The evaluated configuration of the report (`reports/paper/main.tex`, version 2). It supersedes
the single-day run of [`../RUN.md`](../RUN.md) as the primary result: it gives every partition a
real past (two warm-up days) and tests on more campaigns. The single-day run is kept, unchanged,
as the configuration it replaced.

## Provenance

| | |
|---|---|
| Code | commit `17e897c` (merge of the tie-order, checkpointing and ECOD-exclusion fixes) |
| Outputs committed | commit `c641c4c`, 15 September 2026 |
| Machine | rented Linux server, about 30 GB of RAM (provider and cost not recorded) |
| Source | LANL `auth.txt.gz`, SHA-256 `9c6b0cc2…f672`; days 0–13 converted |
| Split | train days 0–7 · validation 8–10 · test 11–13, `history_warmup_days: 2` |
| Fitting | M2b, M3a, M3b on a seeded sample of 5,000,000 training events; M1 weights on validation labels |
| Excluded | **M3b ECOD**, by name before fitting (`AUTHBENCH_EXCLUDE_MODELS=M3b_ecod`): its whole-split scoring was OOM-killed on this machine. `skipped_models.json` is empty because, at the time, it listed only failures during scoring; the pipeline now records exclusions there too. |
| Bootstrap | 2,000 campaign-level resamples, Holm correction over 21 pairs |
| Feature build | 21 minutes |

## Split

From [`split_summary.json`](split_summary.json):

| Split | Days | Events | Positives | Campaigns | Positives from C17693 |
|---|---|---:|---:|---:|---:|
| Train | 0–7 | 133,093,586 | 50 | 11 | 42 (84.0%) |
| Validation | 8–10 | 51,687,105 | 266 | 45 | 261 (98.1%) |
| Test | 11–13 | 54,690,768 | 282 | 52 | 275 (97.5%) |

Day 11 has no red-team event; the test positives are on days 12 (207) and 13 (75).

`split_summary.json` was computed after the run, from the labelled partitions this code version
produced, with `label.redteam_join.split_label_summary`, the function `build_features` now calls
on every run. Its event counts match the ones logged at run time exactly, and its 282 test
positives match the run's own `recall_at_fixed_fpr` (1/282 = 0.003546 for M2b).

**Read the last column before any interval.** 50 of the 52 test campaigns start from one source
host. The campaigns are the bootstrap's resampling unit, so the intervals describe variability
across the accounts and phases of one intrusion, not across intrusions. Larroche (2026) reports
the same concentration for the whole dataset.

## Results

| Model | AUC-PR [95% CI] | ROC-AUC | Camp. @10 | @50 | @100 | @500 | Budget for 1st detection |
|---|---|---:|---:|---:|---:|---:|---:|
| M0a random *(floor)* | 5.05e-6 [2.37e-6, 9.54e-6] | 0.488 | 0% | 0% | 0% | 0% | 85,070 |
| M0b always-fail *(floor)* | 5.16e-6 [2.52e-6, 8.98e-6] | 0.496 | 0% | 0% | 0% | 0% | 5,505,419 |
| M1 rules | 4.69e-4 [1.88e-4, 9.37e-4] | 0.563 | 0% | 0% | 0% | 0% | **999** |
| M2a pair rarity | **7.75e-4** [2.57e-4, 1.66e-3] | **0.941** | 0% | 0% | 0% | 0% | 2,852 |
| M2b PCA | 3.50e-5 [2.01e-5, 5.78e-5] | 0.930 | 0% | 0% | 0% | 0% | 1,569 |
| M3a Isolation Forest | 1.92e-5 [7.91e-6, 3.86e-5] | 0.876 | 0% | 0% | 0% | 0% | 141,522 |
| M3b HBOS | 3.32e-5 [1.26e-5, 6.91e-5] | 0.884 | 0% | 0% | 0% | 0% | 104,963 |

- **No campaign is detected at any operational budget, whatever the tie order**: every
  `campaign_recall_tie_bracket` is `[0, 0]`. M0b, the only coarse score, had 413,567 tied events
  competing at each budget and still detects nothing under the most favourable order.
- **The budget for first detection separates what the zeros do not**, read against its exact
  distribution under a random ranking (median 49,373, central 95% range [1,806, 261,368]). Share
  of random rankings that do as well: rules 1.4%, PCA 2.2% (both beat chance), pair rarity 3.9%
  (inside the range), HBOS 77.1% and Isolation Forest 86.3% (indistinguishable from chance). The M0a value is one
  draw of that distribution, not a threshold.
- **Orderings.** Spearman against the budget for first detection: ROC-AUC −0.10, AUC-PR 0.70,
  recall at FPR 1e-4 0.87 (exact p 0.95, 0.23, 0.10). Without M1, ROC-AUC and AUC-PR both give
  0.80. Five models; descriptive only.
- **Significance (AUC-PR).** 18 of 21 differences significant after Holm, all 18 at the
  resolution floor (Holm-adjusted p ≤ 0.0210). Not significant: M0a vs M0b, M2b vs M3b HBOS,
  and **M2a vs M1** (Holm-adjusted p = 0.333).

The paper's numbers, tables and figure are generated from these files by
`scripts/paper_assets.py`.
