# Reviewer 2 (statistics and resampling): "Detection Under an Alert Budget"

Materials reviewed: main.pdf (23 pages), main.tex, scripts/paper_assets.py, src/authbench/evaluate/{stats_tests,budget,ledger}.py, docs/methodology.md, docs/limitations.md, run 20261022T102638Z (metrics_summary.json, budget_summary.json, pairwise_comparisons.json, bootstrap_user_clusters.json, fit_variance.json, positives_ledger.csv, day_sizes.csv, evaluation logs), review_round1.md. Scratch scripts: `C:\Users\NB03\AppData\Local\Temp\claude\C--Users-NB03-Videos-Test-Recherche\bd23f87f-e8de-433e-a1bb-8f74f1d4f079\scratchpad\review2\` (recompute.py, recompute2.py, recompute3.py, recompute4.py, check_bca.py, misc.py, userday_null.py).

## Summary

1. The statistical machinery is correctly implemented. I reproduced every number I could check: the exact random-ranking null, its quantiles, the Spearman coefficients and permutation p-values, the Holm decisions and the BCa intervals (the authors' `bca_interval` matches scipy to 1e-15).
2. The inference layer is weaker than the machinery. The abstract says ROC-AUC and first detection are "unrelated" (ρ = 0.03 and 0.15). That holds only for b*, a single-event statistic, and with n = 10 or 11 detectors it is absence of evidence (95% CI about [-0.62, 0.66]). With the median campaign budget, ρ(ROC-AUC) is 0.55 (harmonised) and 0.91 (temporal, p < 0.001).
3. The "PCA, VAE, autoencoder" trio depends on the fit. Under Holm over the 10 non-trivial detectors, all three survive only with PCA seeds 42 and 7. With seeds 13, 21 and 99, only the VAE survives. The VAE and the autoencoder were fitted once.
4. The one-sided b* "interval" is an artifact of bootstrapping a minimum: its upper end is always the 4th-smallest campaign budget. The user-day results are compared with one random draw (M0a), the same flaw round 1 fixed for events. M0a's user-day b* of 309 sits at the 95th percentile of its exact null.
5. Reproducibility is exemplary and the paper is honest about the one-operation limit. As a preprint it is nearly ready, needing text fixes only. For a top-tier venue the contribution is a single-operation case study, and the headline inferences need the changes below.

## Scores (1 to 5)

| Criterion | Score | Reason |
|---|---:|---|
| Soundness | 2 | Machinery correct; two headline inferences (the "unrelated" correlation and the trio) depend on the choice of statistic and of seed. Rises to 3 or 4 with text changes and the analyses in M1 to M3. |
| Presentation | 3 | Precise and well sourced, but dense. Several statistical caveats live only in docs/limitations.md and not in the paper (b* interval, M2a tie dependence of the null). |
| Contribution | 2 | Case study of one operation; the harness and exact null are the durable part. |
| Reproducibility | 5 | Every number regenerates from committed outputs; paper_assets.py cross-checks the exact null against the simulated one; the per-positive ledger allowed independent recomputation. |

## Recommendation

- **Top-tier security venue (USENIX Security, S&P, CCS, NDSS):** reject in current form (weak reject), with resubmission encouraged. The statistical machinery is not the reason. The reasons are the single operation, the dependence of the headline claims on one statistic and one fit, and a contribution that is mostly a harness. Fixing M1 to M4 would make the statistical part publishable; the external-validity limit needs a second dataset (e.g. OpTC, which Larroche 2026 harmonises).
- **arXiv/Zenodo preprint:** ready after text-only changes, none needing a new LANL run: rewrite the "unrelated" sentence (M1), qualify the trio by seed (M2), describe the b* interval as an order statistic (M3), and replace the single-draw user-day comparison with the exact null I computed below (M4).

## Strengths

- **Exact random-ranking null for b\* (Eq. 2).** Correct, cheap, and a real improvement over comparing with one random draw. `random_first_detection_survival` agrees with my independent log-gamma implementation, an exact rational computation, and a 200,000-draw simulation.
- **Bootstrap engineering.** Sufficient-statistic resampling via multinomial benign bins is exact in distribution for per-model AUC-PR. The jackknife over 98 account blocks plus about 3.6e8 benign events, grouped by bin, is exact. The BCa code reproduces scipy.
- **Resolution and Monte Carlo awareness.** `minimum_resamples_for_family`, an explicit floor flag, refusal of significance with fewer than 2 blocks, R = 10,000 (floor 2.0e-4 below Holm's first threshold of 6.4e-4).
- **Honest scope.** Sections 3.3, 12 and 13 state that the effective number of intrusions is one and that p-values describe stability under resampling of accounts.
- **Sensitivity analyses.** The source-user cluster bootstrap, five-seed refits and campaign-gap regrouping were all run and reported, which is rare.

## Fatal issues

None in the statistical machinery. M1 would be fatal to the abstract's second finding if left as worded, but it is fixable in text.

## Major issues

### M1. The "ROC-AUC and first detection are unrelated" claim is absence of evidence, and specific to b* (Abstract; Section 7.3; Table 5; Section 15)

- With n = 10 (harmonised) and n = 11 (temporal), ρ = 0.03 and 0.15 have approximate 95% CIs of [-0.62, 0.66] and [-0.51, 0.70]. "Unrelated" is not supported; "no evidence of association in this sample" is. Round 1 removed exactly this wording (R2 M2), and it has returned in the abstract.
- **The result depends on the summary chosen.** All of these come from the run's own budget_summary.json (Spearman with exact or 400,000-draw permutation p):

  | Summary of b(c) | Harmonised ρ(ROC) | ρ(AUC-PR) | Temporal ρ(ROC) | ρ(AUC-PR) |
  |---|---:|---:|---:|---:|
  | b* (paper) | 0.03 (p 0.95) | 0.10 (0.79) | 0.15 (0.67) | 0.54 (0.09) |
  | 10% quantile | 0.42 (0.23) | 0.60 (0.07) | 0.59 (0.06) | 0.83 (0.003) |
  | 25% quantile | 0.44 (0.21) | 0.72 (0.02) | 0.89 (0.001) | 0.89 (0.001) |
  | median | 0.55 (0.11) | 0.79 (0.009) | 0.91 (<0.001) | 0.91 (<0.001) |
  | log-area | 0.35 (0.33) | 0.67 (0.04) | 0.28 (0.41) | 0.60 (0.06) |

  This fits the paper's own mechanism (ROC-AUC is a mean-rank statistic, b* a top-rank one), but the abstract presents the b* result alone. A reader leaves believing ROC-AUC is uninformative about budgeted detection, which these numbers contradict for every summary except b* and log-area.
- **Leverage.** Leave-one-detector-out on the harmonised ROC-AUC ρ gives values from -0.33 (dropping M1u) to +0.27 (dropping PCA). The round-1 leave-one-out table (R2 M1) is no longer in the paper.
- **Seed.** Over the 625 combinations of the five seeds of PCA, Isolation Forest, HBOS and ECOD, harmonised ρ(ROC, b*) ranges 0.03 to 0.24 and ρ(AUC-PR, b*) ranges 0.10 to 0.42 (median 0.36). The reported values are the minimum of each range, because the reported PCA fit (seed 42) is the outlier: its AUC-PR is 1.23e-5 against 4.2 to 4.6e-5 for the other four seeds.
- **Campaign-weighted AP** (computed in metrics_summary.json as `auc_pr_campaign_weighted`, not reported) gives ρ with b* of 0.39 (harmonised) and 0.745 (temporal, p = 0.011).
- **Fix.** Pre-specify a family of budget summaries (b*, 10/25/50% quantiles, log-area). For each, report ρ with a CI, leave-one-detector-out and the seed range, and adjust across the family. Rewrite the abstract along these lines: "ROC-AUC did not rank the detectors by first detection (ρ = 0.03, 95% CI -0.62 to 0.66, n = 10), although it tracked the median campaign budget (ρ = 0.55 and 0.91)." Add campaign-weighted AP to Table 5.

### M2. Per-detector null probabilities: multiplicity and fitting variance (Abstract; Section 7.2; Tables 3, 4, 8)

- Section 7.2 discloses that the null probabilities are unadjusted, but the abstract selects the three smallest of 10 to 13 as "the detectors whose first detection a random ranking rarely matches". Holm-adjusted over the 10 harmonised non-trivial detectors: PCA 0.020, VAE 0.025, autoencoder 0.048. Over all 13: the autoencoder fails (0.065). In the temporal protocol over 11: M1 0.006, PCA 0.034, autoencoder 0.062 (fails).
- **Seed dependence.** PCA's harmonised null probability across the five seeds is 0.0020, 0.0013, 0.0159, 0.0137 and 0.0242. With seeds 13, 21 and 99, Holm over 10 keeps only the VAE. "PCA's position as the earliest harmonised detector is therefore a property of one fit" (Section 7.7) is the right sentence, but the abstract still headlines PCA at 20.
- **Single fits.** The VAE and autoencoder, the other two members of the trio, were fitted once (Section 12), so their seed sensitivity is unknown.
- **Only five seeds.** The range of five draws is expected to contain a sixth draw only 4/6 of the time, so Table 8 understates fit spread.
- **Support for b\* differs by detector.** The ledger shows the smallest campaign budgets: PCA 20, 300, 394, 403; VAE 27, 50, 58, 229; autoencoder 59, 162, 190. PCA's b* rests on one campaign followed by a 15-fold gap; the VAE's on three campaigns under 60. That is more informative than the null probability alone.
- **Fix.** Report Holm-adjusted null probabilities beside the raw ones. Use 20 or more seeds for the seeded detectors and at least 5 for the VAE, autoencoder and Euler (Euler was already run with 5 seeds in the fidelity study). Report b* and its null probability per seed as a median and range, and make the abstract's trio conditional on that. Print the 3 smallest b(c) per detector.

### M3. The b* bootstrap interval is a method artifact (Section 6.3; budget_summary.json; docs/limitations.md item 13)

- `ledger.resample_summaries` resamples accounts with ranks fixed. With K = 98 accounts, P(the k best accounts are all absent) ≈ e^(-k), which is below 0.025 only from k = 4. So the 97.5% point is the 4th-smallest account-level budget, by construction.
- Verified for every model I checked. Harmonised: PCA [20, 403] (b(c): 20, 300, 394, 403), VAE [27, 229], autoencoder [59, 528], Euler [203, 679], M1 [41,065, 298,235]. Temporal: PCA [246, 282], M1 [42, 1,036]. Each upper end is the 4th order statistic.
- The bootstrap is inconsistent for extremes (Bickel and Freedman 1981; Andrews 2000), which limitations.md admits. Yet Section 6.3 still calls it a "one-sided interval", and limitations.md calls it "an upper bound on how many alerts a day the first detection could need". It has no coverage interpretation.
- **Fix.** Drop the interval. Report the smallest 3 to 5 b(c), with how many distinct accounts they cover. If uncertainty for b* is wanted, use m-out-of-n subsampling, or present b* only together with the exact null and the seed distribution (M2).

### M4. Alert-unit results are compared with one random draw (Abstract; Section 7.4; Table 6; Appendix C)

Round 1 fixed this for events (R2 M3), but it returns for user-days. Because two campaigns of one account are more than 24 hours apart, (day, campaign) pairs are exactly the positive user-days (181 harmonised, 61 temporal). Eq. 2 therefore applies directly with N_d = user-days. My computation (userday_null.py):

- **Harmonised user-day b\*.** A random ranking reaches 6 with probability 0.056, 9 with 0.083, and 309 (M0a's realised value) with 0.95; the null median is 72. The user-day first detections of M2a (6), M2c and Euler (9) are therefore not distinguishable from random. M0a's 309 is an unlucky draw that makes every detector look better in Table 6.
- **Temporal user-day b\*.** P(b*_random ≤ 1) = 0.003, ≤ 2 = 0.006, ≤ 12 = 0.036; null median 226.
- **Harmonised user-day recall at 500.** Null expectation 3.0%, 95% range [0.6%, 5.6%]; M0a's 4.3% is one draw inside it. Euler (51.2%) and M2c (50.6%) are far outside, so the abstract's main user-day claim survives, but against the wrong reference.
- **Fix.** Use the exact unit-level null (expectation and range) instead of M0a in the abstract, Table 6, Appendix C and Figure 1. Add account-cluster intervals for the 51.2% and 50.6% (83 and 82 of 162 campaigns, from at most 98 accounts).

### M5. The intervals in Tables 3 and 4 understate the uncertainty the paper itself measured (Sections 6.3 and 7.7; bootstrap_user_clusters.json)

- The account-block bootstrap treats benign events as independent singletons. The source-user cluster bootstrap, which the paper runs, widens some intervals by an order of magnitude or more. Harmonised M2c: [1.08e-5, 2.83e-5] (block BCa) against [4.41e-6, 1.15e-3] (user clusters, 40 times the upper end); PCA [7.83e-6, 1.78e-5] against [6.38e-6, 7.00e-5]; autoencoder [2.55e-5, 6.85e-5] against [1.79e-5, 2.09e-4].
- The paper reports only the count of surviving significant pairs (54 of 78 and 53 of 78). Among the pairs lost are Euler against pair rarity, Euler against ECOD, and PCA against the autoencoder: these are the scientifically interesting ones. The surviving ones are mostly comparisons with floors or M1u, separated by orders of magnitude.
- **Fix.** Put the user-cluster interval in Tables 3 and 4, at least beside the block one, and treat it as primary since it respects the known dependence. Replace counts of significant pairs with a short list of pre-specified contrasts (for example, the best detector against each runner-up, and each family against its best competitor), with intervals under both schemes.

### M6. The R1 and R2 comparison confounds training data with fitting noise (Section 7.6; Table 7)

- Section 7.6 concludes that "the 50 red-team events among 133,093,586 training events are enough to move these detectors". But Isolation Forest's R1 to R2 change (98,938 to 55,094) is within its own temporal seed range (20,183 to 102,041; Table 8), and HBOS (8,572 to 8,987) is within its seed range (8,572 to 9,315).
- For the autoencoder, VAE and Euler, fitting variance is unknown, so the 494 to 1,565 change (with ROC-AUC 0.988 to 0.881) cannot be attributed to the 50 rows. Removing rows also changes the seeded 5M-row training subsample, so R2 is effectively a new random fit.
- The 1% screening threshold has no relation to the noise level.
- **Fix.** Compare R1 and R2 across matched seeds and report whether the R1 to R2 difference exceeds the seed spread. Otherwise state that no effect of the 50 events is detectable.

## Minor issues

1. **Monte Carlo SE of the p-value is understated** (stats_tests.py, around line 254, `p_value_mc_se = sqrt(p(1-p)/R)`). Since p ≈ 2t/R, the SE is about sqrt(2p(1 - p/2)/R): about √2 larger for small p, and 0.0097 rather than 0.0044 at p = 0.736. No Holm decision changes: the closest is harmonised p = 0.0102 against a threshold of 0.0056, 3.2 corrected SEs away. Fix the formula.
2. **Pairing is partial** (Section 6.3, "every detector evaluated on the same resamples"). Benign multiplicities are drawn per model from that model's own bins (stats_tests.py, around lines 407 to 420). Only the positive-block multiplicities and the benign total are shared. This is conservative for differences, but the text should say so (methodology.md already does).
3. **The pairwise test is still percentile inversion** (Eq. 3). Round 1 asked for studentised or BCa-based tests (R2 M5); only the per-model intervals became BCa. Since the jackknife already exists, BCa-inverted p-values for the differences are cheap. At minimum, name the test "percentile-interval inversion".
4. **68 of 69 harmonised significant pairs and 56 of 60 temporal ones are at the resolution floor** (p ≤ 2.0e-4). Say so in Section 7.7, so that readers know the p-values are bounds rather than measurements.
5. **M2a's null probability depends on tie order.** It is 0.040 under arrival order, but 0.0001 (b* = 1) to 0.183 (b* = 2,001) across its tie range. Give the bracket in Table 3 as for recall, or mark M2a's null as tie-dependent.
6. **The R@1e-4 correlation in Table 5 is quasi-definitional and unadjusted.** In the temporal protocol, 8 of 11 detectors tie at 0%, so ρ = 0.75 is the event "the three detectors with nonzero recall are the three earliest" (p ≈ 2/C(11,3) = 0.012). Across the six Table 5 tests, the smallest Holm-adjusted p is 0.078. State both.
7. **The zero-recall bound was dropped** (round 1 R2 m7). Restore it with its caveat. One-sided 95% Clopper-Pearson upper bounds: 1.8% for 0/162 campaigns, 3.0% for 0/98 accounts, 5.6% for 0/52, 6.0% for 0/48. All are conditional on this operation. The VAE's 7/162 at 500 has a CP interval of [1.8%, 8.7%] before clustering.
8. **Campaign recall curves (Figure 2) have no bands.** Account-cluster pointwise bands are cheap from the ledger (round 1).
9. **"Every interval endpoint and p-value carries its Monte Carlo error" (Section 6.3)** is true of the JSON but not of the paper. Add one sentence giving the largest Monte Carlo range: the temporal PCA upper endpoint, [6.06e-3, 7.13e-3].
10. **Stale documentation.** methodology.md still says "conf/eval/default.yaml sets 10,000 (2,000 in the published runs)" and "the published LANL runs used the per-event draw". The run at bd833eb used the bins scheme with R = 10,000. It also says "52 campaigns map to 50 accounts", while the logs show 48 positive blocks (50 accounts include 2 unlabelable events). Align these.
11. **Detectors are not independent units for Spearman.** PCA, the autoencoder and the VAE are reconstruction errors on the same 25 features; M2a and M2c share F3. Say that the permutation p tests exchangeability within this convenience set, not a population of detectors.
12. **Campaign definition.** The gap sensitivity (Section 7.8) reports campaign counts but not how recall at 500 and b* change across gaps. b* is gap-invariant (a minimum over all positives), but recall is not. One line per gap in an appendix would close this.

## What I recomputed, and whether it matched

| Quantity | Paper | Recomputed | Match |
|---|---|---|---|
| Harmonised null quantiles of b* (2.5/50/97.5%) | 251, 6,859, 36,412 | 251, 6,859, 36,412 | yes |
| Temporal null quantiles | 1,806, 49,373, 261,368 | 1,806, 49,373, 261,368 | yes |
| P(random b* ≤ 500), harmonised / temporal | 4.9% / 0.7% | 4.925% / 0.699% | yes |
| Null probabilities, all 13 models x 2 protocols (Tables 3, 4), e.g. harmonised PCA 0.002, VAE 0.003, autoencoder 0.006, Euler 0.020, M2a 0.040, HBOS 0.120; temporal M1 0.001, Euler 0.200 | as printed | 0.0020 (exact rational 0.002018), 0.0027, 0.0059, 0.0203, 0.0401, 0.1197; 0.0006, 0.2004 | yes (log-gamma, rational and simulation methods agree) |
| Exact null formula against simulation (toy days, 200,000 draws) | n/a | e.g. 0.2728 against 0.2727 | yes |
| Spearman ρ, Table 5 | 0.03/0.10/0.54; 0.15/0.54/0.75 | 0.030/0.103/0.544; 0.145/0.536/0.749 | yes |
| Permutation p, Table 5 (harmonised by full enumeration of 10! = 3,628,800) | 0.946/0.786/0.111; 0.673/0.095/0.013 | 0.946/0.785/0.111; 0.673/0.094/0.012 | yes, within Monte Carlo error |
| Holm significant pairs (78) | 69 / 60 | 69 / 60 | yes |
| User-cluster significant pairs | 54 / 53 | 54 / 53 (file) | yes |
| Holm decisions within 2 MC SE of their threshold | not stated | none (none even with the corrected SE) | n/a |
| BCa intervals, Tables 3 and 4 | as printed | identical to metrics_summary.json; authors' `bca_interval` equals scipy BCa (1.2253945763519207, 2.5130900080720213) on a test case; grouped-jackknife weights give the same interval | yes |
| Account blocks used | 98 / 48 accounts | logs: "over 98 positive blocks", "over 48 positive blocks" | yes |
| Fit variance ranges, Table 8 | e.g. PCA [13, 243], Isolation Forest temporal [20,183, 102,041] | same | yes |
| b* bootstrap upper end | "one-sided interval" | equals the 4th-smallest b(c) in every case checked | artifact (M3) |
| User-day exact null | not given | harmonised P(≤ 6) = 0.056, P(≤ 9) = 0.083, P(≤ 309) = 0.95; recall at 500 null 3.0% [0.6%, 5.6%] | new (M4) |
| ρ with other budget summaries, seeds, leave-one-out, campaign-weighted AP | not given | see M1 | new (M1) |

Not recomputable from the committed outputs: the AUC-PR bootstrap resamples themselves (they need per-model benign bins, which are not saved). I verified that code by reading it and testing its BCa and jackknife components.

## Status of the round-1 statistical points

| Round-1 point | Status now |
|---|---|
| Title and abstract generalised (R2 M1) | Resolved: case-study title, Section 13 scope. |
| "ROC-AUC carries no information": absence of evidence (R2 M2) | **Regressed**: the abstract says "unrelated" (M1). |
| Contrast driven by one detector; leave-one-out table (R2 M1) | **Not carried over**: no leave-one-out table. M1u is now the leverage point (harmonised ρ -0.33 without it). |
| b* compared with one random draw (R2 M3) | Resolved for events (exact null, verified). **Reappears for alert units** (M4). |
| b* without uncertainty; publish b(c) (R2 M3) | Partly resolved: ledger published (excellent); quantiles and log-area with intervals in JSON but not in the paper; b* interval is an artifact (M3); no bands on the curves. |
| Floor p-values; R = 2,000 cannot certify Holm (R2 M5) | Resolved: R = 10,000, floor 2.0e-4 < 6.4e-4; no decision is Monte Carlo fragile. |
| Significance under one operation; benign clustering; fitting variance (R2 M4) | Partly resolved: cluster bootstrap and five seeds run; not propagated into the headline intervals or claims; the VAE, autoencoder and Euler fitted once; "significant" still used (M2, M5). |
| Percentile-inversion test (R2 M5) | Partly resolved: per-model intervals are BCa (verified); the test is unchanged (minor 3). |
| Event-weighted against campaign-weighted AP (R2 M6) | Computed, not reported; it changes Table 5 (M1). |
| "Stratified" mislabel (R2 m2) | Resolved in code and documentation. |
| Clopper-Pearson zero-recall bound (R2 m7) | **Dropped** from this version (minor 7). |
| Fisher test for single-day change (R2 m8) | Not applicable (the passage no longer exists). |
| Accounts with two campaigns resampled separately (R2 m5) | Resolved: account blocks in both bootstraps (verified in code and logs). |

## The three changes that would most raise acceptance odds

1. **Rebuild the metric-agreement analysis (Section 7.3, Table 5, abstract) as a pre-specified family**: b*, budget quantiles, log-area, event-weighted and campaign-weighted AP, and R@1e-4. Give each ρ a CI, a leave-one-detector-out range and a seed range, and adjust across the family. Then state the finding the data support: ROC-AUC does not rank detectors by first detection in this sample, but does track the median campaign budget.
2. **Propagate fitting variance into every headline number.** Use 20 or more seeds for the seeded detectors and at least 5 for the VAE, autoencoder and Euler. Report b*, null probability (Holm-adjusted) and AUC-PR per detector as a distribution over seeds. Redo Table 7 on matched seeds. Retire the b* percentile interval in favour of the smallest few b(c) (M2, M3, M6).
3. **Make the uncertainty reported match the dependence measured.** Use user-cluster intervals in Tables 3 and 4, named contrasts instead of counts of significant pairs, and the exact unit-level null in place of the single M0a draw for user-days and edge-hours (M4, M5).
