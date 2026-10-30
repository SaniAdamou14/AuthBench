# Review round 2, Reviewer 1 (evaluation methodology for ML in security)

Paper: "Detection Under an Alert Budget: A Leak-Free Case Study of Authentication Anomaly Detectors on the LANL Data, with a Reproduction of Euler" (main.pdf, 23 pages, run 20261022T102638Z, code at bd833eb).

## 1. Summary

The paper presents AuthBench, a harness that enforces a temporal split in code, tests row-locality of every detector to the last bit, ranks events (and edge-hours, user-days) within each day, and reports per-campaign detection budgets with an exact random-ranking null for the first-detection budget b*. Thirteen detectors (floors, rules, rarity scores, PCA, PyOD outlier detectors, AE, VAE, an Euler reimplementation) are run on LANL under Larroche's (2026) harmonised protocol (train hours 0 to 41, test 57 days, 702 positives, 162 campaigns) and a day-based temporal split. At event level no detector catches a campaign at 10 alerts/day and none exceeds 9.6% at 500; in user-days Euler and a source-novelty score reach about 50% at 500. Over 10 to 11 detectors, ROC-AUC does not rank-correlate with b*. The authors' Euler code, rerun, reproduces the published AUC but not the published AP. The paper frames itself as a case study of one intrusion.

## 2. Scores (1 to 5)

| Criterion | Score | Comment |
|---|---:|---|
| Soundness | 3 | The measurement layer is unusually careful (I'd score it 4 to 5). The inferential layer has problems: the headline claims rest on an extreme, seed-unstable statistic, n = 10 detectors, unequal alert-unit costs and design decisions made after the test labels were seen. |
| Presentation | 3 | Precise and honest, but very dense. The abstract alone carries about 20 numbers. Several sections read as a lab notebook (Appendix D, Section 11). |
| Contribution | 2 | It adds engineering value and a useful Euler reproducibility finding. The empirical findings mostly confirm Larroche (2025, 2026), Garchery and Granitzer (2020) and King and Huang (2022) on one operation. |
| Reproducibility | 4 | This is exemplary: committed run outputs, CI regeneration of every number, exact provenance and cost. Two points hold it back. Raw data came from a mirror, with content hashes checked on only 3 of 58 days, and the AE, VAE and Euler were fitted once. |

## 3. Recommendation

- **Top-tier venue (USENIX Security, S&P, CCS, NDSS): Reject.** The work is not wrong. The reason is that one dataset and one intrusion cannot carry a top-4 empirical claim, and the novelty beyond Larroche (2026) is mostly infrastructure. A resubmission with a second dataset (OpTC, which Larroche 2026 already harmonised) and published detectors run inside the harness would be a different, stronger paper.
- **Preprint (arXiv/Zenodo): ready after minor fixes.** Fix the abstract's "unrelated" wording (M1), the false "same budget of 20 candidates" (m1), the omitted M0b floor in the unit tables (M4), and the "identical test set" phrase (m5). Also add a one-paragraph disclosure of post-hoc design choices (M3). None of these needs a new run.
- **Realistic venue tier.** As it stands, the right home is a workshop: ANUBIS at ESORICS (where Larroche 2026 appeared), AISec, CSET or WOOT-style measurement venues. A journal is also realistic: Computers & Security, after major revision. RAID, ACSAC, EuroS&P and DIMVA would need major revision and at least the first of the three changes in Section 9.

## 4. Strengths

1. **Leakage engineering is above the norm in this literature.** Specific points:
   - a code guard on partition order;
   - a bit-exact row-locality check, re-run on the positives of every scored day (`evaluate/statistics.py`, `add_day`);
   - ECOD refitted so it is not transductive (a real pitfall in PyOD's default);
   - rarity denominators anchored to the start of the day;
   - M1 normalisation moved from the test partition to the last training day.
2. **The exact random-ranking null for b* (Eq. 2) is correct and useful.** I recomputed it from `day_sizes.csv` with the paper's `random_first_detection_survival`:
   - harmonised: median 6,859, 95% range [251, 36,412];
   - temporal: median 49,373, range [1,806, 261,368];
   - P(b* <= 500) is 4.9% and 0.7%.

   It replaces the single-draw comparison criticised in round 1.
3. **Tie handling is explicit.** The paper states a label-free tie order, reports a bracket, and counts contenders. This is the right way to report coarse scores.
4. **The Euler section is a genuine contribution to reproducibility.** It runs the authors' own code, reconciles the test sets edge by edge (525 vs 518 vs 514 anomalous edges) and documents undocumented choices (10 negatives per edge, early stopping on AUC + AP, DropEdge). The finding that the published AP (0.0155) is not reached by the authors' code (0.0040 to 0.0080 over five runs) is worth publishing.
5. **The paper is honest about scope.** Section 3.3 and the abstract both state that this is one operation, 95.5% from one host. Section 13 restricts the claims.
6. **The fixed-threshold variant (Section 6.4) is a useful corrective.** It answers the retrospective daily top-k criticism and shows that 41 h calibrations do not transfer.

## 5. Fatal issues (for a top-tier venue)

### F1. One intrusion, no second dataset: the evidence base cannot support a venue-level claim

Section 3.3 states that the effective number of independent intrusions is one, and 97.5% of temporal positives share one source host. Every comparative statement depends on the accounts of that single operation:
- detector A detects earlier than B;
- the alert unit matters as much as the detector;
- ROC-AUC does not track b*.

The bootstrap can only measure stability under resampling of those accounts (as Section 12 admits). For a top-4 venue this is fatal regardless of how well the harness is built. Larroche (2026) already harmonised OpTC alongside LANL, so a second dataset is available.

**Fix:** add OpTC (or another labelled enterprise log with several intrusions) under the same harness, or reposition the paper as a tool/artifact paper.

### F2. The headline comparative claim depends on the choice of the most fragile summary statistic

The abstract claims "over the non-trivial detectors, ROC-AUC and the first-detection budget are unrelated (Spearman rho = 0.03 and 0.15)". The conclusion claims "ROC-AUC did not indicate which of them detect earliest". Three problems undermine this:

1. **b* is a minimum over 162 campaigns, i.e. the rank of one event.** It is the least stable quantity the harness produces. The paper itself shows this:
   - PCA's b* moves from 13 to 243 over five seeds (Table 8);
   - Isolation Forest's moves from 20,183 to 102,041 in the temporal protocol;
   - the AE, VAE and Euler were fitted once.

   Correlating any metric with a near-random-order statistic over n = 10 detectors has little power: |rho| must exceed about 0.65 for p < 0.05.
2. **The finding flips with stabler budget summaries.** I recomputed Spearman correlations from `budget_summary.json` and `metrics_summary.json`, using the same detector sets as `paper_assets.py`, with a positive value meaning agreement:

| Metric vs | b* (paper) | q10 budget | median campaign budget |
|---|---:|---:|---:|
| ROC-AUC, harmonised | 0.03 | 0.42 | 0.55 |
| ROC-AUC, temporal | 0.15 | 0.59 | 0.91 |
| AUC-PR, harmonised | 0.10 | 0.60 | 0.79 |
| AUC-PR, temporal | 0.54 | 0.83 | 0.91 |

   With the budget at which 10% or 50% of campaigns are detected, the threshold-free metrics do track the operational ordering, strongly so in the temporal protocol. The "unrelated" result is therefore a property of b*, not of budgeted detection.
3. **The detector set includes M1u, an anti-detector.** M1u has ROC-AUC 0.244 because rules R3, R4 and R5 point the wrong way (Table 9). It anchors one end of both rankings. Without M1u, rho(ROC-AUC, b*) becomes -0.33 (harmonised) and -0.14 (temporal). The coefficient is driven by which detectors are included, not by a stable relation.

**Fix:**
- Drop "unrelated" from the abstract. Report rho for b*, q10, q25 and q50, and for the partial area of the campaign-recall curve over 1 to 500 alerts/day on a log axis.
- Use the seed-median b* for every detector, with multi-seed fits for the AE, VAE and Euler.
- Report the leave-one-detector-out sensitivity that round 1 promised. It is absent from this version.
- State the power of n = 10.

If the claim survives this, it is a finding. If not, the paper should say that ROC-AUC tracks the bulk of the campaign-recall curve but not its extreme left end, which is a more defensible and still useful message.

## 6. Major issues

### M1. Absence of evidence is again stated as evidence (abstract, Section 7.3, Section 15)

Round 1 flagged "ROC-AUC carries no information" as absence of evidence, and the response says the wording was removed. The abstract now says "unrelated". That is the same claim, with p = 0.946 from 10 points.

**Fix:** "we find no rank correlation (rho = 0.03, n = 10, p = 0.95); the study cannot detect correlations below about 0.65", plus F2.

### M2. Unequal and label-informed tuning in the temporal protocol is under-reported (Section 5, Section 12, Table 4)

- **The tuned detectors all used the operation's labels.** PCA, Isolation Forest, HBOS and M1 were all chosen on validation AUC-PR, which uses the labels of the same operation (266 positives, 261 of them on day 8). Section 12 only discusses M1.
- **The choice moves results by large factors.** In `temporal/tables/tuning.json`, PCA's validation AUC-PR is 6.7e-5 at its default (0.95) and 5.5e-3 at the chosen 0.5, an 80x difference. Appendix D attributes PCA's b* moving from 1,569 to 246 to this choice. So the temporal protocol's best non-M1 operational result (PCA, 9.6% at 500) is label-informed by the same intrusion.
- **The AE, VAE, Euler, ECOD and the rarity scores receive no tuning at all.** This is the unequal tuning pitfall (Arp et al. P6/P7) in the opposite direction from what the text implies.
- **The "same budget of 20 candidates" claim is false** (Section 5). PCA has 6 candidates, Isolation Forest 9, HBOS 12 and M1 20 (`tuning.json`; `pipeline/tuning.py` caps at the grid size).

**Fix:**
- Say "at most 20 (6, 9, 12, 20)".
- Flag every label-tuned temporal result in Table 4.
- Either give the AE, VAE and Euler a comparable validation budget, or report the temporal tuned detectors at their defaults beside the tuned version.
- Treat the harmonised protocol as the only label-free evidence.

### M3. Post-hoc design on the test operation (data snooping; Sections 4.3, 7.4, 8 "The source side", Appendix D)

The catalogue and the alert units were changed between versions in response to observations about the test labels:
- source-side features M2c and R8 were added after round-1 reviewers noted that 97.5% of positives share one source host;
- edge-hour and user-day units were added;
- rarity anchoring was changed.

The temporal test days 11 to 13 have now been evaluated by at least two code versions (Appendix D), and they lie inside the harmonised test window. The abstract's most positive result ("a source-novelty score reach[es] about half of the campaigns at 500 [user-days] per day") comes from a detector added because of what the test labels showed. "Leak-free" in the title refers to code-level temporal leakage, and it is accurate in that sense. It is not leak-free in the data-snooping sense (Arp et al. P1/P3).

**Fix:**
- Add an explicit "design history" paragraph listing what was changed after seeing which results.
- Qualify the title (e.g. "temporally leak-free").
- Ideally freeze the catalogue and confirm on data not used during design (OpTC, or LANL red-team days not inspected).

### M4. The alert-unit comparison is not cost-equal, has no unit-level null, and omits the floor that does well

The abstract claims "the alert unit matters as much as the detector" (Sections 7.4 and 6.1, Table 6, Appendix C). This comparison has several problems:

- **Unequal workload.** 500 user-days is about 3% of a day's 11,636 to 19,358 user-days. Each user-day bundles many events, so an analyst investigating a user-day does far more work than for one event. The staffing model prices both at 10 to 20 minutes.
- **Biased random floor.** The comparison uses the single M0a realisation (4.3%), not an exact null. M0a pooled by max is also size-biased: busy user-days get higher maxima. This is the same single-draw issue round 1 raised for events.
- **No tie bracket for units, and a non-neutral tie order.** Units carry no tie bracket, and units are tie-ordered by the time of their first event. That order is label-free but not neutral.
- **The omitted floor reveals the problem.** The M0b always-fail floor is left out of Table 6 and Appendix C (the generator uses `non_trivial() + ["M0a_random"]`). In `metrics_summary.json`, in user-days:
  - harmonised: M0b reaches 7.4% at 500 with b* = 16 user-days, better than PCA (18), VAE (19), Isolation Forest (26) and the AE (30);
  - temporal: M0b reaches 17.3% at 500, matching the AE and pair rarity and beating M1 (11.5%), HBOS, Isolation Forest and ECOD.

  The temporal random floor's user-day b* is 12 (Appendix C). So user-day first detections barely separate detectors from floors.

**Fix:**
- Add M0b to both unit tables.
- Derive the exact uniform-random null for units (the same hypergeometric argument, with units per day).
- Compute tie brackets for units.
- Report recall against workload (events or estimated analyst minutes reviewed), not against nominal k.
- Add account-level bootstrap intervals for the unit recalls.
- Add a source-computer-day unit. With 95.5% of the activity on C17693 it will likely catch the operation at a budget of a few units. That would show plainly how much the unit result is a single-host artefact, and the paper should say so.

### M5. The primary metric has no inferential support; the inference targets the metric the paper argues against (Section 6.3)

- **Pairwise testing covers AUC-PR only.** The paired bootstrap, BCa intervals, Holm over 78 pairs and the user-cluster variant are all computed for AUC-PR.
- **The budget metrics get only point estimates in the tables.** No pairwise test is given for campaign recall at k, b(c) quantiles or the partial area of the recall curve. b* has only a one-sided percentile interval, and it is not printed.
- **The campaign-weighted AP is computed but not reported.** It was promised in round 1, and it is present as `auc_pr_campaign_weighted` in `metrics_summary.json`.

**Fix:** move the paired account-level bootstrap to a stable budget statistic (q10, or the log-area over 1 to 500 already in `budget_summary.json`), print its intervals in Tables 3 and 4, and report the campaign-weighted AP.

### M6. Fitting variance is measured for the cheap detectors only; the headline detectors are single fits (Sections 6.3, 7.7, Table 8)

- **The detectors that carry the headline numbers are not refitted.** The VAE (best at 500 events), the AE (b* = 59), and Euler (best AUC-PR, best user-day recall) are each fitted once.
- **The reported PCA fit is not the typical one.** For PCA, the reported seed 42 has the lowest AUC-PR of the five seeds (1.23e-5 against about 4.5e-5) and the second-best b* (20). The seed median is 137, with null probability 0.014 rather than 0.002 (my recomputation).
- **The variance is partly self-inflicted.** The harmonised training set has only 10.7M events, yet fitting uses a 5M seeded subsample. That is a needless source of the variance the paper reports.

**Fix:**
- Five seeds for the AE, VAE and Euler.
- Report the seed-median as the point estimate, with the range.
- Fit the harmonised detectors on all 10.7M training rows.

### M7. The benchmark's Euler is weaker than the authors' Euler (Sections 5, 9, Table 10)

On the same data the reimplementation is about 3 AUC points and half the AP below the authors' code. All of the paper's Euler claims are therefore claims about a weaker Euler:
- event b* 203;
- 51.2% at 500 user-days;
- the R1/R2 shift.

**Fix:**
- Label M5c "Euler-style (reimplementation)" throughout.
- Better, run the authors' code inside the harness. Their predictor produces edge scores that can be mapped to events per snapshot.
- Or test the undocumented choices (10 negatives per edge, early stopping on AUC + AP, DropEdge) that the paper lists but does not test.

### M8. No published budgeted detector is run inside the harness, so the harness is never calibrated against prior numbers (Section 10, Table 11)

Table 11 places published numbers beside the case study and correctly says they are not comparable. A reviewer will ask why at least one comparison was not made comparable. Two would be cheap:
- M2c is essentially Garchery and Granitzer's "known source pc" rule. Run it under their split (days 1 to 8 / 9 to 13, system accounts removed, user-day budgets) and check whether CR-1000 of about 0.254 is reproduced.
- DECADES or UltraLMD++ could be run under Larroche's own protocol.

Round 1 also promised Bowman's rule, and it is not in the catalogue.

**Fix:** reproduce at least one published budgeted number with the harness. This validates the harness and gives Table 11 a comparable row.

### M9. Operation-level detection is the natural question for one intrusion, and it is not reported

With effectively one intrusion, a SOC's question is "on which day, at budget k, is the intrusion first noticed?", not "what fraction of 162 account-phases are noticed?". From `positives_ledger.csv` (harmonised, event level, arrival order):
- at 100 alerts/day the AE alerts on a red-team event on day 1, the operation's first day;
- at 500/day the AE and Euler both alert on day 1, and PCA and the VAE on day 8.

Reading "no detector exceeds 9.6% of campaigns" without this changes the operational message substantially. Both of these day-1 detections also fall on test day 1, which holds only hours 41 to 47 (1.65M events against about 6.3M on a full day; `day_sizes.csv`). A fixed daily k is therefore a larger hourly budget on that day.

**Fix:**
- Report operation-level time-to-first-detection per budget and unit.
- Handle the partial day 1 explicitly: merge it with day 2, express budgets per hour, or report results with day 1 excluded.

## 7. Minor issues

- **m1 (Section 5).** "the same budget of 20 candidates" is false (6, 9, 12, 20). See M2.
- **m2 (Section 7.3, Table 5).** Recall at FPR 1e-4 is exactly zero for 8 of 11 temporal and 7 of 10 harmonised detectors. The rho = 0.75 is mostly a contrast of three non-zero detectors against tied zeros. State this; the "close to definitional" remark is fair.
- **m3 (Sections 7.3 and 6.2).** The paper notes that AUC-PR pools days a daily queue keeps apart, and 28 of the 57 harmonised test days hold no positive. The obvious control, the mean per-day AP over days with positives, is not reported. Add it.
- **m4 (Table 6, Section 7.5).** The fixed-threshold runs compute campaign recall at the realised volume (`operational.fixed_threshold[*].campaign_recall`), but the paper reports only volumes. For example, temporal PCA calibrated for 50/day raises 134/day on average and catches 7.7% of campaigns. Harmonised ECOD calibrated for 500 raises 2,391/day and catches 11.1%. Report recall against realised mean volume; this is the most deployment-faithful number in the run.
- **m5 (abstract, Section 9).** "our reimplementation reaches 0.9570 and 0.00270 on an identical test set" reads as identical to the authors' evaluation, which scores 514 of 14,414,775 edges against 525 of 14,400,984. Write "on the same data (our snapshot alignment)".
- **m6 (Section 9).** Say explicitly that the authors' code ran on CPU with torch 1.13.1+cpu and PyG 2.2.0. Their table V counts (45,871,390 events, 17,685 nodes) match neither their own loader's output (15,610 nodes) nor yours, so the published AP may come from a different preprocessing. "On the published data" in the abstract is stronger than "on the public data processed by their released loader".
- **m7 (Section 7.2).** Per-detector null probabilities: with 13 detectors, Bonferroni at 0.05 is 0.0038. Only PCA (0.002) and the VAE (0.003) pass, and only in this fit. Say "rarely matches" only after adjustment.
- **m8 (Section 3.1).** Content hashes were verified on 3 of 58 days, with a total row count for the rest. State it as partial verification. A per-day row count table against LANL's published per-day counts, if any exist, would help.
- **m9 (Section 3.3).** 97.5% is computed over red-team lines in the test window (277/284, `redteam_concentration.json`), not over the 282 labelled positives. Say so.
- **m10 (Section 7.8).** M2a's daily anchoring creates about 217k tied contenders and a [0, 61.7%] bracket, so M2a's event-level result is now uninterpretable. Break its ties with a secondary label-free signal (e.g. src-dst rarity, or the unanchored k/n) and keep the anchored version as the tie-bracket illustration.
- **m11 (Section 6.1, staffing).** The 10 to 500 range rests on a vendor survey and a 2026 preprint, and the paper acknowledges both. Hopper's figure of fewer than 9 alerts/day is a better-sourced anchor for the low end. Make 10 to 50 the primary operating range.
- **m12 (Section 4.1).** The temporal tuning metric is validation AUC-PR, the metric the paper argues is misaligned with budgets. Add an ablation that tunes on validation campaign recall at 100, or on the log-area.
- **m13 (Section 6.3).** The one-sided b* interval's upper end exists in `budget_summary.json` (`ci_high`) but is not shown. Show it, or drop the interval language.

## 8. Presentation

- **The abstract is too dense.** About 20 numbers make it unreadable. Keep three findings: event-level zeros, the unit effect with its caveats, and the Euler reproduction. Move the rest.
- **Some material reads as a lab notebook, not a paper.** Appendix D (earlier code version), the "previous version of this report" wording in Section 4.1, and Section 11 (synthetic circularity control) belong in the artifact. Section 11 is a sound observation but tangential to LANL.
- **Naming is inconsistent across the paper, code and run outputs.** The code and RUN.md use `M3b_ecod` and "M3b ECOD" while the paper uses M3c. RUN.md's p-values (simulated, e.g. Euler 0.0221) differ from the paper's (exact, 0.020). The paper is right, but state that the run files hold simulated values.
- **Sentences need breaking up.** Many sentences chain three or four results; split them, and put one claim per sentence in Sections 7.1 to 7.4.
- **Figures 1 and 2.** Add the seed range as error bars on b* in Figure 1, and the unit-level curves to Figure 2.
- **Venue formatting.** The contents page is fine for a preprint; remove it for a venue.

## 9. Claims I verified (file and value)

All files are under `reports/runs/20261022T102638Z/` unless stated otherwise.

**Paper tables and null**

| Claim | Source | Status |
|---|---|---|
| Table 3 and Table 4 (AUC-PR, CIs, ROC-AUC, R@1e-4, @50, @500, b*) | `harmonised/metrics_summary.json`, `temporal/metrics_summary.json` | Match |
| Null p in Tables 3 and 4 | Recomputed exactly with `random_first_detection_survival` from `day_sizes.csv` | Match (e.g. PCA 0.0020, VAE 0.0027, AE 0.0059, Euler 0.0203, temporal M1 0.0006) |
| Null quantiles and P(b* <= 500) | Same recomputation | Match: 6,859 [251, 36,412]; 49,373 [1,806, 261,368]; 4.9% and 0.7% |
| Table 5 Spearman coefficients | Recomputed independently | Match exactly (0.03/0.10/0.54 and 0.15/0.54/0.75) |

**Alert units, thresholds and ties**

| Claim | Source | Status |
|---|---|---|
| Table 6 unit recalls (Euler 51.2%, M2c 50.6%, M0a 4.3%, Euler 13.6% at 50) | `metrics_summary.json` (`alert_units`) | Match |
| Fixed thresholds: PCA 149/8,222; VAE 167/9,151; M2a 58,396; HBOS 4; temporal PCA 26, M1 14; M2a and M2c 0; Isolation Forest 1 | `operational.fixed_threshold` | Match |
| Tie brackets: 32.1%/61.7% harmonised, 32.7%/67.3% temporal; 217,692 contenders | `metrics_summary.json` | Match |

**Variance and resampling**

| Claim | Source | Status |
|---|---|---|
| PCA seeds 13 to 243; Isolation Forest 718 to 1,641 and 20,183 to 102,041 | `*/tables/fit_variance.json` | Match |
| 69/78 and 60/78 Holm-significant pairs | `pairwise_comparisons.json` | Match |
| 54 and 53 significant under source-user clusters | `bootstrap_user_clusters.json` | Match |
| Gap regrouping: 224/185/157/143 and 79/60/50/48 | `campaign_gap_sensitivity.json` | Match |

**Data and rules**

| Claim | Source | Status |
|---|---|---|
| 683/715 (95.5%) from C17693 | `diagnostics/redteam_concentration.json` | Match |
| Rule ROC-AUCs (R3 0.15, R5 0.17, R1 0.70, harmonised); machine accounts 66.8% of benign | `diagnostics/m1u_rules_harmonised.txt` | Match |
| Harmonised positive rate 1.96e-6 (702 / 358,864,145) | Day sizes | Match; daily positives sum to 702 |

**Euler**

| Claim | Source | Status |
|---|---|---|
| Table 10 configurations, 5 seeds each | `euler_fidelity_{all,ntlm,ntlm_vocabulary,ntlm_code}.json` | Match (0.9521/0.00035, 0.9570/0.00270, 0.8261/0.00011, 0.9590/0.00125) |
| Authors' code AUC 0.9879, AP 0.0058 | `euler_original/results.txt` | Match; AP 0.0040 to 0.0080 over runs; sd = se x sqrt(5) = 0.00178 |

**Facts I found that the paper does not state**

| Finding | Source |
|---|---|
| Tuning grid sizes are 6, 9, 12 and 20, not 20 each | `temporal/tables/tuning.json` |
| M0b user-day results (7.4% and 17.3% at 500; b* 16 and 31) are omitted from the unit tables | `metrics_summary.json` |
| The AE and Euler b* fall on partial day 1 | `positives_ledger.csv` |
| Operation-level first-alert days at each budget (see M9) | `positives_ledger.csv` |

## 10. Claims I could not verify

- Published external numbers: Larroche 2026 Table 5 (32.27 vs 0.09 +- 0.04), DECADES DR-1K 0.081, ADSAGE CR-1000 0.254/0.255, UltraLMD++ Rec@10 0.1788, and King and Huang's table V/VI values (0.9906, 0.0155, 518, 45,871,390). I did not access the sources.
- End-to-end regeneration of `generated/*.tex` by `scripts/paper_assets.py`. I did not run it, because it writes into the repository. The CI step exists in `.github/workflows/ci.yml`.
- Raw-data integrity and the row-locality and leakage unit tests. The data is not local and I did not run the test suite.
- The edge-by-edge reconciliation counts in Section 9 (14,416,149; 1,374; 3 unscored anomalous edges). The output files exist, but I did not rerun the scripts.
- The claim that every number in `temporal/` and `harmonised/` came from bd833eb (module import timing). The argument in RUN.md is plausible, but I cannot check it independently.

## 11. Status of round-1 points

| Round-1 point | Status | Note |
|---|---|---|
| Title and abstract generalise from n = 5 | Partly | Retitled as a case study with 13 detectors. "Leak-free" overreaches given M3, and "unrelated" is back in the abstract. |
| "ROC-AUC carries no information" (absence of evidence) | Not resolved | Reworded to "unrelated", the same claim (M1, F2). |
| Contrast driven by M1 | Partly | The harmonised protocol removes label tuning. The promised leave-one-detector-out table is absent, and M1u now drives the coefficient (F2). |
| Prevalence-sensitivity non sequitur | Resolved | Replaced by the top-weighted vs mean-rank argument. |
| Low-FPR metric unused | Resolved | Reported and correlated (but see m2). |
| Novelty overstated; prior art | Resolved | Contribution reframed; prior work cited. |
| "We show" for an identity | Resolved | |
| b* compared with one random draw | Partly | Exact null for events. User-days still use the single M0a draw (M4). |
| b* extreme statistic; b(c) not published | Partly | Ledger and curves published. Inference still centred on b* (F2, M5). |
| Floor p-values; R = 2000 | Resolved | R = 10,000; the Monte Carlo error is reported. |
| One operation; benign clustering; fitting variance | Partly | User-cluster bootstrap done. Multi-seed only for the classical detectors (M6). |
| Percentile-inversion test; BCa | Resolved for AUC-PR | Not for budget metrics (M5). |
| Campaign-weighted AP | Partly | Computed in the JSON, not reported. |
| "Stratified" in docstring | Resolved | `stats_tests.py` now says "not a stratified one". |
| Accounts with two campaigns | Resolved | Resampling by account. |
| Daily top-k retrospective | Mostly resolved | Fixed-threshold variant added; its recall is not reported (m4). |
| M1 normalisation look-ahead | Resolved | Empirical CDF on the last training day. |
| Training warm-up claim | Resolved | |
| Static architecture test | Resolved | Described as static. |
| Split vs Larroche 41 h | Resolved | Harmonised protocol is primary. |
| M1 calibrated on the same operation | Partly | Label-free in harmonised. In temporal, PCA, Isolation Forest and HBOS are also label-tuned, and this is not flagged (M2). |
| M2a tier ordering | Partly | Anchoring now makes M2a tie-dominated (bracket [0, 61.7%]) (m10). |
| No source-side feature | Resolved but post-hoc | M2c and R8 added after seeing the labels (M3). |
| Event universe and units | Resolved | LogOn universe; edge-hour and user-day units (but see M4). |
| Published budgeted results not used | Partly | Table 11 lists them; none is reproduced (M8). |
| No deep or graph detector | Resolved | AE, VAE and Euler added (Euler is weaker than the authors' code, M7). |
| Budget range and staffing model | Partly | Model given; inputs weak (m11). |
| Unused red-team days 14 to 29 | Resolved | Harmonised test covers them. |
| Campaign gap g | Mostly resolved | Campaign counts reported; recall under each gap is in the JSON, not the paper. |
| Bowman's UA rule | Not resolved | Not in the catalogue. |

## 12. The three changes that would most raise acceptance odds

1. **A second dataset and calibration against prior work.** Run the frozen harness and catalogue on OpTC under Larroche's (2026) harmonisation. Reproduce at least one published budgeted number with the harness (the known-source-PC rule under Garchery and Granitzer's split, or DECADES/UltraLMD++ under Larroche's). This addresses F1, M3 and M8 and turns a case study into a benchmark result.
2. **Rebuild the inference around stable, operational budget statistics.** Use the seed-median over five fits for every learned detector, including the AE, VAE and Euler (on the full 10.7M training rows). Use a paired account-level bootstrap on q10 or the log-area over 1 to 500 alerts/day. Report operation-level time-to-first-detection per budget. Re-examine the ROC-AUC/AUC-PR claim against those, with power stated. This addresses F2, M1, M5, M6 and M9.
3. **Make the alert-unit result fair and auditable.** Use the exact uniform null for units, unit tie brackets, and the M0b floor. Report recall against workload (events or analyst minutes), add a source-computer-day unit that exposes the single-host artefact, and add a disclosed design-history paragraph. This addresses M3 and M4 and protects the paper's most quotable positive finding from the obvious objection.
