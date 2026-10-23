# Review round 3, reviewer R2: evaluation methodology and statistics

Scope: `reports/paper/main.tex` (report v2), `generated/*.tex`, `scripts/paper_assets.py`, `src/authbench/evaluate/*`, `src/authbench/features/novelty.py`, `src/authbench/label/redteam_join.py`, and the run outputs under `reports/runs/20261022T102638Z`. I did not read the earlier review rounds. Verification scripts were run with the repo's `.venv` on the committed run outputs and on `data/interim/auth` (days 0 to 13 only; days 14 to 57 are not on this machine). I did not modify any repository file except this one.

Labels used below: **[verified]** means I recomputed the result or read the code path. **[suspected]** means the reasoning is sound but I could not run the check.

---

## 1. The evaluation design in my own words

The test partition is ranked one day at a time with a row-local scorer. Each day, the top-k events (or edge-hours, or user-days, each scored as the maximum of its events) are taken as alerts. Every red-team "campaign" (one account's events with gaps of 24 h or less) receives the smallest daily rank any of its events reaches. Campaign recall at k is the empirical CDF of those ranks. The headline statistic is the first-detection budget b*, the minimum of those ranks over all campaigns. Each b* is compared with its exact distribution under a uniformly random within-day permutation, with Holm correction across detectors. AUC-PR gets an account-block bootstrap (BCa, 10,000 resamples, Holm over 78 pairs). A user-cluster bootstrap and a five-seed refit serve as sensitivity analyses. There are two protocols: a label-free "harmonised" one (41 h of training, 57 test days, remote LogOn only) and a "temporal" one (days 0–7/8–10/11–13, four detectors tuned on validation labels). The labelled data are one intrusion, about 95% of it from one host.

## 2. Verdicts

**(a) Master's thesis / research project in CS: minor revision.** This is well above the usual Master's standard in rigour. Leakage control is enforced in code and tested. The null distribution is exact and correct, and I confirmed it by Monte Carlo. Ties are handled explicitly. The paper reports its own fragilities: the partial first day, fit variance, the design history, and one intrusion from one host. The numbers regenerate from the committed outputs, and every number I recomputed matches. Before a thesis examiner signs off, three things need correcting. (i) The user-day headline claim needs a trivial activity baseline. A one-line, label-free "distinct sources/destinations per user-day" ranking matches or beats every detector in that unit (issue M1). (ii) Two factual statements about how campaigns and user-days are keyed are wrong (M6). (iii) The abstract should stop presenting single-fit, single-event results without qualification (M2).

**(b) First-/second-year PhD paper at a peer-reviewed security/ML venue: major revision.** At a top-tier security venue I would reject it in its current form. The harness is careful, but the empirical contribution is one operation on one dataset. Its evidential weight sits on extreme order statistics (b*, first day noticed) of single fits. The one finding that looks positive, the user-day lift for Euler and source rarity, does not survive a simple activity/fan-out baseline. The random-ranking null used for that unit is misspecified, as the paper's own floors show. The "ROC-AUC tracks the middle" claim is significant only in the label-tuned temporal protocol, and it is partly mechanical. Uncertainty is quantified carefully for AUC-PR, which is not a headline metric, but not for campaign recall at k, which is. The paper would become a solid evaluation paper with: unit-level baselines and a size-aware null; seed replicates for every detector; intervals on the budget metrics; and above all a second dataset with several intrusions (OpTC is already harmonised by Larroche).

## 3. Strengths

- **The temporal split is enforced, not just described.** There is a guard on partition timestamps, an import-architecture test, and causal windows (`L137`). Every scored day re-checks bit-exact row-locality on the positives (`statistics.py:287-292`). I verified that the code does this. This is better practice than most published LANL work.
- **The exact random-ranking null for b* is correct.** Eq. 2 (`L198`) is the right survival function for independent within-day uniform permutations. `budget.py:299-312` implements it as `prod_i (1 - x/(N-i))`, which is algebraically identical, and correctly returns 0 when x > N − p. A Monte Carlo check matched it to three decimals [verified]. The quantiles and p-values in the paper reproduce from `day_sizes.csv` [verified].
- **Ties are handled explicitly** (`budget.py:39-71`; `statistics.py:315-324`). The tie order is label-free and stated, there is an exact unfavourable bracket and an achievable favourable one, and the number of contenders is reported.
- **The resolution floor of the bootstrap p-values is disclosed** (68/69 and 56/60 significant pairs sit at the floor [verified]). The resample count is chosen so Holm over 78 pairs is resolvable (`stats_tests.py:38-48`).
- **The BCa interval is implemented correctly**, with an exact weighted jackknife over positive blocks and benign score bins (`stats_tests.py:85-129, 330-357`). Resamples with no positive are redrawn, not scored 0.
- **The paper avoids a bootstrap interval for b***, correctly noting that the bootstrap is inconsistent for a minimum (`L217`). It offers the exact null, the next campaign budgets and the seed distribution instead.
- **Threats to validity are reported candidly**: the design history (`L141`), the single intrusion (`L108`, `L539`), the partial first day (`L115`, `L290`), and fit variance (`L399`).
- **I reproduced the harmonised universe exactly from raw parquet.** For days 1–13 my independent filter (LogOn, src ≠ dst, day 1 from hour 41) gives exactly the events, positives and user-days per day in `day_sizes.csv` [verified]. This is good evidence that the pipeline does what the text says.
- **Euler test sets are reconciled edge by edge** (`L471`). This is unusually thorough.

## 4. Major issues (ranked)

### M1. The user-day results lack a trivial baseline, and their "exact" null is misspecified [verified]

*Where:* abstract `L56` ("reach about half of the campaigns at 500 per day, against 3.0% for a random ranking"); `L204`; `L336`; Table 6 (`L338-354`); `L475`; scope `L557`; conclusion `L565`. Code: `statistics.py:341-347` (a unit scores the maximum of its events) and `paper_assets.py:564-604` (the null treats every user-day as equally likely to enter the top k).

*Why it matters.* Under the null H0 "the event ranking ignores labels", max-aggregation does not give a uniform ranking of user-days. Users with more events, more failures, or more fan-out rank higher by construction. The paper's own floors show this. The always-fail floor M0b reaches 7.4% (harmonised) and 17.3% (temporal) at 500 user-days, outside the null's 95% ranges [0.6, 5.6]% and [0.0, 7.7]%. The random floor M0a has temporal user-day b* = 12, null p = 0.036 (`table_units_temp.tex`). A detector that ignores labels is "significant" against this null.

I ranked user-days by simple label-free daily counts, using the paper's own unit key (`src_user`) and arrival tie order:

| Ranking of user-days | Temporal @50 | Temporal @500 | Temporal b* | Harmonised, days 1–13 only, @50 / @500 (as share of all 162 campaigns, so a lower bound) | Harmonised b* (days 1–13) |
|---|---|---|---|---|---|
| distinct destination computers | 25.0% | **61.5%** | 5 | **≥ 24.1% / ≥ 40.7%** | 3 |
| distinct source computers | 23.1% | **69.2%** | 5 | ≥ 14.2% / ≥ 39.5% | 2 |
| event count | 5.8% | 21.2% | 4 | ≥ 1.9% / ≥ 8.6% | 21 |
| best detector in the paper | 9.6% (M2c) | 26.9% (PCA, Euler) | 1 (AE) | 13.6% / 51.2% (Euler) | 9 |
| random-ranking "exact null" | 0.3% | 2.9% | median 226 | 0.3% / 3.0% | median 72 |

In the temporal protocol, a fan-out count beats all thirteen detectors at 500 user-days by 2.5×. In the harmonised protocol, counting only the 108 campaigns active on days 1–13, the destination-count ranking already detects 66 of 162 campaigns at 500/day and 39 at 50/day. Euler reaches 13.6% (22 campaigns) at 50/day over the whole test. I could not score days 14–57, so for 500/day the harmonised comparison is a lower bound. In the five-point gap between 40.7% and Euler's 51.2%, this baseline is still being counted against the full 162 campaigns. Compromised user-days are high-fan-out days: median 27 vs 11 distinct destinations in temporal day 12.

Related [suspected]: M2c's user-day score is itself an activity proxy. For a new (user, source) pair, `user_src_rarity` = −log(1/(n+1)) = log(n+1), where n is the user's event count before the day (`novelty.py:56-72, 88`). Its maximum over a user-day therefore ranks users with a new source by their historical volume.

*Fix:*
1. Add label-free unit baselines (event count, distinct destinations, distinct sources, failures) to Table 6 and the abstract comparison.
2. Replace the uniform-unit null with a null matched to the aggregation, for example M0a's event scores max-aggregated (simulate many hash seeds). Alternatively, use a conditional permutation null that shuffles "positive" status among user-days of similar size.
3. Rewrite the user-day sentences in the abstract, `L336`, `L475`, `L557` and `L565`. A defensible statement is that "user-day ranking by any of these detectors adds little or nothing over counting fan-out", unless the harmonised days 14–57 say otherwise.

### M2. Headline intrusion-level and first-detection claims rest on single fits and single events [verified]

*Where:* abstract `L56` ("the autoencoder notices the attack on its first day at 100 … two other detectors notice it a week later at 50"); `L290`; `L309`; conclusion `L565`. Seeds: `harmonised/tables/fit_variance.json`.

*Why.* b* and the "first day noticed" are each the rank of a single event. PCA's seed-42 b* is 20. Across five seeds its b* values are 20, 13, 159, 137 and 243 [verified]. So "PCA notices the intrusion on day 8 at 50 alerts per day" holds in **2 of 5 fits**, and its null p ranges from 0.001 to 0.024. The autoencoder, the VAE and Euler, which carry the other headline claims, were fitted once (`L219`). On this evidence the abstract's "two other detectors notice it a week later at 50" is a property of one seed of PCA and one unreplicated VAE fit.

The abstract also says the autoencoder detects day 1 "only because that day is partial". That is a causal claim the paper does not test. The exact null already conditions on N₁ = 1.65 M. Removing day 1 changes the set of positives, not just the day's size. Hours 41–47 are also a particular time of day, right after the training window. The partial-day effect is plausible but confounded.

*Fix:*
- Run ≥ 5 seeds for AE, VAE and Euler, and report b* and the first-noticed day as distributions: median, range, and the fraction of seeds that detect at k.
- In the abstract, state results as "in m of 5 fits".
- As a sensitivity analysis, add a volume-normalised budget (k proportional to N_d, i.e. a fixed fraction of each day) so the partial day does not need a verbal argument.
- Soften "only because" to "on a partial day; without it, 528".

### M3. "ROC-AUC tracks where most campaigns are found" is significant only in the label-tuned temporal protocol, and is partly mechanical [verified]

*Where:* abstract `L56`; `L313`; Table 5 (`L315-332`); conclusion `L565`; scope `L557`.

*Why.*
- **Harmonised.** ρ(ROC-AUC, median b(c)) = 0.55 with Holm p = 0.734, and ρ(AUC-PR, median) = 0.79 with Holm p = 0.100. Nothing is significant in the protocol the paper calls "the label-free evidence" (`L543`).
- **Temporal.** The significant result (ρ = 0.91, Holm p = 0.004) comes from 11 detectors, 4 tuned on validation labels of the same operation, and 2 positive days.
- **The link is partly mechanical.** ROC-AUC is one minus the mean normalised rank of the positives, and the median of b(c) is a quantile of a function of the same ranks. Agreement in "the middle" is expected by construction. The informative finding is the decoupling at the extreme.
- **The first-detection claim is absence of evidence.** "Not which detector finds the first" rests on ρ ≈ 0.03 to 0.15 with n = 10 to 11. `L313` correctly calls this absence of evidence, and b* is noisy within a detector (M2), which biases ρ toward 0. The abstract and conclusion still word it as a finding.

*Fix:*
- State the harmonised non-significance in the abstract.
- Note the mechanical relation.
- Phrase the first-detection result as "no detectable association (n = 10)".
- Better: correlate seed-median b* rather than single-fit b*, since attenuation from b*'s own noise is a known bias.

### M4. The headline budget metrics carry no uncertainty; the inferential effort goes to AUC-PR [verified]

*Where:* `L211-219`; Tables 3, 4 and 6; `L286`; `L336`.

*Why.*
- **The bootstrap tests the wrong metric.** The full block bootstrap and the 78-pair Holm family are spent on AUC-PR. Those values are around 10⁻⁵, the paper does not use them for conclusions, and "significant" differences between, say, 1.2e-5 and 5.5e-5 have no operational meaning.
- **The headline numbers have no interval at all.** These are campaign recall at k (events and user-days), the b(c) quantiles, log-area and the intrusion day. Yet the run already computes account-block bootstrap intervals for the quantiles and log-area (`harmonised/tables/budget_summary.json`, e.g. Euler median b(c) 3,592 [2,652, 5,845]); they are simply not reported.
- **Small percentages hide tiny counts.** 0.6% is one campaign of 162, and 4.3% is seven. The text compares 41.4% with 51.2% (`L336`) as if the difference were informative.
- **The effective sample size is one intrusion.** The paper says so in `L539`. But Holm-adjusted p-values and BCa intervals invite a population reading that the design cannot support.

*Fix:*
- Report campaign recall at k as counts (x/162).
- Give account-block bootstrap intervals for recall@k and the quantiles (the code path exists).
- Move the 78-pair AUC-PR family to an appendix.
- In the Uncertainty section, define the estimand explicitly as "variation over the accounts of this operation", and keep population language out of the abstract.

### M5. The Euler reproduction test is not the right test for n = 5 [verified]

*Where:* abstract `L56` ("p = 2.6×10⁻⁴ over five runs"); `L450`; `paper_assets.py:979-990` (`ttest_1samp`).

*Why.*
- **The arithmetic is right** (t = −12.19, p = 2.6×10⁻⁴; AUC t = −2.67, p = 0.056), but the test answers a different question.
  - It treats the published 0.0155 as a known population mean. It is a value from the authors' own runs with its own variance (see Q1).
  - It assumes normality of AP over n = 5. AP is bounded and right-skewed: on the log scale p = 0.0017.
  - The five runs share one environment (pip instead of conda, post-paper `requirements.txt`, one machine). The between-run variance therefore excludes exactly the factor that `L450` says may explain the gap.
- **A non-parametric exact test cannot go below p = 0.0625 at n = 5** (sign or Wilcoxon).
- **"Reproduces its published AUC but not its AP" is a distinction of effect size, not of test.** All five AUC runs (0.9848–0.9900) are also below the published 0.9906. The sign pattern is identical to AP's.

*Fix:*
- Drop the p-value from the abstract.
- Report "all five runs below half the published AP (max 0.0080 vs 0.0155)".
- The published value lies outside the 95% prediction interval for a further run: [0.0004, 0.0112] on the raw scale, [0.0022, 0.0142] on the log scale.
- If King et al. report a mean ± sd over runs, use a Welch test against that summary.
- State the AUC result as an equivalence statement with a margin (gap −0.0027), not as "reproduces".

### M6. Campaigns, accounts and user-days use different identity keys; two statements in the text are false [verified]

*Where:* `L204` ("a positive user-day belongs to exactly one campaign (181 …)"); `L420` ("The regrouping sees only the partition's positives, which is why it gives 157 rather than 162"). Code: `redteam_join.py` `group_into_campaigns` keys on user@domain, whereas `statistics.py:32-35` (`user_day = ["src_user"]`), `ledger.py:216-230` (`regroup_campaigns`) and `ledger.py:313-320` (`campaign_users`, the bootstrap "account") key on `src_user` without the domain.

*Evidence:*
- **User-days.** U86, U737 and U8777 appear under several domains (DOM1 and local domains C10, C583, …).
  - Temporal: 59 positive user-day units against 61 (day, campaign) pairs; 2 units hold two campaigns.
  - Harmonised days 1–13: 116 units against 120 pairs; 3 units hold two campaigns.
  - So `L204` is false, and the unit null slightly overcounts positive units.
- **Regrouping.** Removing events can only keep or increase the campaign count, so the explanation in `L420` cannot be right. Regrouping the temporal positives at 24 h by `src_user` gives 50, as reported in `TempGapCampaignsTwentyFour`. By user@domain it gives 52, the reported campaign count.
- **Consequences.** The gap-sensitivity row at 24 h is not the main analysis, the sensitivity analysis is confounded with a key change, and the "98 accounts" merge distinct accounts.

*Fix:* use one key (user@domain) everywhere, or justify `src_user` as "one human" and use it for campaigns too. Then regenerate `L204`, `L420`, the gap table and the unit null.

### M7. Multiplicity across the analysis space is controlled only locally, and highlights were selected after the fact [suspected / partly verified]

*Where:* `L200` (Holm over detectors for b*); Table 4 (`L296`; intrusion p-values unadjusted); `L141` (design history).

*Why.* There are 13 detectors × 3 units × 4 budgets × 2 protocols (plus R2) × 5 summaries × with/without day 1. Holm is applied inside a few families, yet the abstract picks the most striking cells: the AE at k = 100 on day 1, and Euler and M2c in user-days. The intrusion-day p-values (e.g. AE 0.001 at k = 100) are not adjusted across the 10 × 3 cells. M2c and the user-day unit were added after earlier results on the same operation had been seen (`L141`). The paper discloses this, but the abstract does not.

*Fix:*
- Declare one primary endpoint per question; for example, campaign recall at k = 100 events in the harmonised protocol, plus first-noticed day at k = 100.
- Make everything else explicitly exploratory, and apply Holm to Table 4.
- Pre-register the endpoints before running a second dataset.

## 5. Minor issues

1. **Unit rankings have no tie bracket** (`L204`, `L336`, `L547`). Coarse scores (M2a, M0b) tie at the unit top and are then ordered by the unit's first-event time, which correlates with user activity. M2a's user-day 21.6% may be mostly tie order. Computing an "against" bracket for units is cheap: put positive units last within ties. [verified from the code; size of the effect suspected]
2. **The bootstrap's benign component is unpaired across detectors.** Each detector draws its multinomial over its own score bins from the same seed (`stats_tests.py:418-431`), so benign multiplicities do not correspond across detectors. The tests become conservative, not invalid. `L213` says "paired in the campaign component", which is accurate; add "benign draws are independent across detectors". [verified]
3. **Harmonised fixed-threshold calibration** (`L223`; `statistics.py:606-620`). The threshold is the (k × n_days)-th score of the calibration partition. Training hours 0–41 count as 2 days, but they span 1.75 days, so "10 per day" calibrates to about 11.4 per day. This is small, but should be stated. [verified from code; day count inferred from the hour range]
4. **702 positives vs 715 − 14 = 701 labellable rows** (`L103-104` vs `L115`). `redteam_matches.out` finds 701 red-team rows with a matching authentication event, yet 702 events are labelled. Days 1–13 contain no duplicate red-team key [verified], so presumably one row on days 14–29 labels two authentication events [suspected]. State that "positives" are labelled authentication events, not red-team rows.
5. **"Exact" is used twice for the user-day null** (`L73`, Table 6 last row). It is exact only for a uniform permutation of units; see M1.
6. **b* is a single order statistic.** Consider reporting the budget at which m ∈ {1, 3, 5} campaigns are detected, or the expected first-noticed day under seed variation. The paper partly does this with "next two b(c)" (`L194`), but the abstract does not use it.
7. **Correlations** (`L313`, `L643`). n = 10 is a convenience set with shared features, as the paper states. The leave-one-out ranges are good practice. Add bootstrap-over-campaigns intervals for ρ, or drop the Holm p-values: Holm over 15 highly dependent tests is very conservative and hard to interpret.
8. **Regimes** (`L378`). R1 vs R2 is compared on one fit for AE and Euler, and the AE's ROC-AUC drops from 0.988 to 0.881. The paper's conclusion ("cannot be separated from fitting noise") is the right one. The table could therefore move to an appendix.
9. **Wording at `L150`.** "all 10,705,481 would fit in the harmonised protocol" reads as if 10.7 M ≤ 5 M. Say "could all be used".
10. **Cost per alert.** Precision or alerts per detected campaign at k is never reported. At 500/day over 57 days, that is 28,500 alerts to reach 7 campaigns (VAE) at the event level. One column would make the operational argument directly.
11. **Unexplained stored null estimate.** The run's simulated null for Euler b* (0.0221 stored vs 0.0203 exact) passes the 4σ check in `paper_assets.py:305`. The text quotes the exact value, which is correct; the stored simulation is redundant and could confuse readers of the JSON.

## 6. Number spot-check

Sources: `generated/numbers.tex` and the tables, recomputed from `reports/runs/20261022T102638Z/*/tables/*` and `data/interim/auth` with `.venv` Python.

| # | Claim | Location | Source / recomputation | Match? |
|---|---|---|---|---|
| 1 | Null b* median 6,859, 95% [251, 36,412] (harmonised) | `L200` | `random_first_detection_quantile` on `day_sizes.csv` | ✓ |
| 2 | Null b* median 49,373, [1,806, 261,368] (temporal) | `L200` | same | ✓ |
| 3 | P(random detects within 10) = 0.10% | `L286` | 1 − S(10) = 0.00101 | ✓ |
| 4 | Zero-recall upper bound 1.8% (162), 5.6% (52) | `L286` | 1 − 0.05^(1/n) = 1.83%, 5.6% | ✓ |
| 5 | PCA b* = 20, null 0.002, Holm 0.020 | `L309`, Table 3 | exact 0.0020; ×10 | ✓ |
| 6 | AE b* 59 → 528 without day 1, null 0.049 | `L290`, App. D | ledger; S on days 2–57 = 0.0489 | ✓ |
| 7 | AE day 1 at k = 100, p = 0.001; PCA day 8 at k = 50, p = 0.002 | Table 4 | 0.0006, 0.0024 (rounded) | ✓ |
| 8 | 69 significant pairs (68 at floor); 60 (56) | `L217` | `pairwise_comparisons.json` | ✓ |
| 9 | User-cluster significant 54/78 and 53/78 | `L399` | `bootstrap_user_clusters.json` | ✓ |
| 10 | PCA seeds 13–243, median 137, reported fit "one of the two best"; AUC-PR factor ≈ 4 | `L309`, `L399` | `fit_variance.json`: {20, 13, 159, 137, 243}; 4.59/1.23 = 3.7 | ✓ |
| 11 | Euler authors' AP 0.0058 ± 0.0018, t = −12.2, p = 2.6e-4; AUC t = −2.7, p = 0.056 | `L450` | `results.txt`, sample sd (ddof = 1) | ✓ |
| 12 | 65 singleton campaigns, largest 83, busiest days 8 (45), 12 (39), 13 (22), 97 campaigns | `L104` | ledger | ✓ |
| 13 | Temporal positives day 12: 207, day 13: 75 | `L115` | `day_sizes.csv` | ✓ |
| 14 | Harmonised day 1 = 1,653,645 events; per-day events and user-days (days 1–13) | `L115`, `L204` | rebuilt from raw parquet with the LogOn and src ≠ dst filter | ✓ exact |
| 15 | Staffing 0.6–1.2 analysts at 10/day, 30.9–61.7 at 500/day | `L206` | k × 10–20 min / 162 min | ✓ |
| 16 | 4 rules below chance in each protocol | `L425` | `table_rules_lanl.tex` | ✓ |
| 17 | AE 494 → 1,565, Euler 15,945 → 8,481 (R1 → R2); 8 unchanged | `L378` | `table_regimes.tex`, macros | ✓ |
| 18 | M0b user-day @500 = 7.4% vs null 3.0% (0.6–5.6%) | `L336`, Table 6 | macros | ✓ (but outside the null range, see M1) |
| 19 | "a positive user-day belongs to exactly one campaign (181)" | `L204` | join with `src_user` key: temporal 59 units vs 61 pairs; harmonised days 1–13: 116 vs 120 | **✗** |
| 20 | "157 rather than 162 … because the regrouping sees only the partition's positives" | `L420` | `ledger.py:216-230` keys on `src_user`; regrouping by user@domain reproduces 52 (temporal) | **✗ (wrong explanation)** |
| 21 | 715 rows, 14 unlabellable → 702 test positives | `L103-104`, `L115` | `redteam_matches.out`: 701 matchable rows | **? (inconsistent wording)** |

No macro or table value is out of sync with the run outputs. Rows 19 to 21 are errors or inconsistencies in the prose, not in the numbers.

## 7. Questions to the author

1. Is King et al.'s published AP of 0.0155 a single run, or a mean over runs with a reported dispersion? This decides which test applies in M5.
2. Why do campaigns use user@domain while user-days, accounts and regrouping use `src_user`? Which identity is intended to define a "compromised account"?
3. Do the harmonised days 14–57 confirm the fan-out baseline of M1? It is a few lines on top of the existing unit layout in `statistics.py`.
4. Was M2c's dependence on the user's historical event count (`log(n+1)` for a new source) intended? Have you compared it with a pure "new source for this user" indicator plus a label-free tie order?
5. What fraction of AE, VAE and Euler seeds reproduce "day 1 at k = 100" and "day 8 at k = 50"?
6. Which single endpoint would you pre-register for the second dataset?
7. Why are the bootstrap intervals already computed for b(c) quantiles and log-area (`budget_summary.json`) not reported?

## 8. What would move it up one level

**From (a) minor revision to a clean Master's accept:**
- Fix M6.
- Add the unit-level activity baselines and rewrite the user-day claim (M1).
- Qualify the abstract by fit count (M2).
- Drop the Euler p-value from the abstract (M5).

**From (b) major revision to a workshop or mid-tier venue paper:**
- Add a size-aware unit null and unit-level tie brackets.
- Run ≥ 5 seeds for every detector and report b* and first-noticed day as distributions.
- Put account-block intervals on recall@k and the quantiles, and report counts.
- Pre-declare one primary endpoint and treat the rest as exploratory.
- State the mechanical ROC-AUC / median-rank link and the harmonised non-significance.

**To a top-tier venue:** the evidence has to go beyond one operation.
- Run the harness on OpTC (already harmonised by Larroche) or on another multi-intrusion dataset, with the endpoint frozen beforehand.
- Include at least Argus or the authors' Euler inside the harness.
- Show that some conclusion (e.g. "ROC-AUC does not predict first detection", or "unit aggregation dominated by fan-out") replicates across intrusions.

The harness is already the strongest part of the work. The case study needs the second dataset.
