# Review round 2: response (23 October 2026)

Four independent reviewers, each starting without the author's context, read the report
built on run `20261022T102638Z`, the run outputs and the code, recomputed what they could and
checked the citations: R1 evaluation methodology (`R1_methodology.md`), R2 statistics
(`R2_statistics.md`), R3 lateral-movement detection on LANL (`R3_domain.md`), R4
reproducibility and Euler (`R4_reproducibility.md`).

| Reviewer | Soundness | Presentation | Contribution | Reproducibility | Top-tier venue | Preprint |
|---|---:|---:|---:|---:|---|---|
| R1 | 3 | 3 | 2 | 4 | Reject | Ready after minor fixes |
| R2 | 2 | 3 | 2 | 5 | Weak reject | Ready after text-only changes |
| R3 | 3 | 3 | 2 | 5 | Reject | Ready after a short minor revision |
| R4 | 3 | 3 | 2 | 4 | Major revision | Ready after minor corrections |

**Meta-review.** All four confirm the numbers they recomputed: the exact null, the
correlations, the intervals, the Euler counts (514, 517, 518, 525), the citations they checked;
R4 regenerated every generated file of the paper byte for byte in a clean clone, and R3 found
that the harmonised event set equals Larroche's (2026) to the event. None found a fatal flaw for
a preprint. For a top-tier venue all four agree on the limits: one intrusion in one dataset, an
empirical delta over Larroche (2026) and Zhao and Hassan (2026) that the text did not make
explicit, no state-of-the-art detector in the harness, and headline statements that rested on
the most fragile statistic (the first detection, one fit, a partial first day).

**What changed.** Every point that the committed outputs allow was addressed in the text and in
`scripts/paper_assets.py`, which now also computes the exact user-day null, Holm-adjusted null
probabilities, seed medians, the correlation family with leave-one-out ranges, the
intrusion-level detection day, the first detection without the partial day, the fixed-threshold
recall and the comparison with Larroche's Table 5, and checks the harmonised universe against
his Table 4. Every new number agrees with the reviewers' independent recomputations. Points that
need a new run or new code are marked **Phase B** and grouped at the end.

Status: **addressed**, **partly**, **Phase B** (needs a run), **declined** (with the reason).

## R1 (methodology)

| Point | Status | Where |
|---|---|---|
| F1 One intrusion, no second dataset | Partly: stated as the first threat and in the scope; a second dataset is Phase B | Sections 12, 13 |
| F2 Headline rests on the first detection | Addressed: correlation family (first detection, 10/25/50% quantiles, log-area) with Holm and leave-one-out; abstract rewritten ("tracks the bulk of the curve, not its leading edge"); power stated | Section 7.4, Table 6, appendix C |
| M1 "Unrelated" is absence of evidence | Addressed | Abstract, Section 7.4 |
| M2 Tuning under-reported; "same budget of 20" false | Addressed: 6, 9, 12, 20 candidates; the four label-tuned temporal detectors flagged; harmonised named as the label-free evidence. Tuning the autoencoders and Euler: Phase B | Section 5, Table 4 note, Section 12 |
| M3 Post-hoc design (data snooping) | Addressed: design-history paragraph and limitation 19; "leak-free" defined as temporally leak-free. The title is kept by the author's choice. Confirmation on unseen data: Phase B | Section 4.2 |
| M4 Units: cost, no null, M0b omitted | Partly: exact user-day null (agrees with R2's), M0b added, cost caveat. Unit tie brackets, workload-priced recall, a source-host-day unit, edge-hour null: Phase B (positive edge-hours are not in the outputs) | Section 7.5, Table 7 |
| M5 Inference only on AUC-PR; campaign-weighted AP unreported | Partly: campaign-weighted AP in the correlation family; budget quantiles reported. A paired bootstrap of budget summaries: Phase B | Appendix C, D |
| M6 Fitting variance only for cheap detectors; PCA's fit atypical; 5M sample | Partly: seed medians and per-seed null probabilities reported, the reported PCA fit called second best of five. Seeds for the autoencoders and Euler, fitting on all training rows: Phase B | Section 7.8, Table 10 |
| M7 The benchmark's Euler is weaker | Addressed in text (deviations listed, fitted once, not tuned). The authors' code inside the harness: Phase B | Sections 5, 9 |
| M8 No published detector run in the harness | Partly: Larroche's Table 5 on the same events (Table 13). Reproducing ADSAGE's CR-1000 on its split: Phase B | Section 10 |
| M9 Operation-level detection | Addressed: intrusion-level day per budget with its exact null, and the partial first day | Section 7.2, Table 5, appendices D and E |
| m1 Candidate counts | Addressed | Section 5 |
| m2 R@1e-4 mostly zeros | Addressed | Section 7.4 |
| m3 Per-day AP control | Phase B (needs per-event scores) | |
| m4 Recall at the realised fixed-threshold volume | Addressed | Table 8 |
| m5 "Identical test set" | Addressed ("the same test events") | Abstract, Section 9 |
| m6 Environment, table V mismatch, "published data" | Addressed | Section 9, Scope |
| m7 Holm for null probabilities | Addressed | Tables 3, 4 |
| m8 Partial verification of the mirror | Addressed | Sections 3.1, 12 |
| m9 Basis of 97.5% | Addressed: labelled positives (split summary) | Section 3.3 |
| m10 M2a tie dominance | Partly: stated as a property of the score's resolution, null tie range given. A secondary tie-break baseline: Phase B | Section 7.9 |
| m11 Hopper as the low-end anchor | Addressed | Section 6.1 |
| m12 Tuning on a budget metric | Phase B | |
| m13 First-detection interval | Addressed: no longer reported as an interval | Section 6.3, limitations 13 |
| Presentation: abstract density | Addressed: fewer numbers, three findings and Euler | Abstract |
| Presentation: seed bars in Figure 1 | Phase B (needs seeds for every detector) | |

## R2 (statistics)

| Point | Status | Where |
|---|---|---|
| M1 "Unrelated" is specific to the first detection; leverage; seeds | Addressed: family with Holm, leave-one-out ranges; seed dependence stated. A confidence interval for each rho: declined, the permutation p and leave-one-out range are reported instead | Section 7.4, appendix C |
| M2 Multiplicity and fitting variance of null probabilities | Addressed: Holm column; per-seed nulls (median 0.014 for PCA); next campaign budgets. 20 seeds and seeds for the deep detectors: Phase B | Tables 3, 4, 10; Section 7.3 |
| M3 The first-detection interval is a method artifact | Addressed: removed; three smallest b(c) reported | Section 6.3, appendix D |
| M4 User-day results against one random draw | Addressed: exact null, values equal to R2's (181 positive user-days, median 72, 3.0% [0.6%, 5.6%]). Account-cluster intervals for the 51.2% and 50.6%: Phase B (accounts are not in the ledger) | Section 7.5, Table 7 |
| M5 Intervals understate the measured dependence | Partly: text says the block intervals are lower bounds and names the lost pairs. User-cluster intervals in the main tables and named contrasts: Phase B | Section 7.8 |
| M6 R1/R2 confounds the training data with fitting noise | Addressed | Section 7.7 |
| minor 1 Monte Carlo SE of p | Addressed: fixed in code with a test; methodology corrected; no decision changes | `stats_tests.p_value_monte_carlo_se` |
| minor 2 Partial pairing | Addressed | Section 6.3 |
| minor 3 Name of the test | Addressed ("percentile-interval inversion") | Section 6.3 |
| minor 4 Floor counts | Addressed | Section 6.3 |
| minor 5 M2a null depends on ties | Addressed | Section 7.9 |
| minor 6 R@1e-4 quasi-definitional | Addressed | Section 7.4 |
| minor 7 Zero-recall bound | Addressed (1.8% for 0/162) | Section 7.1 |
| minor 8 Bands on the recall curves | Phase B | |
| minor 9 Monte Carlo error in the paper | Partly (resolution-floor statement) | Section 6.3 |
| minor 10 Stale documentation | Addressed | `docs/methodology.md` |
| minor 11 Detectors not independent units | Addressed | Appendix C |
| minor 12 Recall per campaign gap | Partly: first detection and intrusion day are gap-invariant; recall per gap stays in the JSON | Section 7.9 |

## R3 (domain)

| Point | Status | Where |
|---|---|---|
| F1 Novelty against Larroche (2026) and Zhao and Hassan (2026) | Partly: the delta is stated in Positioning; Larroche's Table 5 and the campaign detection rate of Zhao and Hassan are used. State-of-the-art detectors through the harness: Phase B | Sections 2, 10 |
| M1 Larroche's Tables 4 and 5 | Addressed: universe equality checked by the generator; Table 13 | Sections 4.1, 10 |
| M2 Catalogue not representative; M4c refused silently | Partly: absences and the sequence model's exclusion disclosed. Argus, the authors' Euler, UltraLMD++, a sequence model, a lexicographic novelty baseline: Phase B | Sections 2, 5, 12 |
| M3 Partial first test day | Partly: sensitivity without day 1 for every first detection and in the intrusion analysis. Days redefined from the test start: Phase B | Section 7.2, appendix D, limitation 18 |
| M4 Intrusion-level time to detection | Addressed | Section 7.2, Table 5 |
| M5 DECADES, ADSAGE, UltraLMD++ misdescribed | Addressed after reading the primary sources (quotes in `references_log.md`). Recomputing their statistics on our rankings in their splits: Phase B | Section 2, Table 14 |
| M6 Units priced like events; source-host-day unit; machine accounts in the top 500 | Partly: caveats stated. The rest: Phase B | Sections 6.1, 7.5 |
| M7 Prior replications; NTLM filter label-informed | Addressed: R+R and Bouček and Husák cited; Larroche's NTLM remark quoted | Sections 2, 9 |
| m1 Hopper "deployed" | Addressed ("evaluated on fifteen months of enterprise logins", verified) | Section 1 |
| m2 Zhao and Hassan on LANL | Partly: campaign detection rate and the 42.1-point seed range added (verified). The claim that another detector wins on LANL could not be verified from the paper's abstract and is not added | Section 2 |
| m3 "Scores every event independently" | Addressed | Abstract, Table 2 |
| m4 Protocols differ in split and universe | Addressed | Section 4.1 |
| m5 Null at 10 alerts per day | Addressed (0.10%) | Section 7.1 |
| m6 RUN.md nulls disagree | Addressed: RUN.md states that run files hold simulated values | `reports/runs/20261022T102638Z/RUN.md` |
| m7 Campaign structure | Addressed (65 singletons, largest 83, busiest days) | Section 3.2 |
| m8 M2a ties | Addressed | Section 7.9 |
| m9 `redteam_concentration.json` harmonised block | Declined: the block counts red-team rows in the window (715, all after hour 41), as the script's docstring says; the 702 are labelled events of the remote LogOn universe | |
| m10 Tuor et al. report ROC-AUC | Addressed | Section 2 |
| m11 Cite the TOPS version | Addressed | Section 2 |
| Missing related work | Partly: R+R, Bouček and Husák, Jbeil, King and Huang (2023) added and verified. The others listed (Bilot et al., Tariq et al., Larroche 2024, Lee et al., Tang and Ge, Powell, AI2, Oprea et al., Gutflaish et al., Majorczyk et al.) are kept for the venue version | |

## R4 (reproducibility and Euler)

| Point | Status | Where |
|---|---|---|
| M1 "Published data" not established | Addressed: released code and preprocessing in the November 2023 environment; 15,610 nodes and 49.3M/44.7M lines against 17,685 and 45.9M; 4 workers against the script's 8 | Section 9, Scope |
| M2 Undocumented choices incomplete | Addressed: each verified in the authors' code (edge weights, identity features, non-edge negatives, time tail, no seed, dropout and DropEdge keeping 20%); our own choices and our two unintended deviations disclosed | Section 9, methodology, limitation 17 |
| M3 Cheap alternative explanations of the AP gap | Partly: one-sample t-tests, TPR, FPR and precision reported; the four lost edges, float32 saturation and edge alignment named as candidates. Measuring them: Phase B | Section 9 |
| M4 Ablations to diagnose the gap; the vocabulary red flag | Partly: the red flag is stated. Ablations: Phase B | Section 9 |
| M5 Euler in the benchmark: unseen node, input graph, one fit | Partly: input graph, shared unseen node and single untuned fit stated. Share of events touching the unseen node, its ablation, seeds: Phase B | Section 5 |
| M6 The Euler path is not one command; commits per result | Partly: per-result commits and the missing single script stated. The script: Phase B | Section 14 |
| minor 1 Prevalence column | Phase B (presentation) | |
| minor 2 Events of the authors' row | Addressed in text (49,341,299 lines) | Section 9 |
| minor 3 "Pinned dependencies" | Addressed | Section 9 |
| minor 4 ReLU then tanh | Addressed | Section 9, methodology |
| minor 5 Contact the original authors | Not yet: stated in the ethics statement; an action for the author before the venue version | Ethics statement |
| minor 6 DropEdge wording | Addressed | Section 9 |
| minor 7 Hash figures in CI | Phase B | |
| minor 8 Stale `dvc.yaml` comment | Phase B | |
| minor 9 Biased standard deviation | Not listed (negligible) | |
| minor 10 p-values in the abstract | Addressed in Section 9, not in the abstract | Section 9 |

## Phase B: what needs a new run or new code

1. A second dataset with several intrusions (OpTC with Larroche's labels), with the catalogue
   frozen.
2. State-of-the-art detectors through the harness on the same events: Argus and the authors'
   Euler (per-event scores), UltraLMD++, a sequence model, a lexicographic novelty baseline.
3. Five or more seeds for the autoencoders and Euler, twenty for the seeded detectors; fitting on
   all harmonised training rows.
4. Euler: one-at-a-time ablations toward the authors' code, both implementations evaluated with
   one function on one edge set, the share of events on the shared unseen node; one script for
   the whole Euler reproduction; contact with King and Huang.
5. Units: tie brackets, positive edge-hours saved for an exact null, a source-host-day unit,
   recall against analyst workload, account-cluster intervals.
6. Test days redefined as 24-hour windows from the test start; a paired bootstrap of the budget
   summaries; user-cluster intervals in the main tables; recall bands on the curves.
7. Published statistics recomputed on our rankings in their splits (DR-1K, CR-1000, Rec@10 per
   hour), including the legacy split train 0-7 / test 8-12.
