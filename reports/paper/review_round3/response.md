# Review round 3: response (23 October 2026)

Four reviewers, none of whom read rounds 1 and 2, reviewed the report after round 2 (software
v2.0.0) at two levels, a Master's thesis and a PhD student's paper: R1 venue fit and novelty
(`R1_venue_novelty.md`), R2 methods and statistics (`R2_methods_statistics.md`), R3 artifact and
code (`R3_artifact_code.md`), R4 Master's examiner and admissions reading
(`R4_examiner_writing.md`).

| Reviewer | Master's level | PhD level |
|---|---|---|
| R1 | Accept, minor revision | Main track: reject as it stands; workshop (CSET, LASER, AISec) or DTRAP: minor to major revision |
| R2 | Minor revision | Major revision; reject at a top-tier venue |
| R3 | Clear pass, top band (artifact) | Available and Functional now; Reusable within reach; Results Reproduced not yet |
| R4 | Distinction, low band, minor corrections | Strong research signal; strong-workshop level |

**Meta-review.** R2's first point decided this round. Ranking user-days by a label-free count
of the computers a user touched that day matches or beats every detector in that unit, so the
user-day result that the abstract presented as the paper's positive finding measured the fan-out
of the compromised accounts. The uniform random-ranking null used against it was the wrong one,
as the floors showed. R2 had computed this for the temporal protocol and as a lower bound for the
harmonised one; it was recomputed here on all harmonised days with red-team events (days 1 to 29),
streaming days 14 to 29 from the same mirror, and it holds: 56.8% of the campaigns at 500
user-days and 30.2% at 50 for the count of distinct destinations, against 51.2% and 13.6% for
Euler, the best detector. Every other point was taken at face value only after it was checked in
the code, the run outputs or the source; one was corrected (R4 wrote that the campaign-weighted AP
tracks the median budget significantly in both protocols; it does in the harmonised one only).

Status: **addressed**, **partly**, **Phase B** (needs a new run or new code), **declined** (with
the reason).

## R2 (methods and statistics)

| Point | Status | Where |
|---|---|---|
| M1 User-days lack a trivial baseline; the uniform null is misspecified | Addressed. `scripts/user_day_activity.py` recomputes user-day activity from the log with the pipeline's own parsing (same `event_id`, day 13 streamed and on disk identical by row count, identifier range and row hash); the generator checks events, user-days and positives per day against the run. Four label-free rankings and a null of random event scores aggregated by their maximum (the floors fall inside it) in Table 7 and the temporal appendix; abstract, units section, discussion, scope and conclusion rewritten. A conditional permutation null: not needed, the aggregation null is the one the floors validate | Sections 6.1, 7.5, 10; Table 7 |
| M2 Headline detections are single fits | Partly: stated as such in the abstract, Section 7.2 and the conclusion ("PCA in 2 of its 5 fits"; the VAE, the autoencoder and Euler fitted once); the partial-day explanation called plausible, not demonstrated. Seeds for the autoencoders and Euler, a volume-normalised budget: Phase B | Abstract, Sections 7.2, 7.8 |
| M3 "ROC-AUC tracks the middle" holds only in the tuned protocol, and partly by construction | Addressed: harmonised non-significance stated, the mechanical link named, the campaign-weighted AP added to Tables 3, 4 and 6 | Section 7.4, Table 6 |
| M4 No uncertainty on the headline metrics | Partly: recall at 500 and the median budget with account-bootstrap intervals (appendix E), recall as counts in Tables 3 and 4, the estimand stated as the accounts of one operation. Intervals for every budget and unit: Phase B | Section 6.3, Tables 3, 4, 16, 17 |
| M5 The Euler test is not right for n = 5 | Addressed: the t-test is gone; every run is reported below half the published AP, which lies above the 95% prediction interval for a further run (log and raw scales); the AUC is stated as a gap five runs cannot separate from zero; the p-value is out of the abstract | Section 9, abstract |
| M6 Identity keys differ; two statements false | Addressed in the text: user-days by user name, campaigns by user and domain; 176 positive user-days for 181 pairs; the regrouping explained by the key (162 by user and domain, checked by the generator). The code is unchanged so that the paper describes the run; one key everywhere is Phase B | Sections 6.1, 7.9 |
| M7 Multiplicity across the analysis | Addressed in the text: an "Analytic flexibility" threat, results called exploratory, intrusion probabilities marked unadjusted, endpoints proposed for a second dataset | Section 11 |
| minor 1 Unit tie brackets | Phase B | |
| minor 2 Benign draws unpaired | Addressed | Section 6.3 |
| minor 3 Calibration over two calendar days | Addressed (about 1.17k per 24 hours) | Section 6.4 |
| minor 4 702 positives from 701 rows | Addressed: one red-team row labels two identical authentication events, verified on the streamed days | Section 3.2 |
| minor 5 "Exact" user-day null | Addressed | Section 6.1, README |
| minor 6, 7 b* as one order statistic; correlation intervals | Partly: next two budgets reported; leave-one-out ranges kept | Appendix E |
| minor 8 Regimes table | Addressed: moved to an appendix | Appendix G |
| minor 9 "all would fit" | Addressed | Section 5 |
| minor 10 Cost per alert | Phase B | |
| minor 11 Stored null estimate | Declined: the generator checks it against the exact value and quotes the exact one | |

## R1 (venue and novelty)

| Point | Status | Where |
|---|---|---|
| M1 One host may explain the positive findings | Addressed for user-days (the activity rankings explain them) and stated: all but 11 of the 162 campaigns hold only events from the top host. A user-day table stratified by host for the detectors needs per-unit ranks the run did not save: Phase B | Sections 7.5, 8 |
| M2 Positioning against R+R, CyberGFM, Zhao and Hassan | Addressed: a positioning paragraph lists what is and is not new; CyberGFM (random split, Euler AP 0.0627) and R+R's rerun of Euler checked in the sources and logged in `references_log.md` | Sections 2, 9 |
| M3 Abstract and conclusion overclaim | Addressed: rewritten around the robust results; the conclusion answers three stated research questions | Abstract, Sections 1, 14 |
| M4 Seeds for the detectors behind the headlines | Phase B; within-PCA contrast added (its two fits with the lowest AUC-PR give its two earliest detections, checked by the generator) | Section 7.8 |
| M5 Catalogue lacks the state of the art | Phase B; README tagline no longer claims "published performance" | README, Section 11 |
| M6 Euler diagnosis | Partly: test replaced (R2 M5), R+R and CyberGFM discussed. Float32 and alignment measurements, contact with the authors: Phase B (the contact is the author's action) | Section 9 |
| M7 Not confirmatory | Addressed in the text (exploratory, endpoints for the next dataset) | Section 11 |
| minor: title "Temporally Leak-Free" | Declined for now: the row-locality test is what neither close work does, and the title is that of the archived software release | |
| minor: M2a fixed-threshold recall | Addressed (table note) | Table 8 |
| minor: one fit vs ten runs | Addressed | Section 10.1 |
| minor: mirror verification | Addressed (the exact harmonised event set is the strongest check) | Sections 3.1, 11 |

## R3 (artifact and code)

| Point | Status | Where |
|---|---|---|
| M1 The leakage guard is not on the LANL path | Addressed both ways: the text describes what the run used; `train_eval.main` now runs `verify_temporal_order` on the written partitions (which accepts a protocol without validation), with a test | Section 4.2; `split/temporal.py`, `pipeline/train_eval.py` |
| M2 The code that produced the numbers has no test | Addressed: `tests/integration/test_lanl_entry_points.py` composes the real Hydra configuration and runs `build_features.main` and `train_eval.main` on the demo for both protocols, checking the ledger, day sizes and first detections against each other | tests |
| M3 Reproduction entry points | Partly: Makefile portable, DVC comment and the paper's reproducibility section state what the run was. One command for the whole run, per-day hashes for all 58 days: Phase B | Makefile, `dvc.yaml`, Section 13 |
| M4 Dead configuration | Addressed: `build_features` refuses feature settings it would ignore (`SUPPORTED_FEATURES`), with tests; `conf/features/base.yaml` and `params.yaml` state what is computed; unread model configurations are marked descriptive and `rules.yaml` matches the code | `pipeline/build_features.py`, `conf/` |
| M5 No detector registry | Phase B | |
| minor 1 Stale docstring | Addressed | `models/base.py` |
| minor 2 Stale README claim | Addressed | README |
| minor 3 Calibration | Addressed (R2 minor 3) | |
| minor 7 No `.gitattributes` | Addressed | `.gitattributes` |
| minor 8 "Exact" null | Addressed | README |
| minor 4, 5, 6, 9 to 14 | Partly or Phase B: pyproject version left at 0.1.0 so the lockfile is untouched | |

## R4 (examiner)

| Point | Status | Where |
|---|---|---|
| M1 No research question | Addressed: three research questions, answered in the conclusion | Sections 1, 14 |
| M2 Abstract claims more than the body | Addressed (R1 M3); the campaign-weighted AP claim corrected to the harmonised protocol | Abstract |
| M3 Role of the temporal protocol | Partly: its label-tuned results are flagged in captions and text; moving it wholly to an appendix is left for a venue version | |
| M4 OpTC not discussed | Addressed: why it was not attempted and what it needs | Section 11 |
| M5 "Reproduces" from a non-significant test | Addressed (R2 M5) | Section 9 |
| M6 Strongest findings buried | Addressed: campaign-weighted AP in Tables 3, 4 and 6; regimes and the circularity control moved to appendices | Section 7.4, appendices |
| M7 No discussion | Addressed: a Discussion section (comparison, what the case study suggests) | Section 10 |
| M8 Related work as a list | Partly: positioning rewritten; a feature-matrix table and further literature are left for a venue version | Section 2 |
| Figures and tables | Addressed: Figure 2 shows the random ranking's mean and band instead of one random draw; captions state their finding; Tables 3 and 4 no longer cramped; appendix tables numbered and captioned; floats kept within their sections; one shaded band (10 to 500) in both figures | |
| Strom citation | Addressed | bibliography |

## Phase B: what needs a new run or new code

1. A second dataset with several intrusions (OpTC under Larroche's harmonisation: an adapter for
   its host event records), with endpoints fixed beforehand.
2. Seeds for the autoencoders and Euler; seed medians as the primary numbers.
3. State-of-the-art detectors through the ledger (the authors' Euler, Argus).
4. One identity key for campaigns, user-days and the bootstrap; per-unit ranks in the ledger, so
   unit tie brackets and host-stratified unit results can be computed.
5. Intervals for every budget summary and unit; cost per detected campaign.
6. A detector registry; one command for the whole run; per-day content hashes of all 58 days.
7. Contact with Euler's authors before the next version.
