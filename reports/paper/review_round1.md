# Review round 1: simulated peer review and response (21 October 2026)

Three independent reviewers, each starting without the author's context, read
version 2 of the report, the run outputs and the code, and recomputed what they
could. Reviewer 1: evaluation methodology for ML in security. Reviewer 2:
statistics and resampling. Reviewer 3: lateral-movement detection on LANL.

| Reviewer | Soundness | Presentation | Contribution | Recommendation |
|---|---:|---:|---:|---|
| R1 | 2 | 3 | 2 | Major revision |
| R2 | 2 | 3 | 2 | Major revision |
| R3 | 2 | 3 | 2 | Reject in current form, resubmission encouraged |

**Meta-review.** Major revision. The harness, the leakage controls and the
transparency are judged above the norm by all three; the empirical claim was
not. The title and abstract generalised from five detectors, one intrusion and
one label-calibrated detector; the "worse than random" reading compared with a
single random draw; the significance language overstated a bootstrap that
cannot vary the intrusion and whose floor is not Monte Carlo certified; and the
case study was not placed next to published budgeted results on almost the
same days, which detect a non-zero share of the red-team activity.

Status: **addressed** in the revised text below; **run** needs a new LANL run
and is scheduled for a later run.

## Claims and framing

| Point | Raised by | Status |
|---|---|---|
| Title contradicted by AUC-PR result; "do not predict" generalises from n = 5 | R1 W1, R2 M1, R3 M1 | Addressed: retitled to a case study; abstract and scope rewritten |
| "ROC-AUC carries no information": absence of evidence | R1 W1, R2 M2 | Addressed: exact permutation p-values (0.95, 0.23, 0.10) reported; wording removed |
| Contrast driven entirely by M1, the AUC-PR-calibrated detector | R1 W2, R2 M1 | Addressed: leave-one-detector-out table; attribution stated in abstract and results |
| "Prevalence sensitivity" explanation is a non sequitur | R1 W9, R2 M2 | Addressed: replaced by the top-weighted versus mean-rank explanation |
| Low-FPR metric computed but unused (ρ = 0.87) | R1 W8, R3 M3 | Addressed: recall at FPR 1e-4 in the results table and the correlation analysis |
| Novelty overstated; prior art (Euler's AUC remark, McClish, Tuor 2017, Hopper) | R1 W9, R3 M3, m1, m2 | Addressed: contribution reframed as harness plus case study; citations added and verified |
| Contribution "we show" for a one-line identity | R1 W9, R3 m1 | Addressed: presented as a property of the harness |

## Statistics

| Point | Raised by | Status |
|---|---|---|
| b* compared with one random draw; "worse than random" unsupported | R1 W3, R2 M3, R3 M4 | Addressed: exact random-ranking null derived and implemented (`random_first_detection_survival`, tested against exhaustive enumeration); every b* reported as a null probability; figure shows the null band |
| b* is an extreme statistic without uncertainty; full b(c) not published | R1 W3, R2 M3, R3 M2 | Run: publish b(c) per campaign, quantiles, curves with intervals |
| Floor p-values called upper bounds; R = 2000 does not certify Holm at 0.05 | R2 M5 | Addressed in text (Monte Carlo bound 0.063, 2,515 resamples needed); run: R = 10,000 |
| Significance under one operation; benign clustering; fitting variance | R1 W10, R2 M4, R3 M9 | Addressed in text (reframed as stability under resampling of accounts); run: benign-clustered bootstrap, multi-seed fits |
| Percentile-inversion test, not a null-resampling test | R1 W10, R2 M5 | Addressed in text; studentised or BCa intervals planned |
| Event-weighted AUC-PR against campaign-weighted operation | R2 M6 | Run: campaign-weighted AP |
| Bootstrap described as "stratified" in code | R2 m2 | Code docstrings to correct |
| Zero-recall bound; single-day change not significant | R2 m7, m8 | Addressed: Clopper-Pearson bound with its caveat; Fisher p = 0.18, "not reproduced" |
| Accounts with two campaigns resampled separately | R2 m5 | Run: resample by account |

## Protocol and leakage

| Point | Raised by | Status |
|---|---|---|
| Daily top-k is retrospective within a day | R1 W6a | Addressed in text; run: fixed-threshold variant calibrated on validation |
| M1 normalisation spans the three test days (look-ahead) | R1 W6b | Addressed in text; code: training-fitted normalisation |
| Abstract claimed warm-up for training | R1 W6d | Addressed: training starts cold, stated; covariate shift listed |
| Architecture test is static | R1 W6e | Addressed: described as static |
| Split argued against a straw man; Larroche's 41-hour split | R3 M8 | Addressed in text; run: harmonised split as primary, test on all activity after 41 hours |
| M1 calibrated on the same operation; Larroche advises unsupervised use only | R1 W2, R3 M8 | Addressed in text; run: label-free M1, equal tuning budgets |

## Features, universe and baselines

| Point | Raised by | Status |
|---|---|---|
| M2a ranks by novelty tier, then latest first | R1 W4 | Addressed: diagnostics section with measured new pairs per day (76,194 on day 12) and the tier argument; run: time-invariant rarity |
| No source-side feature although 97.5% of positives share a source host | R1 W8, R3 M7 | Addressed in text; run: (user, source) and (source, destination) novelty, known-source-PC rule, Bowman's UA rule |
| Event universe and unit differ from prior work (61.7% machine-account events) | R3 M5, R1 W5 | Addressed in text (measured share, comparison table); run: LogOn-only universe, edge and user-day units |
| Published budgeted results not used | R3 M6 | Addressed: comparison table with DECADES, ADSAGE, UltraLMD++ (numbers checked in the sources) |
| No deep or graph detector | R1 W8, R3 M10 | Addressed in framing; run: Euler-style temporal link prediction |
| Budget range not derived from a staffing model | R1 W5, R3 M11 | Addressed as a threat; Hopper cited as reference point; staffing model planned |
| Unused red-team activity of days 14 to 29 | R1 W7 | Run planned |
| Campaign gap g not analysed | R1 W7, R3 M9 | Run planned |

## Factual and presentation fixes (all addressed)

- PCA lower bound of the ROC-AUC range (R1 m1, R2 m1): the sentence is removed with the reframing.
- "Two ends inverted" (R1 m2): removed.
- Bowman split described as non-chronological (R1 m5, R3 m7).
- M0a row-local status corrected to "no" (R1 m8); ECOD renamed M3c in the text (R1 m9).
- Night window stated as unused (R1 m10); R2/R3 inputs and unseen-user threshold documented (R1 m11).
- Event order "(time, event identifier)" stated (R1 m12); Isolation Forest parameters attributed to the specification (R1 m13).
- Time to detection no longer presented as a reported result (R1 m14, R3 m12).
- Pairwise appendix gives raw and Holm-adjusted p (R1 m17); M0b single-day recall marked n/a (R3 m11).
- ATT&CK mappings qualified as design intent with the loose cases named (R3 m8).
- Truncated campaigns (2, 9 events) stated (R2 m6, R1 W7); "involve" instead of "originate" for host counts (R2 m10).
- UltraLMD++ described for both configurations; its Rec@10 of 0.1788 is the deduplicated configuration (R3 m6, corrected after checking the source).
- Argus AP unit (percent) stated (R3 m5).
- Siadati and Memon cited (R3 m3); uncited bibliography entries are not printed by biblatex.
- No campaign spans validation and test (R3 m14, checked); no red-team event has a machine-account source (R3 m15, checked).

## Not adopted

- Anonymisation (R1 m23): a submission copy will be anonymised for a double-blind venue; the repository version keeps the author's name.
- "Reject" (R3): the revision narrows the claims to what the data support; the experiments R3 lists are scheduled and will decide whether a resubmission is a benchmark paper or a case study.
