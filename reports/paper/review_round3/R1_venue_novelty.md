# Review R1: venue fit, novelty and proportionality of claims

Reviewer lens: senior programme-committee member at a security venue (ACSAC / RAID / DIMVA / AISec / USENIX Security).
Material read: `reports/paper/main.tex` (all 732 lines), every file in `reports/paper/generated/`, `references.bib`, the relevant entries of `references_log.md`, `README.md` (head), `src/authbench/evaluate/budget.py` (null), `src/authbench/features/novelty.py`, `src/authbench/models/stats.py`, `scripts/paper_assets.py` (unit null), and `reports/runs/20261022T102638Z/harmonised/tables/{fit_variance,budget_summary}.json`. I did not open `review_round1.md` or `review_round2/`.
External literature checked (Oct 2026): Larroche 2026 (arXiv 2607.29390, full text), Zhao & Hassan 2026 (arXiv 2609.12263, CCS 2026, full HTML), Wang et al. R+R (arXiv 2503.20281 = ACSAC 2025, full text and Fig. 3), King et al. CyberGFM (arXiv 2601.05988, full text), Bouček & Husák CNSM 2025 (abstract only), LMDG (arXiv 2508.02942, abstract), a public GitHub prototype (Dr-Deep-Learning/lanl-lateral-movement).

"Verified" means I checked it against the file, the run outputs or the source. "Suspected" means I reasoned about it but did not check it against data.

---

## 1. The contribution in my own words

AuthBench is an evaluation harness for authentication anomaly detectors. It enforces a chronological split in code and tests that each detector scores an event the same way regardless of what else is scored. It then reads every detector's daily ranking as an alert queue. For each red-team "campaign" (one account's activity, with gaps under 24 h) it records the smallest daily budget at which the campaign would surface. It compares the best of those budgets with its exact distribution under a random ranking, and repeats the analysis for edge-hours and user-days. The harness is applied to LANL under Larroche's (2026) harmonised event set, which it reproduces exactly, and to a 3-day temporal split. Thirteen mostly generic detectors are evaluated. Almost nothing is found within 10 to 100 alerts per day. User-day ranking does much better than random at 500 per day. ROC-AUC does not predict first detection. The authors' Euler code, rerun here, does not reach its published AP.

## 2. Verdicts

### (a) Master's thesis / Master's research project (CS, security/ML): **Accept (minor revision)**

For a Master's project this is well above the usual standard on three points: the engineering discipline, the statistics and the honesty. The engineering includes a code-level leak guard, a static import test, bit-exact row-locality tests and CI that regenerates every number in the paper. The statistics include exact nulls, block bootstrap, Holm correction, seed variance and a user-cluster bootstrap. The honesty shows in the threats, design-history and scope sections and in the correction notice for v1. The weaknesses are a single dataset and intrusion, incremental novelty, and an abstract that says more than the robust results support. A Master's examiner would ask for these to be discussed, not fixed. The minor revisions I would require are:
- tone down the abstract (issue M3 below);
- position the work against R+R and CyberGFM (issue M2);
- state plainly that the seed-dependent and post-hoc results are exploratory.

### (b) Paper by a first/second-year PhD student

- **Main track (ACSAC, RAID, DIMVA full paper, USENIX Security): Reject in its current form.** Where a venue has a revise-and-resubmit cycle, the verdict would be major revision.
  - The paper's own positioning concedes the core observation: budgeted metrics on LANL exist (DECADES, ADSAGE, UltraLMD++) and harmonised re-evaluations exist (Larroche 2026; Zhao 2026 at CCS 2026; R+R at ACSAC 2025).
  - What remains new is mostly a reporting device: the exact random-ranking null for the first-detection budget, three alert units, and an intrusion-day reading that operationalises a remark of Larroche (2026, §5.2).
  - It is applied to one intrusion, with a catalogue that leaves out every state-of-the-art lateral-movement detector except a reimplemented Euler that underperforms the authors' code.
  - Two of the substantive findings cannot be separated from the specifics of this single operation: the user-day lift and the "a week later at 50" detections. The paper says so itself (§13, l. 539), which is to its credit but is fatal for a main-track empirical claim.
  - A main-track committee would expect at least one of three things: a second dataset with several intrusions, the SOTA detectors run through the ledger, or a methodological result that generalises beyond LANL.
- **Workshop (CSET, LASER, AISec, ANUBIS, where Larroche 2026 appeared) or a journal reproducibility track (ACM DTRAP): Minor-to-major revision. Likely accept once condensed to 6 to 8 pages and repositioned.**
  - These venues value evaluation methodology, honest negative results and artefacts, and the artefact here is strong.
  - CSET and LASER are the best fit: both are explicitly about experimental methodology and authoritative negative results.
  - AISec fits if the ML-evaluation angle is foregrounded.
  - The Euler reconciliation alone (§9) could also be a short reproducibility note, provided the Euler authors are contacted first and R+R is discussed.
- **SoK track: no.** The paper is not a systematisation.
- **Replication / R+R track: not yet.** An R+R submission would need the original systems (Euler, Argus, Jbeil, CyberGFM) run under the budget ledger, which is what Wang et al. 2025 and Zhao 2026 already did for AUC/AP.

## 3. Strengths

1. **Leak control is enforced, not just declared** (§4.2, l. 137 to 139). It combines a timestamp guard, a static architecture test, and bit-exact row-locality checked per scored day (verified: `tests/unit/test_row_locality.py`, `tests/unit/test_no_test_leakage.py` exist). Few published LANL papers can say this. CyberGFM (King et al., Jan 2026), from the Euler authors, still uses a random 80/10/10 split on LANL (verified, its §5, "This split was infeasible…"). This is current evidence that the problem the harness targets is real.
2. **Exact reproduction of Larroche's event set** (l. 115). 369,569,626 events and 702 positives, verified against Larroche 2026 Table 4 in the arXiv PDF. This makes the AUC/AP comparison in Table 9 (l. 475 to 491) legitimate, which is rare.
3. **The exact first-detection null is correct and cheap** (Eq. 2, l. 198; verified implementation `src/authbench/evaluate/budget.py:299-312`). The user-day recall null is also correctly simulated (`scripts/paper_assets.py:564-604`). Reporting "the probability that a random ranking does at least as well" next to every b* is a good habit the community could adopt.
4. **The Euler edge-by-edge reconciliation** (l. 471) is useful forensic work: snapshot alignment, the skipped first red-team line, the unscored first test snapshot. It explains count mismatches that other re-evaluations only note.
5. **Candour.** The paper does all of the following:
   - discloses the design history (l. 141);
   - shows its first detections depend on the seed (Table 8);
   - shows that a partial first day drives the autoencoder's day-1 detection (l. 290);
   - shows that thresholds calibrated on 41 h do not transfer (Table 7);
   - limits its claims explicitly (§14).

   The README carries a correction notice for v1's false novelty claim. A committee member will trust this author.
6. **Rule-by-rule decomposition and the circularity control** (§7, §11) show concretely how a synthetic generator can validate the author's own rule. This is a good teaching example.

## 4. Major issues (ranked)

### M1. One intrusion and one source host: the main positive findings may be properties of C17693, not of detectors. *(Partly verified, partly suspected)*
- **Where:** abstract l. 56; §3.3 l. 106 to 108; §6.6 l. 336; §8 l. 443; §12 l. 475; §14 l. 557; conclusion l. 565.
- **What:**
  - 95.5% of red-team events start from one host (verified, `numbers.tex`: `RedTeamTopHostShare`).
  - The user-day result is the paper's strongest positive claim: Euler 51.2% and M2c 50.6% at 500 user-days, against 3.0% for random (verified, `table_units_harm.tex`).
  - M2c scores a new (user, source) pair as log(n_user+1) (verified, `features/novelty.py:71-107`). If every compromised account is seen authenticating from C17693, which is a new source for that account, then "source novelty finds half the campaigns" restates the fact that the red team worked from one new host. I suspect, without having verified it, that the same mechanism drives Euler's user-day lift (one shared node for unseen computers, l. 183) and the general lift of most detectors at the user-day level: PCA 41%, VAE 44%, AE 38%, ECOD 29%.
  - The intrusion-level "a week later at 50" falls on test day 8, the busiest day (45 of 162 campaigns, verified, `HarmBusiestDays`).
- **Why it matters:** these are the only results in the abstract that are not negative. A reviewer will read them as one fact about one operation, measured many times.
- **Fix:**
  - (i) For each campaign detected in user-days, report whether its top-ranked unit contains a C17693-sourced event, and recompute the user-day table with C17693 events removed, or stratified by source host.
  - (ii) Say in the abstract that the user-day result is exploratory and probably host-driven, unless (i) shows otherwise.
  - (iii) The real fix is a second dataset with several intrusions. Larroche (2026) already harmonises OpTC. LMDG (arXiv 2508.02942) reports 35 multi-stage lateral-movement attacks, though I could not confirm that it is public or contains Windows logon events.

### M2. Positioning against the closest 2025–2026 work is incomplete, and the claimed novelty is thinner than §2 suggests. *(Verified)*
- **Where:** l. 69, l. 92, l. 94 to 95, l. 71 to 77.
- **What:**
  - Wang et al. (R+R, ACSAC 2025; arXiv 2503.20281) already reproduced Euler on LANL with the authors' released code and preprocessed data, together with Argus, Pikachu and VGRNN. Their Fig. 3a shows reproduction, replication and original TPR/FPR/precision/AUC/AP. Read from the bar chart, their Euler reproduction AP on LANL is visibly below the original; this is my reading of the figure, not of a table. They also identify the random 5% validation-set threshold in Euler/Argus as a source of irreproducibility, and conclude that "AUC is insufficient to assess R+R of GIDS". The paper cites R+R only as "re-run published … detectors" (l. 92) and does not mention any of this in §9. Contribution 3 (l. 75) must be stated relative to R+R. What is new is the edge-level reconciliation and the AUC-reproduces/AP-does-not pattern under the same machine; the non-reproduction of Euler's AP from released artefacts may not be.
  - Zhao & Hassan (CCS 2026) already provide:
    - a controlled, chronologically split protocol;
    - campaign-level detection (CDR);
    - seed variance of post-threshold campaign metrics on LANL (42.1 pp for Argus);
    - threshold leakage analysis (Pikachu);
    - an encoder-free control beating graph models.

    They explicitly mention alert budgets as a concern ("rest on alert budgets an analyst would question"). The paper cites them correctly (l. 92), but the "leak-free temporal split" and "campaign view" are therefore not contributions. The title's first adjective ("Temporally Leak-Free") advertises something that is now standard in the two closest papers.
  - CyberGFM (King, Trindade, Bowman, Huang, arXiv 2601.05988, Jan 2026) is not cited (verified: not in `references.bib`). It is by the Euler authors, reports Euler AP 0.0627 on LANL under a random split, and states that a temporal split was "infeasible" on LANL. This is directly relevant to the motivation, to §9, and to the claim that leak-free evaluation matters.
  - Budgeted recall on LANL: DECADES (per day, events), ADSAGE (user-day budgets), UltraLMD++ (per hour, edges). The paper acknowledges all three (l. 69, l. 92). After subtraction, the novel elements are:
    - (1) the exact random-ranking null for b* and the user-day recall null;
    - (2) the per-campaign first-detection budget and its quantiles;
    - (3) the intrusion-day reading;
    - (4) the fixed-threshold transfer check;
    - (5) the edge-level Euler reconciliation.

    Each is useful. None is a research contribution at main-track level on its own. (1) is an elementary hypergeometric identity, and (3) operationalises Larroche 2026 §5.2 ("detecting only some of these can be considered sufficient").
- **Fix:**
  - Add a short "What is and is not new" paragraph listing (1) to (5) explicitly.
  - Rewrite contribution 3 relative to R+R.
  - Cite and discuss CyberGFM.
  - Drop "temporally leak-free" from the title, or justify why it is still a distinguishing feature, e.g. row-locality testing, which neither Zhao nor R+R does to my knowledge.

### M3. The abstract and conclusion say more than the robust part of the results. *(Verified)*
- **Where:** abstract l. 56; conclusion l. 565; scope l. 557.
- **What:**
  - *"Two other detectors notice it a week later at 50."* One of the two is PCA. In `fit_variance.json` (harmonised), PCA detects anything at 50 alerts/day in only **2 of 5 seeds** (seeds 42 and 7, b* = 20 and 13). Seeds 13, 21 and 99 give b* = 159, 137 and 243, so they notice nothing at 50 or 100. The other is the VAE, which was fitted once (l. 219). The claim is therefore one seed-fragile detector plus one untested one.
  - *"the autoencoder notices the attack on its first day at 100 alerts per day, but only because that day is partial."* A claim that the same sentence disowns does not belong in an abstract. Without day 1 its b* is 528 (`table_budgets_harm.tex`).
  - *"ROC-AUC tracks where most campaigns are found but not which detector finds the first."* In the harmonised protocol, which is the only label-free one and the only one not read before, ROC-AUC vs median b(c) is ρ = 0.55, Holm p = 0.734. AUC-PR is ρ = 0.79, Holm p = 0.10 (`table_correlations.tex`). The "tracks" half is supported only by the temporal protocol, which has four label-tuned detectors and test days seen by an earlier code version (l. 141). The "not the first" half is, by the paper's own admission (l. 313), absence of evidence with n = 10 detectors.
  - *"a graph detector and a source-novelty score reach about half of the campaigns at 500 per day."* By the paper's own staffing model, 500 units/day is 31 to 62 analysts even if a user-day cost the same as one event (l. 206; `StaffFiveHundred*`), and it is 2.6 to 4.3% of all user-days (11,636 to 19,358 per day). The abstract presents an operationally implausible budget as the positive result. At 50 user-days/day, an arguably feasible budget, the numbers are Euler 13.6% and M2c 6.8% against 0.3% random. That is still a large lift, and more honest to headline.
- **Why it matters:** PC members read the abstract and conclusion first. Over-reading here costs credibility that the body earns.
- **Fix:**
  - Rewrite the abstract around what is robust: no campaign within 10 events/day in either protocol; user-day lift over random at 50 user-days; first detection is seed-unstable; thresholds do not transfer; the Euler AP pattern; the harness itself.
  - Report "noticed by day d at k" as a fraction of seeds, not as one fit.
  - Say "in the temporal protocol" for the ROC-AUC/median claim, or drop it.

### M4. Seed variance is unknown for exactly the detectors that carry the headline results. *(Verified)*
- **Where:** l. 219, l. 309, l. 378, l. 399, l. 545; Table 8.
- **What:**
  - The autoencoder, the VAE and Euler are fitted once. These are the detectors behind the intrusion-day claims (Table 4) and the user-day claim (Table 6, Euler 51.2%).
  - The seeded detectors show order-of-magnitude swings: PCA b* 13 to 243, harmonised; IForest 20,183 to 102,041, temporal.
  - The paper itself concludes (l. 378) that "no effect of the red-team training events can be separated from fitting noise".
  - The correlation analysis (§6.4) uses one fit per detector, so its ρ has an unquantified fit-level variance on top of its n = 10 problem.
- **Fix:**
  - Run ≥ 5 seeds for AE, VAE and Euler in the harmonised protocol. At roughly 17 machine-hours in total this is affordable.
  - Report the distribution of b*, of the intrusion day and of the @50/@500 user-day recall across seeds.
  - Use seeds × detectors, or a bootstrap over campaigns, to put an interval on the Spearman coefficients in §6.4. The leave-one-out range in Appendix C is a start.
  - Incidentally, PCA's seed 42 has the lowest AUC-PR of its five fits (1.2e-5 vs ~4.5e-5) but one of the best b*. A within-detector, across-seed contrast like this illustrates the AUC-vs-first-detection point more convincingly than ten heterogeneous detectors.

### M5. The catalogue cannot support a benchmark-level message; the SOTA detectors are absent. *(Verified)*
- **Where:** l. 95, l. 541; README tagline ("What does published anomaly-detection performance become under a realistic alert budget?").
- **What:**
  - Argus, Jbeil, UltraLMD++, CyberGFM, a sequence model and the authors' Euler are all missing. The only graph detector is a reimplementation of Euler that falls below the authors' code on its own protocol (Table 10: our NTLM 0.9570 / 0.0027 vs authors' 0.9879 / 0.0058), and whose vocabulary configuration collapses (AUC 0.826) for an undiagnosed reason (l. 448).
  - The rest are generic outlier detectors on a 25-feature vector.
  - A venue reader will ask what "published performance" becomes, and the paper cannot say, because published detectors are not run through the ledger.
  - The author did run the authors' Euler code on this machine (§9). Its per-edge scores on the harmonised universe could be fed to the budget ledger without making it row-local in the harness sense, since the snapshot-causal scoring already avoids future leakage (l. 139).
- **Fix:**
  - Put at least the authors' Euler, and ideally Argus (code public, already re-run by Larroche and by Wang et al.), through the harmonised budget ledger, even as an "external scores" path with a weaker row-locality guarantee that is stated as such.
  - Until that is done, change the README tagline and anything implying an evaluation of "published" detectors.

### M6. The Euler non-reproduction claim is under-diagnosed for a claim placed in the abstract. *(Verified for the facts; the interpretation is my judgement)*
- **Where:** abstract l. 56; §9 l. 450; ethics l. 569.
- **What:**
  - The authors' code sets no seed and was installed with pip rather than conda, with a requirements file added after the paper (l. 450).
  - The released preprocessing keeps a different number of events and nodes from the paper's Table V.
  - Two candidate explanations are named but not measured: float32 saturation of AP at the top, and a different edge alignment.
  - The one-sample t-test treats the published AP (one number, likely one run) as a fixed constant. p = 2.6×10⁻⁴ over five runs is therefore a statement about the rerun's spread, not about whether the published run was attainable.
  - The authors have not been contacted (l. 569).
  - R+R already reported discrepancies for Euler (see M2).
- **Fix:**
  - Contact King and Huang before the next version.
  - Measure the float32 effect, which is cheap: recompute AP in float64 on the saved scores if they exist, or rerun once with float64.
  - Drop the t-test, or describe it as "the published AP lies k standard deviations outside our runs".
  - Move the Euler sentence out of the abstract unless the diagnosis is completed.

### M7. The harmonised evaluation is not fully confirmatory, and the paper's own design-history disclosure undercuts the confirmatory reading. *(Verified from the text)*
- **Where:** l. 141, l. 443, l. 543.
- **What:**
  - Source-side features (M2c, R8), the user-day and edge-hour units, and the day-anchored rarity were added after the temporal test results were seen.
  - The temporal test days (11 to 13) lie inside the harmonised test window, and the operation is the same.
  - The paper says this honestly, but the abstract and conclusion still present the user-day/source-novelty result as a finding.
- **Fix:**
  - Label these as hypotheses generated on LANL, or test them on a dataset that the design history has not touched (see M1).
  - A dated pre-registration of the analysis plan (OSF or Zenodo) before the next dataset is run would turn the next case into confirmatory evidence. This is cheap and would be noticed favourably at CSET or LASER.

## 5. Minor issues

- **l. 41 to 43, title.** "AuthBench" plus "benchmark" (README) for one dataset and one intrusion. Consider "harness" in the title and the README, as the paper body already does.
- **l. 101.** The raw files come from a mirror, with content verified on 3 of 58 days plus the total row count. The exact match to Larroche's 369,569,626/702 is strong indirect evidence. Say so here, since it is the best verification available.
- **l. 104 and 551.** "Campaign" here means account-episode (one user, gaps ≤ 24 h; 65 of 162 are singletons). Zhao 2026 uses "campaign" for attack campaigns. Use a distinct term (e.g. "account episode") to avoid an apples-to-oranges comparison with CDR.
- **l. 115 and 133.** The temporal protocol uses validation labels from the same operation to tune four detectors. Larroche (2026, §5.2) states that LANL "should thus only be used to evaluate unsupervised learning methods". Its headline numbers (M1 at 42) are of limited value. Consider demoting the temporal protocol to an appendix and making the harmonised protocol the paper.
- **l. 150.** Sampling 5M of 10.7M training events "adds fitting variance" for no stated benefit, since all would fit. Remove the sampling in the harmonised protocol.
- **l. 200 and 309.** Many null probabilities are reported across detectors, budgets, units, protocols and with/without day 1. Holm is applied within some families but not across them. Add one sentence on the overall testing family, or state that these are descriptive.
- **l. 206.** The staffing inputs are a vendor-commissioned survey and a self-reported preprint. This is acknowledged. A peer-reviewed source on per-alert triage time would strengthen it (e.g. the Alahmadi et al. 2022 interviews already cited may give ranges).
- **l. 313.** "Only correlations above about 0.65 can be distinguished from zero." Give the exact critical value for n = 10 and n = 11 at the Holm-adjusted level, not "about".
- **l. 358 / Table 7.** M2a raises 58,396 alerts/day with a "10/day" threshold, because the threshold falls inside a tie. Report it as not applicable rather than as a recall of 89.5%, which a skimming reader will misread.
- **l. 475.** "Our Euler … average precision between theirs." Their numbers are means of ten runs and ours is one fit; say so in the sentence, not only in the caption.
- **l. 569.** "Their answer will be reported." In a submitted version this must be resolved one way or the other.
- **README l. 1 to 4 and l. 22.** "What does published anomaly-detection performance become…?" and "AuthBench does not propose a new model: it measures … what survives." These overclaim relative to the catalogue (M5).
- **Length and format.** 29 pages, with a table of contents and the dense, numbers-heavy style of a technical report. For any venue this must be cut to the contribution, the protocol, three tables and one figure. The current abstract (l. 56) contains about 15 distinct quantitative claims; a PC reader retains three.

## 6. Questions to the author

1. What AP did Wang et al. (R+R, ACSAC 2025) obtain for their Euler reproduction on LANL, and is your non-reproduction consistent with theirs? If they reproduced it, what differs?
2. Among campaigns detected at 500 user-days by M2c and Euler, how many have their top-ranked user-day containing an event sourced from C17693? What does the user-day table look like with C17693 events excluded?
3. Why is 500 user-days per day the headline, when your own staffing model prices it at 31 to 62 analysts?
4. Do you still have the authors' Euler per-edge scores? Could they be pushed through the harmonised ledger as an "external scores" detector?
5. Is OpTC under Larroche's (2026) harmonised policy feasible on your hardware? It would turn "one intrusion" into "two datasets", and his paper already supplies the event set.
6. Across PCA's five seeds, AUC-PR and b* move in opposite directions (seed 42: lowest AUC-PR, b* = 20). Is this systematic across detectors? A within-detector analysis would address the n = 10 limitation of §6.4.
7. Was the harmonised analysis plan (metrics, budgets, units, nulls) fixed in a dated commit before the harmonised run? If so, cite the commit; it partly answers M7.
8. Have you contacted King and Huang? Has the CyberGFM paper by the same group changed their reported Euler numbers on LANL?

## 7. What would move it up one level

- **From Master's-level "accept" to a solid PhD-workshop paper (CSET, LASER, AISec):**
  - Rewrite the abstract and conclusion around the robust findings (M3).
  - Add the "what is and is not new" paragraph, with R+R and CyberGFM discussed (M2).
  - Run seeds for AE, VAE and Euler (M4).
  - Run the C17693-stratified user-day analysis (M1, step i).
  - Contact the Euler authors (M6).
  - Cut to 6 to 8 pages, foregrounding the harness, the exact nulls and the negative results.

  This is about two to three weeks of work and no new data.
- **From workshop to main track (ACSAC, RAID, DIMVA):** all of the above, plus two of the following three:
  - (i) a second dataset with several independent intrusions, under the same harness (OpTC via Larroche's policy; LMDG or similar if public; or an industrial partner's logs);
  - (ii) the SOTA lateral-movement detectors (authors' Euler, Argus, Jbeil, CyberGFM, ideally UltraLMD++) through the same budget ledger, so that the paper answers "what does published performance become under a budget";
  - (iii) a result that generalises, e.g. a pre-registered test, across intrusions and datasets, of whether threshold-free metrics predict first-detection budget, with enough detectors × seeds × intrusions for the test to have power.

  Without (i), every finding remains "a case study of one operation", which is the paper's own accurate self-description and the reason it is not yet a main-track paper.
