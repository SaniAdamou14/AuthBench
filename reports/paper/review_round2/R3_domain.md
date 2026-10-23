# Review round 2, Reviewer 3 (domain: lateral-movement and authentication anomaly detection on LANL)

Paper: "Detection Under an Alert Budget: A Leak-Free Case Study of Authentication Anomaly Detectors on the LANL Data, with a Reproduction of Euler" (main.pdf, 23 pp., October 2026). Run: `reports/runs/20261022T102638Z`.

## Summary (5 lines)

1. The paper presents AuthBench, a harness that ranks LANL authentication events per day (plus edge-hours and user-days), reports each red-team campaign's first-detection budget and the exact random-ranking null for it.
2. Thirteen detectors (floors, rules, rarity, PyOD outlier detectors, AE/VAE, an Euler reimplementation) are run under Larroche (2026)'s 41-hour protocol and a legacy day-based split.
3. At the event level no detector catches a campaign within 10 alerts/day. PCA, VAE and AE reach first detections a random ranking rarely matches, and ROC-AUC does not order detectors by first detection. In user-days, Euler and source rarity reach about half of the campaigns at 500/day.
4. The authors' Euler code reproduces the published AUC but not the published AP. An edge-by-edge reconciliation explains 518 vs 514 vs 525 anomalous edges.
5. The labelled activity is one intrusion (95.5% from C17693). The paper scopes itself as a case study. The engineering and transparency are excellent. The empirical delta over Larroche (2026) and Zhao and Hassan (2026) is small, and the detector catalogue leaves out the detectors those works already re-evaluated under the same protocol.

## Scores (1 to 5)

| Criterion | Score | One-line reason |
|---|---:|---|
| Soundness | 3 | The harness, labels and null are correct (I recomputed the null and the universe matches Larroche's exactly). Three problems remain: a partial first test day biases day-1 detections, the SOC-relevant intrusion-level metric is missing, and the cross-paper comparison mixes statistics. |
| Presentation | 3 | Clear and honest, but very dense. The abstract carries about 15 numbers, and the headline metric (campaign recall) is not the one the paper's own one-intrusion argument calls for. |
| Contribution | 2 | The harness and the Euler reconciliation are real contributions. The empirical findings overlap heavily with Larroche (2026) (same protocol, same 702 positives, Euler/Argus already re-evaluated) and Zhao and Hassan (2026) (campaign detection rate, alert-budget recall, seed variance on LANL), neither of which is fully engaged. |
| Reproducibility | 5 | Exceptional: committed run outputs, per-positive ledger, the authors' code run beside the reimplementation, and recounts from the authors' own loader. |

## Recommendation

- **Top-tier security venue (S&P, USENIX Security, CCS, NDSS): reject in current form.** No result is wrong in a way that invalidates the paper. But the empirical claims rest on one intrusion and on a catalogue without the state of the art, and the methodological novelty (budgeted and campaign-level evaluation on LANL) is partly pre-empted by Larroche (2026) and Zhao and Hassan (2026, CCS). A resubmission becomes competitive only with the three changes at the end.
- **arXiv/Zenodo preprint: ready after a short minor revision.** Before posting, do four things: cite Larroche (2026) Table 4/5 and state that the harmonised universe equals his to the event (369,569,626 events, 702 positives); disclose the partial first test day; correct the DECADES and ADSAGE descriptions (M5); and cite Zhao and Hassan's CDR, R+R and Boucek and Husák. None of these needs a new run.
- **Realistic venue tier now:** a specialised workshop such as ANUBIS@ESORICS (where Larroche 2026 appeared), AISec@CCS, DLS/WOOT-adjacent workshops or CSET@USENIX. A reproducibility or practice journal also fits: ACM DTRAP, or Computers & Security, ideally under a "replication" framing like Boucek and Husák (CNSM 2025). With Argus and UltraLMD++ in the harness, intrusion-level time-to-detection, and a second dataset (OpTC with Larroche's labels), ACSAC, RAID or EuroS&P would become realistic.

## Strengths

- **S1. The universe matches the reference protocol exactly.** Harmonised train plus test is 10,705,481 + 358,864,145 = 369,569,626 events with 702 positives. That is exactly Larroche (2026) Table 4 ("LANL 16,159 / 369,569,626 / 702"). The paper never says so (see M1), but it is strong evidence that the preprocessing is correct.
- **S2. The null is exact and correct.** I recomputed Eq. 2 from `harmonised/tables/day_sizes.csv` and `temporal/tables/day_sizes.csv`. I get P(b*_random <= b*) = 0.020 (Euler), 0.002 (PCA), 0.003 (VAE), 0.006 (AE), 0.040 (M2a), 0.070 (IF), 0.314 (M2c) in the harmonised protocol, and 0.200 (Euler), 0.0006 (M1), 0.003 (PCA) in the temporal one. These match Tables 3 and 4.
- **S3. The Euler reconciliation is genuinely new.** It covers the skipped first red-team line labelled benign, the 518 vs 517 vs 514 edges, and the unscored first snapshot. This is the kind of evidence replication papers rarely give.
- **S4. Exact-join labelling** on (time, user@domain, src, dst), with the 14 unmatchable red-team lines stated.
- **S5. Honest scoping.** The one-intrusion caveat is in the abstract, and Section 13 lists what is not shown.
- **S6. Equal tuning budgets, M1u without labels, R1/R2 regimes, seed refits and user-clustered bootstrap.** All round-1 statistical requests are run, not just promised.

## Fatal issues (for a top-tier venue; not fatal for the preprint)

**F1. Novelty and positioning against two 2026 papers that already did most of the empirical work (Sections 1, 2, 9, 10).**
- Larroche (2026) uses the identical protocol and universe (S1). He re-evaluates Euler and Argus at the event level with the authors' code: "Euler [17] 97.97±0.17 [AUC] ... 0.02±0.00 [AP, %]" and "Argus [30] 98.44±0.78 ... 0.09±0.04" (Table 5). He also argues that, for one intrusion, "detecting only some of these can be considered sufficient" (Section 5.2).
- Zhao and Hassan (2026) define a campaign detection rate: "We define CDR as |Ĉ|/|C|" (Section 4). They report recall "against the alert budget" on LANL (Fig. 8) and seed ranges of CDR on LANL ("Argus on LANL has ... a CDR range of 42.1 percentage points"). Their encoder-free control GIDS-Lite is "a 12-component PCA reconstruction head", which is strikingly close to this paper's best first detector, M2b PCA.

The paper cites both but uses neither's numbers or concepts. As written, the empirical contribution ("leak-free LANL, budgets, campaigns, seed variance") reads as a re-derivation.

Fix: make the harness plus the intrusion-level budget analysis the explicit delta. Put Larroche's Table 5 numbers into Tables 3 and 11. Run Argus and the authors' Euler through the harness (M2). Position campaign recall against CDR (yours is per daily budget; theirs is at a calibrated threshold).

## Major issues

**M1. Larroche (2026) Table 4/5 are missing from Sections 4.1, 9 and 10. This is the only published result under the identical protocol.**
- State in Section 4.1 that the universe equals his to the event (S1).
- In Section 10 / Table 11, add a row: Larroche (2026), same universe and split, event level, Euler (authors' code, link-detection variant) AUC 0.9797 / AP 0.02%, Argus 0.9844 / 0.09%, Pikachu 0.7828 / 0.00%.
- Compare these with your reimplementation in the same universe: ROC-AUC 0.961, AUC-PR 7.19e-4 = 0.072% (Table 3). Your reimplementation has a higher AP and lower AUC than the authors' code under Larroche's preprocessing. That is worth one paragraph, noting that the variants differ (yours is prediction, his is detection).

**M2. The detector catalogue is not representative enough for "thirteen detectors" to carry the conclusions (Sections 5, 7, 13, 15).**
- **Missing detectors.** Argus (S&P 2024; the state of the art per Larroche 2025 and 2026), UltraLMD++ (Larroche 2025), Pikachu (improved implementation by Boucek and Husák), Jbeil (S&P 2024), DECADES, ADSAGE, and the LANL-classic sequence family are all absent. The sequence family covers Tuor et al. (2018), Brown et al. (2018) and Gökstorp et al. (2024), the latter reporting AUC 0.9989 and AP 0.0410 on LANL. Euler appears only as a reimplementation that Section 9 shows is about 3 AUC points and half the AP below the authors' code.
- **The row-locality contract silently excludes detector families.** `docs/methodology.md` says the LSTM sequence model M4c "declares itself not row-local and is refused by the day-by-day scorer". The paper never mentions M4c. Current-snapshot graph detectors also do not fit the contract as stated. Argus, Euler-DET and UltraLMD++ score an edge using its own window, and Larroche (2025) states UltraLMD++ "needs to observe an entire graph in order to compute ... and score its edges". Yet Euler is admitted via "prepare along the timeline, then look up". The same wrapper, with a stated latency of one window, admits all of them.
- Fix:
  1. Add Argus and the authors' Euler under the harmonised protocol. Larroche's LMDEval already runs them on exactly this event set, so it is enough to export per-event scores and rank them in the harness.
  2. Add one sequence model, either M4c made row-local by the same float64 per-row evaluation used for the autoencoders, or Gökstorp's.
  3. Add a label-free lexicographic novelty baseline (new (user, dst) pair, ties broken by (user, src) novelty, then by per-user count of new pairs that day). It is closer to Bowman's UA and Garchery's "known source pc" than M2a, whose ranking is decided by 217,692 ties, or M1u, which sums three wrongly signed rules.
  4. Disclose in Section 5 every detector built but excluded, and why.

**M3. The partial first test day biases day-1 detections (Section 4.1, Table 3, Section 7.2).**
- Daily ranking uses calendar days (floor(t/86400)). The harmonised test starts at hour 41, so test day 1 is a 7-hour day. `harmonised/tables/day_sizes.csv` gives 1,653,645 events on day 1 against 5.0 to 7.9 million on the other days.
- From `positives_ledger.csv`, four first detections fall on day 1: the autoencoder (b* = 59), Euler (203), HBOS (1,262) and M1u (41,065). A real SOC triaging day 1 would see the full day, about 3 to 4 times more competition. The exact null accounts for N_d, so the p-values are internally consistent, but the operational reading ("the autoencoder detects within 59 alerts/day") is not.
- Fix: define test "days" as consecutive 24-hour windows from the test start (hour 41, 65, ...), or rank day 1 together with the scored events of hours 24 to 41. Report the sensitivity of b*, Table 3 and Figure 2.

**M4. For one intrusion, the headline metric should be intrusion-level time-to-detection, not campaign recall (Abstract, Sections 3.2, 6.1, 7.1, 13, 15).**
- The paper accepts that "the effective number of independent intrusions in this evaluation is one" (Section 3.3). Larroche (2026) draws the operational consequence: "detecting only some of these can be considered sufficient; further investigation by a security analyst can then uncover the rest." Under that reading, "none exceeds 9.6% of the campaigns at 500" is the wrong headline.
- The campaigns are also dominated by a few days. 65 of 162 harmonised campaigns are singletons, and 106 of 162 fall on days 8 (45), 12 (39) and 13 (22), all mostly from C17693 (computed from the ledger).
- From the ledger, the first day on which any red-team event enters the top k (harmonised) is:
  - autoencoder: day 1 at k = 100 (the attack's first day, subject to M3);
  - Euler: day 1 at k = 500;
  - PCA and VAE: day 8 at k = 50, i.e. a week after the attack starts;
  - M2a: day 14 at k = 500;
  - every other non-floor detector: nothing at k <= 500.

  This is the SOC quantity (dwell time under a staffing level, as in Bowman et al.'s M-Trends motivation and Hopper's alerts/day), and it orders the detectors differently from b*.
- Round 1 removed "time to detection" because it was presented poorly. It should come back as the primary intrusion-level metric, defined as the first day d at which some red-team event of the operation is in the top k of day d, with its exact null. Campaign recall can stay as a secondary account-level coverage measure.

**M5. Table 11 and Section 2 misdescribe two published setups, and the table juxtaposes non-comparable statistics.**
- **DECADES.** The paper says Larroche et al. "keep remote LogOn events whose destination is a user account". The source keeps local and remote LogOn events and process creations: "We distinguish two kinds of authentications: local ones ... and remote ones" and "As for processes, we only include process creations". DR-k is event recall: "the proportion of red team events detected by extracting the k most anomalous events each day" (Table 4). In the same table CADENCE has the higher DR-1K: "0.093±0.034" against DECADES "0.081±0.032".
- **ADSAGE.** CR-1000 is a normalised cumulative recall, an area under the daily recall curve: "CR_k = sum_{i<=k} R_i / n ... an approximation of the area under recall curve". R_k is "the recall of malicious users per day, averaged over all test days". It is not campaign recall at a budget.
- **ADSAGE day indexing.** ADSAGE's days are 1-indexed: "days 1 to 8 as train set (44.2M events, 50 anomalies)". Those 50 anomalies equal the 50 positives of this paper's temporal training days 0 to 7. So ADSAGE's and DECADES's test is 0-indexed days 8 to 12, not "9 to 13" in the paper's indexing.
- **ADSAGE labelling.** ADSAGE also "merge[s] normal and red team events" and reports 587 test anomalies, against 473 labelled positives by exact join on days 8 to 12 here (266 + 207).
- **UltraLMD++.** Only remote-authentication edges are scored ("only host-host edges representing ... remote authentications in the LANL dataset are given anomaly scores"). Flows are context. The row should also give the split (first 41 hours to train Argus, all 58 days tested).
- Fix:
  1. Correct the descriptions.
  2. Add a legacy split (train days 0 to 7, test days 8 to 12, 0-indexed).
  3. Compute, on your own rankings, DR-1K (event recall at 1,000 events/day), ADSAGE's CR-1000 on user-days (daily user recall averaged over days with positives, normalised area), and UltraLMD++'s Rec@10 per hourly window after deduplicating lateral-movement edges (harmonised).

  The harness already holds every rank, so this costs no new fitting and turns Table 11 into a real comparison.

**M6. The alert units and staffing model are not unit-specific (Sections 6.1 "What a budget costs", 7.4, Table 6).**
- The staffing model prices every alert at 10 to 20 minutes whatever the unit. A user-day investigation means reviewing a user's whole day. Garchery and Granitzer justify user-days precisely because "the investigator will have to review the entire user activity (at least the whole user-day)". That is much costlier than dismissing one event.
- 500 user-days is 2.6 to 4.3% of all active accounts per day (11,636 to 19,358). That is why the random floor reaches 4.3%, and why "Euler 51.2%" is not comparable with event budgets.
- User-days include machine accounts, which make up 66.8% of benign event sources. Their share of the top 500 should be reported.
- Fix:
  - Price each unit separately, or report user-day budgets as a share of daily units.
  - Add a source-host-day unit. It is the unit EDR alerts naturally aggregate to, and with 95.5% of positives on C17693 it will show how much any unit-level result depends on one host. That is an informative fragility result.
  - Move the user-day headline out of the abstract, or qualify it there.

**M7. The Euler reproduction is not placed against prior replications, and the NTLM filter needs flagging (Section 9).**
- Larroche (2026) Fig. 3 already shows the authors' Euler code under its original preprocessing reaching an AP between roughly 0.1% and 0.4%, against the reported 5.23% (detection variant). R+R (Wang et al., ACSAC 2025) and Boucek and Husák (CNSM 2025) are replication studies of exactly these detectors. Larroche summarises them as having "uncovered ... errors in the published implementations of the detectors".
- Your prediction-variant result (0.0058 vs 0.0155) and the edge reconciliation are new and should be framed as extending these.
- Section 9 calls the NTLM-only subset "as the authors' loader keeps" without saying that the filter is label-informed. The run's own `euler_original/redteam_matches.out` says the 701 matchable red-team events are "all of them NTLM". Larroche (2026, Section 4.2) writes: "Because lateral movements in the LANL dataset happen to leverage NTLM, filtering out other authentication packages makes detection easier". State this, and note that the published AP being matched comes from a leaky universe.

## Minor issues

- **m1. Hopper is overstated (Section 1).** Hopper "deployed in a real enterprise has been reported at fewer than nine alerts per day". Hopper was evaluated retrospectively: "On a 15-month enterprise dataset consisting of over 780 million internal logins, Hopper achieves a 94.5% detection rate across over 300 realistic attack scenarios, including one red team attack". Replace "deployed" with "evaluated on 15 months of enterprise logins".
- **m2. Zhao and Hassan sentence (Section 2).** Add that LANL is not one of the two datasets where the encoder-free control leads: "Pikachu on Campus and VGRNN on LANL" win there. Note too that they report AP up to 0.620 on LANL under their aligned contract (VGRNN), in a different universe. Readers will ask why that is three orders of magnitude above Table 3, so one sentence on edge-level vs event-level and window-level labelling is needed.
- **m3. Table 2 caption and abstract wording.** "Scores every event independently of the others" is not literally true. History features and Euler depend on earlier events, and Euler's previous-snapshot embeddings include test events. Write "causally, and independently of which other events are scored in the same batch".
- **m4. The two protocols differ in both split and universe (Section 4.1).** The temporal protocol uses all events, the harmonised one remote LogOn only, so cross-protocol statements such as "the same shift is smaller" (Section 7.4) confound the two. State this, or add a temporal run with the remote-LogOn universe.
- **m5. Section 7.1.** Put the null at 10 alerts/day next to the zeros. P(b*_random <= 10) = 0.0010 in the harmonised protocol (my computation from `day_sizes.csv`), so "no detector at 10" carries little information on its own.
- **m6. `RUN.md` disagrees with the paper on null values.** Examples: Euler 0.022 vs 0.020, M2c 0.309 vs 0.314, temporal Euler 0.203 vs 0.200, M3a 0.757 vs 0.751. My exact recomputation agrees with the paper. Regenerate `RUN.md` from the same source so readers of the run directory are not misled.
- **m7. Report the campaign structure (Section 3.2).** Give the size distribution (65 singletons; one campaign of 83 events) and the per-day concentration (45 campaigns on day 8). Campaign recall is close to "share of accounts the operation used on days 8, 12 and 13".
- **m8. Ties, Section 7.8.** M2a's upper bracket (32.1% at 10/day) shows that the "new pair" tier contains the red team. The arrival-order tie-break is label-free but arbitrary, and a deployed rule would use a secondary key. Present M2a's result as a property of the score's resolution rather than of novelty-based detection, or add the lexicographic baseline (M2).
- **m9. `diagnostics/redteam_concentration.json`.** The "harmonised_test" block reports 715 events and 104 accounts, the same as "all", while the harmonised test has 702 positives over 98 accounts. Fix the label or the filter.
- **m10. Section 2, Tuor et al. (2018).** Their primary reported metric is ROC-AUC (Table 2; e.g. "C BEM log-line 0.98 AUC"), with average percentile as the second. "Not average precision" is right, but readers should not infer that they avoided ROC-AUC.
- **m11. Euler citation.** Larroche (2026) cites the TOPS version (King and Huang 2023, ACM TOPS 26(3)), whose LANL numbers equal NDSS Table VI. Cite TOPS once so readers can match "99.12 / 5.23" (detection) and "0.9906 / 0.0155" (prediction) to one source.

## Citation checks

| # | Claim in the paper (location) | Source checked | Verdict | Evidence |
|---|---|---|---|---|
| 1 | Larroche (2026) recommends the first 41 h for training, the rest for testing, all remote LogOn events (Section 2, 4.1) | arXiv 2607.29390 v1, Section 4.2 | Correct | "our main recommendation is thus to include all remote LogOn events from the testing period"; "We thus recommend training on the first 41 hours and testing on the rest of the dataset." |
| 2 | Argus on LANL AP 32.27 reported vs 0.09 ± 0.04 re-evaluated, in percent, Table 5 (Section 2) | same, Table 5 | Correct | "Argus [30] 98.44±0.78 99.83 0.09±0.04 32.27" |
| 3 | Labelled LANL authentications are one intrusion, mostly from one host (Section 2, 3.3) | same, Section 5.2 | Correct | "all labeled malicious authentications in the LANL dataset are part of one single (simulated) intrusion; in fact, almost all of them come from the same source host (C17693)" |
| 4 | LANL should be used only for unsupervised methods (Section 2) | same, Section 5.2 | Correct, slightly abridged | "should thus only be used to evaluate unsupervised learning methods (or detectors trained in a supervised manner on another dataset)" |
| 5 | Harmonised protocol "follows Larroche (2026)" (Section 4.1) | same, Table 4 | Correct, and the universe matches to the event (not stated in the paper, M1) | "LANL 16,159 369,569,626 702" |
| 6 | Zhao and Hassan decompose GIDS into interchangeable stages; an encoder-free control ranks first by AP on two of four datasets (Section 2) | arXiv 2609.12263 v1 | Correct, incomplete (m2, F1) | "decomposes each GIDS into six interchangeable stages"; "A single encoder-free configuration leads on two of four datasets"; on LANL VGRNN wins |
| 7 | Zhao and Hassan accepted at CCS 2026 (bib) | same, ACM reference block | Consistent | "CCS '26, November 15-19, 2026, The Hague, Netherlands" |
| 8 | ADSAGE reports cumulative recall over user-day budgets, train days 1 to 8, test 9 to 13 (Section 2, Table 11) | arXiv 2007.06985, Sections 4.1.1 and 4.3 | Numbers correct; metric and indexing need care (M5) | "We use days 1 to 8 as train set (44.2M events, 50 anomalies) and days 9 to 13 (27.7M events, 587 anomalies)"; "CR_k ... approximation of the area under recall curve" |
| 9 | "Known source pc" rule on par with ADSAGE; CR-1000 0.254 and 0.255 (Section 2, Table 11) | same, Table 5 | Correct | "known source pc 0.254 0.669 0.890"; "ADSAGE 0.255 0.653 0.877"; "not statistically significant" |
| 10 | Tuor et al. (2017) introduced daily-budget recall on insider-threat data (Section 2) | arXiv 1710.00811 | Correct | "We assume that there is a daily budget which imposes a maximum number of user-day pairs that can be judged each day"; ADSAGE: "we use recall-based metrics introduced in [36]" |
| 11 | DECADES keeps "remote LogOn events whose destination is a user account", trains on 8 days, tests on 5, DR-1K = 0.081 (Section 2, Table 11) | arXiv 2103.15708 (CAID 2020), Sections 5 to 6, Table 4 | Partly wrong: universe misdescribed (M5); number correct | "we only consider 'LogOn' events for which the destination user is a user account ... We distinguish two kinds of authentications: local ones ... and remote ones"; "Decades 0.497±0.085 0.081±0.032 ..."; CADENCE DR-1K 0.093±0.034 |
| 12 | UltraLMD++ reports recall at the top B edges per hourly window; higher ROC-AUC does not imply higher budgeted recall; Rec@10 = 0.1788 after deduplication (Section 2, Table 11) | arXiv 2504.13527, Section 4.3, Table 3 | Correct | "Rec@B is the proportion of lateral movement edges that are detected when the top B most anomalous edges in each time window are investigated"; dedup row "UltraLMD++ .9868 .0088 .0629 .0927 .1788"; with duplicates, Argus AUC .9749 > UltraLMD++ .8938 but Rec@10 .0900 < .1040 |
| 13 | Euler: published AUC 0.9906, AP 0.0155 (GCN-GRU, link prediction), 518 anomalous edges, 45,871,390 events (Section 9, Table 10) | NDSS 2022 PDF, Tables V and VI | Correct | Table V "Events 45,871,390 ... Anomalous Edges 518"; Table VI Dynamic Link Prediction "GCN GRU 0.9906 0.0155 85.49 0.6088 0.0050" |
| 14 | Euler trains on snapshots before the first anomalous edge; ROC-AUC not a good indicator on LANL (Section 1, 2) | same, Section VI | Correct | "the training set consists of all snapshots that occur before the first anomalous edge"; "this metric is not a good indicator of model quality on data sets with imbalance as extreme as LANL" |
| 15 | Bowman et al. train on 40 days without malicious activity and test on the 18 days with it (Section 2) | RAID 2020 PDF, Section 5 | Correct | "The training set consists of 40 days on which no malicious activity is reported, and the testing set of 18 days with malicious activity." |
| 16 | Tuor et al. (2018): operational metric is the average percentile (Section 2) | arXiv 1712.00557 / AAAI-W 2018 | Correct (m10) | "we use the Average Percentile (AP) metric ... the percentile of its anomaly amongst all anomaly scores for the day" |
| 17 | Hopper "deployed in a real enterprise", fewer than nine alerts/day (Section 1) | USENIX Sec 2021 page | Number correct; "deployed" overstated (m1) | "On a 15-month enterprise dataset consisting of over 780 million internal logins ... generating an average of <9 alerts per day" |
| 18 | Gökstorp et al. (2024) transformer sequence models on LANL (Section 2) | KTH PDF (people.kth.se/~gyuri/Pub/GokstorpNKJD_LogAnomaly_NOMS24.pdf), via search summary | Correct | Self-supervised transformer on LANL authentication; reported AUC 0.9989, AP 0.0410 (relevant to M2) |
| 19 | Paper's exact null values (Tables 3 and 4) | recomputed from run `day_sizes.csv` | Correct | Harmonised Euler 0.0202, PCA 0.0019, VAE 0.0026, AE 0.0058, M2c 0.3144; temporal Euler 0.2004, M1 0.0006 |
| 20 | 702 positives, 162 campaigns, 98 accounts; red-team share 95.5% from C17693 (Section 3) | run `positives_ledger.csv`, `redteam_concentration.json` | Correct | 9,126 ledger rows / 13 models = 702; 162 campaign ids; 683/715 = 0.955 |

## Missing related work (full references)

1. Wang, C., Zheng, P., Gui, J., Hua, C., & Hassan, W. U. (2025). R+R: From claims to crashes: A systematic re-evaluation of graph-based network intrusion detection systems. *2025 IEEE Annual Computer Security Applications Conference (ACSAC)*, 499-515. https://doi.org/10.1109/ACSAC67867.2025.00050 (re-runs Euler and others end to end; cited by both Larroche 2026 and Zhao and Hassan 2026).
2. Boucek, V., & Husák, M. (2025). Replication: Network-based lateral movement detection methods using machine learning. *2025 21st International Conference on Network and Service Management (CNSM)*. https://doi.org/10.23919/CNSM67658.2025.11297412
3. Khoury, J., Klisura, Đ., Zanddizari, H., De La Torre Parra, G., Najafirad, P., & Bou-Harb, E. (2024). Jbeil: Temporal graph-based inductive learning to infer lateral movement in evolving enterprise networks. *2024 IEEE Symposium on Security and Privacy (SP)*, 3644-3660. https://doi.org/10.1109/SP54263.2024.00009
4. Larroche, C. (2024). Inductive lateral movement detection in enterprise computer networks. *ESANN 2024 Proceedings*, 387-392. https://doi.org/10.14428/esann/2024.ES2024-19 (UltraLMD, the predecessor of UltraLMD++).
5. King, I. J., & Huang, H. H. (2023). Euler: Detecting network lateral movement via scalable temporal link prediction. *ACM Transactions on Privacy and Security, 26*(3), 1-36. https://doi.org/10.1145/3588771
6. Bilot, T., Jiang, B., Li, Z., El Madhoun, N., Al Agha, K., Zouaoui, A., & Pasquier, T. (2025). Sometimes simpler is better: A comprehensive analysis of state-of-the-art provenance-based intrusion detection systems. *34th USENIX Security Symposium (USENIX Security 25)*. https://www.usenix.org/conference/usenixsecurity25/presentation/bilot (simple-baseline finding; origin of the attack-level metric that Zhao and Hassan's CDR builds on).
7. Tariq, S., Baruwal Chhetri, M., Nepal, S., & Paris, C. (2025). Alert fatigue in security operations centres: Research challenges and opportunities. *ACM Computing Surveys*, 1-38. https://doi.org/10.1145/3723158 (a better-grounded source for the staffing and alert-fatigue model than a vendor survey).
8. Lee, W., McCormick, T. H., Neil, J., Sodja, C., & Cui, Y. (2022). Anomaly detection in large-scale networks with latent space models. *Technometrics, 64*(2), 241-252. https://doi.org/10.1080/00401706.2021.1952900
9. Tang, M., & Ge, B. (2025). Social-Hunter: A social heuristics-based approach to early unveiling unknown malicious logins using valid accounts. *Computers & Security*, 104269. https://doi.org/10.1016/j.cose.2024.104269 (early detection of malicious logins, directly relevant to M4).
10. Powell, B. A. (2020). Detecting malicious logins as graph anomalies. *Journal of Information Security and Applications*, 102557. https://doi.org/10.1016/j.jisa.2020.102557
11. Veeramachaneni, K., Arnaldo, I., Korrapati, V., Bassias, C., & Li, K. (2016). AI^2: Training a big data machine to defend. *IEEE BigDataSecurity/HPSC/IDS 2016*, 49-54. https://doi.org/10.1109/BigDataSecurity-HPSC-IDS.2016.79 (a fixed number of analyst investigations per day; already in the author's candidate list).
12. Oprea, A., Li, Z., Yen, T.-F., Chin, S. H., & Alrwais, S. (2015). Detection of early-stage enterprise infection by mining large-scale log data. *2015 45th IEEE/IFIP DSN*, 45-56. https://doi.org/10.1109/DSN.2015.14
13. Gutflaish, E., Kontorovich, A., Sabato, S., Biller, O., & Sofer, O. (2019). Temporal anomaly detection: Calibrating the surprise. *Proceedings of the AAAI Conference on Artificial Intelligence, 33*, 3755-3762. https://doi.org/10.1609/aaai.v33i01.33013755 (per-entity score calibration for ranked queues; cited by UltraLMD++ for this purpose).
14. Majorczyk, F., Pilastre, B., & Dijoud, F. (2025). A new hope for DARPA OpTC. *ACSAC Workshops 2025* (as cited by Larroche 2026, ref. 21). This provides OpTC labels for a second, multi-scenario dataset.

## Status of round-1 domain points (R3)

| Round-1 point | Claimed status | Verified status now |
|---|---|---|
| M1 Title and abstract generalise from n = 5 and one intrusion | Addressed | **Resolved.** Retitled as a case study; one intrusion in the abstract and Section 13. |
| M2 b* has no uncertainty; b(c) not published | Run | **Mostly resolved.** b(c) per positive is in `positives_ledger.csv`; the one-sided interval is stated. Figure 2 still has no bands. |
| M3 Low-FPR metric unused; novelty overstated | Addressed | **Resolved for the metric** (R@1e-4 in Tables 3 to 5). **Novelty only partly**: see F1 (Zhao and Hassan CDR, Larroche Table 5). |
| M4 "Worse than random" from one draw | Addressed | **Resolved.** Exact null, and I verified the values. |
| M5 Event universe and unit differ from prior work | Addressed / run | **Resolved for the universe** (it equals Larroche's exactly). **Units partly**: user-day and edge-hour were added but priced like events (M6). |
| M6 Published budgeted results not used | Addressed | **Partly.** Table 11 exists, but DECADES is misdescribed, statistics are not comparable, Larroche (2026) Table 5 is missing, and nothing is recomputed in the published splits (M1, M5). |
| M7 No source-side feature | Run | **Resolved.** M2c, R8 and the F3 relations; consistent with ADSAGE at user-day level. |
| M8 Straw-man split; M1 tuned on labels of the same operation | Run | **Resolved.** Harmonised protocol is primary; M1u without labels; Larroche's advice cited. |
| M9 Significance under one operation; campaign gap | Run | **Resolved as far as possible.** User-clustered bootstrap, 5 seeds, gap at 1/6/24/72 h. The campaign-level headline remains (M4). |
| M10 No deep or graph detector | Run | **Partly.** AE, VAE and an Euler reimplementation were added, but the SOTA (Argus, UltraLMD++), Euler's own code in the harmonised protocol, and any sequence model are missing; M4c was refused silently (M2). |
| M11 Budget range not grounded in staffing | Run | **Partly.** A staffing model exists on weak inputs and is not unit-specific (M6). |
| m3 Siadati and Memon | Addressed | Resolved. |
| m5 Argus AP unit | Addressed | Resolved (percent stated; verified). |
| m6 UltraLMD++ Rec@10 configuration | Addressed | Resolved (0.1788 is the deduplicated row; verified). |
| m7 Bowman split | Addressed | Resolved (verified). |
| m8 ATT&CK mappings | Addressed | Resolved (Appendix B qualifies them). |
| m11 M0b single-day recall | Addressed | Resolved. |
| m12 Time to detection presented as a result | Addressed (removed) | **Over-corrected.** It should return as the intrusion-level primary metric (M4). |
| m14, m15 campaign split across partitions; machine-account sources | Checked | Resolved. |

## The three changes that would most raise acceptance odds

1. **Put the state of the art through the harness on the identical event set and anchor to Larroche (2026).** Export per-event scores of Argus and the authors' Euler from LMDEval (which already runs them on the same 369,569,626 events and 702 positives), plus UltraLMD++ and one sequence model, and rank them in AuthBench. Report Larroche's Table 5 beside them. Then "no detector at 10/day" and "ROC-AUC does not order first detection" become claims about the field, not about a catalogue of PyOD baselines.
2. **Make the operational metric match the data.** Primary: intrusion-level time-to-detection at budget k, with its exact null, after fixing the partial first test day. Secondary: campaign recall. Recompute DR-1K, ADSAGE's CR-1000 and UltraLMD++'s Rec@10/hour on your own rankings in their exact splits (add the 0-indexed legacy split train 0 to 7 / test 8 to 12). Price units separately and add the source-host-day unit.
3. **Break the one-intrusion ceiling, or change the paper's genre.** Either add a second dataset with several intrusions (OpTC with Larroche's or Majorczyk et al.'s labels: three scenarios), or reposition explicitly as a reproducibility and methods paper: the harness, the exact null, and the Euler edge-level reconciliation extending Larroche Fig. 3, R+R and Boucek and Husák. Target DTRAP, a CNSM-style replication track, or ANUBIS.
