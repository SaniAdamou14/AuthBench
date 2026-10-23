# Review R4: external examiner / admissions-committee reading

Reviewer lens: Master's examiner and PhD admissions committee member (security / applied ML). Scope: scholarship, structure, argument, writing, figures and tables, and what the document shows about the author's research ability.
Sources read: `main.tex` (all 732 lines), the generated tables as rendered, `references_log.md` (spot checks), README.md, and all 29 rendered pages of `main.pdf` (PyMuPDF at 100 dpi, Tables 3 and 4 zoomed to 200 dpi). Previous review rounds were not opened.
Locations: "p." is the PDF page and "l." is the line in `main.tex`.

---

## 1. Summary of the work

The report presents AuthBench, an open harness for evaluating authentication anomaly detectors under daily alert budgets. It enforces the temporal split in code and checks bit-exact row-locality. It ranks events, edge-hours and user-days within each day, assigns each red-team campaign its detection budget, and compares the first-detection budget with an exact hypergeometric random-ranking null. On LANL, under Larroche's (2026) harmonised protocol (whose event set it reproduces exactly) and under a day-based protocol, thirteen detectors are compared. The results:

- No detector reaches a campaign within 10 events/day.
- Intrusion-level detections are fragile: they depend on a partial first day and on the fitting seed.
- ROC-AUC ordering is unrelated to first-detection ordering.
- In user-days, Euler and a source-novelty score reach about half the campaigns at 500/day.

A side study reruns the authors' Euler code. It matches the published AUC, not the published AP, and the test sets are reconciled edge by edge. The author frames everything as a case study of one operation.

---

## 2. Verdicts

### (a) As a Master's thesis / Master's research project

**Verdict: pass at Distinction level, low band (about 70–72 on a UK scale), with minor corrections.**

Why distinction:

- The evaluation design is original in its details:
  - the exact null for a minimum over days;
  - the per-campaign budget distribution;
  - intrusion-level "first day noticed" at fixed staffing;
  - fixed-threshold transfer.
- The statistical and engineering rigour is well beyond the usual Master's level:
  - a paired block bootstrap with jackknife BCa;
  - Holm corrections throughout;
  - a cluster bootstrap over users;
  - seed refits;
  - tie brackets;
  - CI that regenerates every number in the paper.
- The Euler reproduction, reconciled edge by edge against the authors' loader (§9, p. 16–17), is publishable work on its own.
- Critical self-assessment is exemplary: the "Design history" paragraph (l. 141) and the Scope section (l. 557) are rare even in PhD theses.

Why not a high distinction:

1. **No stated research question.** The argument is diffuse (Major 1).
2. **Headlines outrun the caveats.** The abstract and conclusion state several results more firmly than the body supports (Major 2).
3. **Effective sample size is one.** The substantive findings rest on one operation, and no second dataset was attempted although one is harmonised by the work the paper builds on (Major 4).
4. **Hard to read.** The prose is dense enough to obscure the contributions (§5 below).

As a *thesis document* it would need two more sections: a short background chapter and a Discussion. These count as minor corrections, because the content exists and is scattered.

### (b) PhD readiness / admissions-committee reading

**Reading: strong evidence of research readiness. For a BSc applicant this is in the top decile of research signals I would expect to see.**

The document shows that the author can do the following:

- identify a methodological gap in a live literature (Larroche 2026, Zhao & Hassan 2026);
- build infrastructure that rules out whole classes of error;
- run and interrogate other people's code;
- correct his own earlier public claims (the v1 correction notice in README.md).

These are the traits that predict PhD success.

As a *first PhD paper*, it is at strong-workshop level today: the venue class of the paper it builds on (ANUBIS@ESORICS), or AISec / LASER / DIMVA short paper. It is not yet RAID/ACSAC/top-4 level, mainly because of n = 1 operation and the lack of a single sharp claim.

Risks for the application:

- **Skim test.** A committee member skimming the abstract and Figure 1 for two minutes will struggle to say what the paper found. The abstract lists seven results and no question.
- **Interview probing.** Expect questions on the statistics. Single-author documents this dense prompt interviewers to check ownership. Be ready to explain without notes:
  - why the bootstrap is inconsistent for a minimum (l. 217);
  - where the BCa acceleration comes from;
  - why Holm and not Benjamini–Hochberg;
  - the hypergeometric null (Eq. 2).
- **Non-reproduction published without contacting the authors.** The paper publicly reports that a named NDSS paper's AP does not reproduce, and states that the authors have not been contacted (l. 569). A committee member who knows the Euler authors' community will notice. Contact them before the next version.

---

## 3. Strengths

1. **Clear operational framing.** The budget unit, the daily top-k and the staffing model (§6.1, l. 189–206) turn an abstract metric debate into a concrete question a SOC would ask.
2. **Exact null instead of a single random floor** (l. 196–200). This is a small, correct, reusable methodological contribution. Comparing against the analytic distribution rather than one realisation of M0a is exactly right.
3. **Leakage control is engineered, not asserted.** It has a static import test (l. 137), bit-exact row-locality (l. 139) and features equal to whole-window computation (l. 145). The explicit admission that the daily top-k is retrospective, with the fixed-threshold variant as the remedy (l. 139, §6.4, §7.6), shows mature thinking.
4. **Honest accounting of researcher degrees of freedom.** See l. 141 and l. 150 ("Those four temporal results are therefore informed by validation labels"), and Appendix G.
5. **The Euler reconciliation** (§9, l. 448–471) is meticulous: recounts with the authors' own loader, an off-by-one header-skip bug found, alignment differences quantified, and unintended deviations of one's own disclosed (l. 452).
6. **Variance sources beyond the test bootstrap** (§7.8, Table 10). Showing that PCA's "best" first detection is one of the two best seeds (l. 309) is the kind of result most papers hide.
7. **Citations are verified.** `references_log.md` documents primary-source checks. My spot checks below found the claims faithful.

---

## 4. Major issues (ranked)

### M1. No explicit research question; three papers in one

**Where:** Introduction, l. 64–77; abstract, l. 56; the section list in the ToC (p. 2).

**Why it matters:**
- The introduction motivates, then says "This paper does not claim the observation; it contributes a harness … and a case study" (l. 69), then lists four contributions. No question is posed, so the conclusion (l. 565) has nothing to answer and reads as a list.
- The paper is three works under one title:
  - (i) a benchmark harness (a tool contribution);
  - (ii) a LANL case study under two protocols;
  - (iii) a reproduction of Euler.
- Each competes for space, which is why the body has 15 top-level sections and 15 numbered tables.
- Tellingly, the README states a crisper question than the paper does: "What does published anomaly-detection performance become under a realistic alert budget?"

**Fix:**
- End the introduction with two or three numbered RQs, for example:
  - RQ1: under a leak-free protocol, at what daily budgets is the LANL operation detected, and how robust is that budget to seed, day boundaries and threshold transfer?
  - RQ2: does ordering detectors by ROC-AUC/AUC-PR predict their ordering by budgeted detection?
  - RQ3: do published Euler results reproduce from the released artefacts?
- Answer each by number in the Conclusion.
- Consider releasing §9 (Euler) as a separate short reproducibility report, for example in ReScience C style or a reproducibility track. It is strong enough to stand alone and would shorten this paper by about 2.5 pages.

### M2. The abstract and conclusion state results more firmly than the body supports

**Where:** abstract l. 56; Scope l. 557; Conclusion l. 565; against Table 6 (p. 13), l. 141, l. 309, Table 10 (p. 15).

Four concrete mismatches:

1. **"ROC-AUC tracks where most campaigns are found."**
   - In the *harmonised* protocol, which l. 543 calls "the label-free evidence", ROC-AUC vs median b(c) is ρ = 0.55, Holm p = 0.734 (Table 6, p. 13; App. C, p. 27).
   - Only the *temporal* protocol gives a significant coefficient (0.91, Holm 0.004). That protocol has four label-tuned detectors (l. 150), its test days were seen by an earlier code version (l. 141), and it is confounded in split and universe (l. 115).
   - The body sentence at l. 313 quotes the harmonised ρ without its p-value and quotes the temporal one with it. That is selective reporting, even if unintended.
2. **"two other detectors notice it a week later at 50."**
   - One of the two is PCA, whose b* = 20 comes from one of its two best seeds out of five.
   - Its seed median is 137 (l. 309, l. 399; Table 10). At the median seed, PCA does not detect within 50 at all.
   - The VAE (the other detector) and the autoencoder (the day-1 one) were fitted once (l. 219).
   - So all three intrusion-level headline results are single-fit or seed-favourable.
3. **User-day result for source rarity.** l. 141 says it is "informed by an earlier look at the same operation, not confirmatory evidence". The abstract and l. 565 present it without that qualifier. l. 565 also drops the random-ranking comparison that l. 557 keeps.
4. **"thirteen detectors"** (l. 56, l. 565). This count includes two floors and, in the harmonised protocol, M1 = M1u. Ten distinct non-trivial detectors is the honest number for the main protocol.

**Fix:**
- State abstract and conclusion claims at the robustness level the body establishes: seed-median results, harmonised-only for confirmatory claims, an explicit "exploratory" label for post-hoc ones.
- Rewrite the ROC-AUC claim along these lines: "In the label-free protocol, neither ROC-AUC nor AUC-PR orders detectors by first detection (ρ ≤ 0.10); their agreement with the median campaign budget (ρ = 0.55–0.79) is not significant after correction."
- Note that campaign-weighted AP *does* track the median budget significantly (ρ = 0.93, Holm 0.005; App. C, p. 27). That is the more interesting and better-supported finding, and it never reaches the body (see M6).

### M3. The temporal protocol's role is unclear and its comparisons are confounded

**Where:** l. 115 ("The two protocols differ in split and in universe at once"); l. 141; l. 150; Table 4; §7.4; App. E–G.

**Why it matters:** The temporal protocol is the *previous version's* protocol. Its test days have been inspected by two code versions. Four of its detectors read validation labels. It differs from the harmonised one in two factors at once, so no difference between protocols can be attributed to anything. Yet it occupies half of every results table and supplies the only significant correlation. A reader cannot tell whether it is a robustness check, a legacy, or a second experiment.

**Fix:** Pick one of the following.
- (a) Make the harmonised protocol the sole primary analysis, and move everything temporal to one appendix titled as continuity with v1.
- (b) Add the missing factorial cell (temporal split on the remote-LogOn universe) so the two factors can be separated.

Option (a) also cuts roughly three pages.

### M4. Effective n = 1, and the obvious second dataset is not used or discussed

**Where:** l. 92 (Larroche 2026 harmonises "LANL and OpTC"); l. 539; l. 565.

**Why it matters:**
- The main contribution is claimed to be a *harness* (l. 73). A harness demonstrated on one dataset with one labelled operation is a harness for LANL.
- The paper says repeatedly that a second dataset "is needed" (l. 539, l. 565). It never says why OpTC, harmonised by the very protocol it adopts, was not used: size, labels, compute, licence?
- An examiner will ask, and the silence reads as an unforced gap.

**Fix:**
- Either run OpTC under Larroche's harmonisation, or give one paragraph explaining why not and what it would take.
- If you do run it, *pre-register* the analysis plan (OSF or a dated commit) before looking at the results. The paper itself identifies a garden-of-forking-paths risk (l. 141). Pre-registration is the textbook remedy and a strong PhD-readiness signal.

### M5. One inferential inconsistency: "reproduces" from a non-significant test

**Where:** l. 450 ("It reproduces the published AUC closely (… t = −2.7, p = 0.056)"); abstract l. 56 ("reproduces its published AUC").

**Why it matters:** A failure to reject at n = 5 is not evidence of reproduction. It is the absence-of-evidence reasoning the paper rightly warns against at l. 313 ("absence of evidence, not evidence of independence"). The asymmetry (AP "not reproduced" by a test, AUC "reproduced" by a non-rejection of a test) will be noticed by a statistically literate reader.

**Fix:** Do one of the following.
- Report the difference with a CI (−0.0027, 95% CI ≈ [−0.0054, 0.0000]) and say "within 0.003 of the published value".
- Or run a TOST equivalence test with a margin stated in advance (for example ±0.005).

A related, smaller point: Table 5 (p. 12) reports per-k random-ranking probabilities over three values of k and several days without any multiplicity note.

### M6. Results presentation buries the strongest findings and over-reports the weakest

**Where:** §7.6–7.9 and §11 (p. 13–15, 18); Table 6 vs App. C.

**Why it matters:**
- §7.7 (Regimes) ends with "no effect … can be separated from fitting noise" (l. 378): a null result given a section and a table (Table 9).
- §11 (circularity) is a synthetic-data anecdote from before the LANL runs. It is instructive, but it is not part of the case study, and it sits between "Comparison" and "Threats".
- Meanwhile campaign-weighted AP is defined (l. 209) and is the only threshold-free metric whose ordering significantly matches the budget ordering in both protocols (App. C). It is absent from Table 6 and never discussed. The body says only "AUC-PR behaves similarly with higher coefficients" (l. 313).

**Fix:**
- Move §7.7, §11 and the "Undocumented choices" paragraph (l. 452) to appendices.
- Promote campaign-weighted AP into Table 6 and into one sentence of the abstract. "A campaign-weighted AP is the threshold-free summary that best predicts budgeted detection" is a constructive, citable takeaway.

### M7. Structure: no Discussion, fragmented ending

**Where:** ToC p. 2. Sections 8–15 are eight top-level sections after Results.

**Why it matters:** Interpretation is spread across "Diagnostics", "Comparison", "Scope of the claims", "Threats" and "Conclusion". Nowhere does the paper step back and say:
- what a SOC or a benchmark designer should do differently;
- why source-side signals win in user-days but not in events (this is touched on at l. 443 only);
- what the null results imply for the LANL dataset's continued use.

**Fix:** Use Results → Discussion (implications; comparison with prior work; scope) → Threats → Conclusion. Diagnostics can become a Results subsection.

### M8. The related work is a list, not a synthesis, with a few relevant omissions

**Where:** §2, l. 82–95 (p. 3–5).

**Why it matters:**
- "Detection on the LANL data" (l. 89) is a roll-call: X did A, Y did B, then a sentence naming five detectors with no characterisation.
- "Budgeted evaluation" (l. 92) is an annotated bibliography with numbers (DR-1K 0.081, 32.27 vs 0.09 ± 0.04, 42.1 pp). Those numbers belong in Table 14.
- Only "Positioning" (l. 95) synthesises.

Omissions an examiner would raise:

| Topic | Missing works | Why they matter |
|---|---|---|
| Daily-budget alert evaluation outside security ML | Dal Pozzolo et al. 2018 (IEEE TNNLS), "alert precision" with *k* alerts per day in fraud detection | The closest methodological precedent for "budget per day" |
| Early or first detection as a metric | Numenta Anomaly Benchmark (Lavin & Ahmad 2015); evaluation critiques in time-series AD (Kim et al. 2022, AAAI; Wu & Keogh, IEEE TKDE) | They address detection timing and benchmark inflation |
| SOC capacity | Sundaramurthy et al. 2014/2015; Kokulu et al. 2019 (CCS) | Peer-reviewed sources to replace or support the weak Vectra survey and the LLM-agent preprint used for the staffing model (l. 206) |
| Top-k metrics | Precision@k / recall@k from information retrieval | The budgeted metric is recall@k per day; saying so places it |

**Fix:** Organise §2 by the evaluation dimensions you address: split/leakage, metric and unit, null baseline, fit variance, reproduction. Add a small feature-matrix table showing which prior works cover which dimension. That table *is* the synthesis, and it makes the gap visible at a glance.

---

## 5. Writing and presentation

### Overall assessment

The English is correct, precise and largely free of jargon inflation. Hedging is careful, sometimes excessive. The main problem is **density**:

- Paragraphs such as l. 290, 309, 336, 358, 399 and 420 each carry 10–20 numbers.
- Several sentences exceed 60 words (l. 73, 92, 194, 450).
- Many paragraphs open with a number rather than a claim.

The reader must reconstruct the argument. Rule of thumb for the next version:
- one claim per paragraph, stated in its first sentence;
- at most three numbers in running text, with the rest in tables.

### Terminology consistency

| Term(s) | Problem | Location |
|---|---|---|
| *intrusion* / *operation* / *attack* | Used as synonyms for the whole red-team activity (31/20/13 occurrences), while "intrusion" is also used generically ("intrusion detection") | e.g. l. 56 "notices the attack", l. 108 "operation", l. 290 "the intrusion" |
| *campaign* | Defined as per-user groups with a 24 h gap (l. 104). Then l. 551 says campaign recall "counts each compromised account of the operation as a campaign", which is false: 162 campaigns over 98 accounts (Table 3 caption) | l. 104 vs l. 551 |
| *alert* | Means an event in "alerts per day" (l. 56, 286) but a user-day in "500 per day" (l. 56, l. 336). The abstract switches units mid-paragraph | l. 56, 286, 336 |
| *R1/R2* | Regimes (l. 133), and also rules R1–R8 (App. B), and *R* = 10,000 resamples (l. 213). "R1 (reported)" vs "R1 new (user, destination) pair" collide on the same pages (p. 14 vs p. 16) | l. 133, 213, App. B |
| *first detection* vs *first noticed* | Two different "first" notions: the minimum daily budget b* vs the first day at fixed k. The abstract uses "notices" for the second without defining it | l. 56, §7.2–7.3 |
| *non-trivial* vs *non-floor* | Used interchangeably | Fig. 1 caption vs Tables 3–4 |
| *AUC-PR* / *AP* / *average precision* | One quantity, three names, and three number formats: scientific (Tables 3, 4, 9, 10, G), decimal (Table 12), percent (Table 13) | throughout |
| Detector codes | M2b is PCA but sits in the rarity family (M2a/M2c). Table 2's note admits "M3c is `M3b_ecod`" in code. The rule names in Table 15 (`R4_offhours`, `R5_machine_account_anomaly`) differ from those in Table 11 ("R4 unusual hour", "R5 machine-account rarity") | Table 2, Table 11, Table 15 |

Fix: rename the codes once (for example R-, D-, B- prefixes), and use one display name everywhere.

### Representative rewrites

1. **l. 56 (abstract)**

   > "Per intrusion, the autoencoder notices the attack on its first day at 100 alerts per day, but only because that day is partial; two other detectors notice it a week later at 50."

   Announcing a result and then retracting it in the same sentence wastes the reader's attention. Rewrite:

   > "No detector reliably flags the operation on its first day: the autoencoder's day-1 detection at 100 events/day vanishes once the partial first day is excluded, and the earliest robust detection comes a week later (VAE, 50 events/day)."

2. **l. 133**

   > "With training data that contain red-team events (temporal training holds 50), a detector can be fitted with them, the situation of a defender who does not know the network is compromised (R1, reported), or without them (R2)."

   Rewrite:

   > "The temporal training partition contains 50 red-team events. In regime R1 (reported) detectors are fitted on them, as a defender unaware of the compromise would; in R2 they are removed."

3. **l. 399, final clause**

   > "… and the intervals of Tables 3 and 4 should be read as lower bounds on uncertainty."

   An interval is not a bound on uncertainty. Rewrite:

   > "The intervals in Tables 3 and 4 therefore understate the uncertainty."

4. **l. 290**

   > "… pair rarity on day 14 at 500; the five other non-trivial detectors never within 500."

   This is a verbless fragment. Rewrite:

   > "… the remaining five detectors place no red-team event in the top 500 on any day."

5. **l. 115**

   > "its partitions are those of the previous version of this report, which the event counts confirm"

   Rewrite:

   > "these are the partitions of version 1; their event and positive counts match version 1 exactly (Appendix G)."

6. **l. 450**

   > "It reproduces the published AUC closely (… p = 0.056)"

   Rewrite:

   > "Its mean AUC is 0.0027 below the published value, a gap five runs cannot distinguish from zero"

   See M5.

7. **l. 206**

   > "about a third of the alerts a typical SOC handles in a day across all its tools"

   This is correct only if "handles" means "actually deals with" (Vectra: 4,484 received and 67% not dealt with, so about 1,480 handled). Most readers will read it as "receives", and 500/4,484 = 11%. State the arithmetic.

### Other presentation points

- **l. 101 (data provenance).** The paper says the mirror was verified on 3 of 58 days. Yet reproducing Larroche's event set *exactly* (369,569,626 events, 702 positives; l. 115) is much stronger evidence of data integrity than the 3-day hash check. Connect the two.
- **Single-author "we".** This is acceptable, but use it consistently. "the author of the rules" (l. 515) is the author himself; say "I" or "we".
- **Whitespace.** About 2.5 near-empty pages (the abstract page p. 1, p. 21, p. 25), plus a ToC page. That is fine for a preprint. For applications, drop the ToC and let References follow directly.

---

## 6. Figures and tables

### Figures

**Figure 1 (p. 11).**
- The content is right: four panels, the null band, error bars.
- Problems:
  - points are labelled only with opaque codes (M2b, M3c), so the reader must keep Table 2 open;
  - labels overlap in the top-left cluster (M3a/M3b/M3c/M2a/M5c);
  - annotation text is about 6 pt in print;
  - in the harmonised panels the dark null band [251, 36,412] overlaps the light 1–500 band, which makes the shading hard to read;
  - the protocol name is folded into the y-axis label instead of a panel title.
- The caption describes the plot but does not state the takeaway. Add: "HBOS has the highest ROC-AUC yet first detects at 1,262 events/day; no monotone relation in either protocol."
- Fix: use names, not codes; protocols as panel titles; larger fonts.

**Figure 2 (p. 12).**
- It shows **M0a random (floor)**, a single random realisation, as the baseline curve. The paper's own principle at l. 200 is to compare with the *exact* null "rather than … the single realisation of the random floor M0a". Since the expected random campaign-recall curve is computable (l. 204), plot it (with its 95% band) instead of M0a.
- Other problems:
  - the legend font is tiny;
  - the dash-dot and dotted styles are hard to tell apart;
  - grey curves are unidentified;
  - the shaded band is "10 to 500" here but "1 to 500" in Figure 1. Pick one.
- The caption lacks a takeaway, for example "In both protocols the curves leave zero only above about 100 events/day; most campaigns need 10^3–10^5."

### Tables

**Tables 3 and 4 (p. 10)** are the most important tables and the least readable.
- They use `\scriptsize` with `tabcolsep 1.5pt` (l. 236, 256) and 10 columns.
- At normal zoom the headers "ROC-AUC R@10^-4" and "Null Holm" run together, and the three recall columns read as one string ("0.0% 0.0% 0.0% 16,718 0.816").
- Fix: move the BCa interval column to an appendix, or report AUC-PR scaled (×10^-5) with the interval below it in small type. That restores normal column spacing.

**Row order is inconsistent across tables:**
- Table 3: by detector ID.
- Table 5: by result.
- Tables 7 and 8: a third order (PCA, VAE, AE, Euler, …).
- Table 13: by AP.
- App. E/F: yet another.

Use one order everywhere (ID order, or order by harmonised b*) and say so.

**Float placement.** Several tables appear *before* the section that introduces them:
- Table 11 above the §8 heading (p. 16);
- Table 13 above §10 (p. 18);
- Table 15 (from §11) lands inside §12 "Threats" (p. 19);
- Table 2 splits a sentence of §6.1 across p. 7–8.

Use `[tbp]` with `\FloatBarrier` at section ends, or place floats after the first reference.

**Appendix tables (App. C–G, p. 27–29)** have no numbers and no captions. Their explanations come as trailing paragraphs. App. D's "Temporal protocol:" dangles at the foot of p. 27, with the table on p. 28. Number them and caption them.

**Table 6 (p. 13)** omits campaign-weighted AP, the metric with the strongest and only consistently significant agreement (App. C). See M6.

**Table 12 (p. 17).** Configuration names wrap over three lines. Abbreviate them and define the abbreviations in a note.

**Table 14 (p. 19)** is useful and honest ("not comparable across rows"). It is the right home for the numbers now packed into l. 92.

**Captions in general** describe columns well but almost never state the takeaway. Tables 7 and 8 come closest. For an admissions reader who looks only at figures and tables, captions are the paper. Add one finding sentence to each main-text caption.

### Citation spot checks

| Claim in text | Check | Result |
|---|---|---|
| Hopper, "fewer than nine alerts per day" over 15 months (l. 67) | USENIX abstract: "<9 alerts per day", 15-month dataset | Faithful |
| Vectra, 2.7 h of triage per analyst-day (l. 206) | Report wording "nearly 3 hours (2.7)" | Faithful (but see rewrite 7 on "a third") |
| Garchery & Granitzer, days 1–8 / 9–13, 50 training anomalies, CR-1000 0.254 / 0.255 (l. 92, Table 14) | Log | Faithful |
| King & Huang Table VI values (l. 448) | Log | Faithful |
| Zhao & Hassan, 42.1 pp CDR range (l. 92) | Log | Faithful |
| Tuor et al. 2018, "average percentile … not average precision" (l. 89) | Correct, and it fixes the v1 error | Faithful |

Source quality and rendering:
- **Rendering defect:** `\textcite{strom2018attack}` (l. 627) renders as "Strom et al. (2018, July/2020)" on p. 26, while the reference list says "(2020, March)". This is a biblatex-apa `origdate` artefact. Drop `origdate` or cite as 2020.
- **Weak sources for the staffing model** (l. 206): a vendor survey and a self-reported LLM-agent preprint. The paper flags this candidly; replace or support them with the SOC literature listed under M8.
- **Very recent foundations:** the protocol the paper adopts rests on a workshop paper accepted in September 2026 (Larroche 2026) and on a CCS 2026 preprint. That is unavoidable, but say once that these are recent and partly not yet in proceedings.

---

## 7. What would move it up one level

From "strong workshop paper / distinction-level MSc" to "solid main-track paper / first PhD paper":

1. **A second operation.** Run the harmonised OpTC set under the same harness, with the analysis plan pre-registered before the run. This single step converts the case study into a study, tests whether "ROC-AUC does not predict first detection" survives a second operation, and substantiates the harness claim.
2. **One question, one paper.** Use the three RQs from M1. Cut the body to about 12–14 pages:
   - harmonised protocol only in the main text;
   - temporal protocol, §7.7, §11 and the Euler deviations list in appendices;
   - Euler reproduction as a separate short reproducibility report, sent to the authors first.
3. **Robust headlines.** Fit every learned detector (including the AE, VAE and Euler) over at least 5 seeds. Report seed-median b* and the seed range as the primary numbers. Let the abstract claim only what holds at the median seed and in the label-free protocol.
4. **A constructive takeaway.** Promote campaign-weighted AP (or another summary that predicts budgeted detection) to a stated recommendation, and test it on the second dataset. Benchmark papers that end with "here is what to report instead" are cited; papers that end with "it depends" are not.
5. **One figure that tells the story.** Redo Figure 2 with the exact null curve and named detectors. Make Figure 1 readable without Table 2. Give every main-text caption a takeaway sentence.
6. **A Discussion section.** Cover implications for SOC deployment (fixed thresholds do not transfer; user-day ranking changes which detector wins), for benchmark design, and for continued use of LANL as a benchmark.
