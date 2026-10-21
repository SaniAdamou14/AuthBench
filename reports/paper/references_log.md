# Verification log for references_verified.bib

Date of checks: 2026-10-21. Companion file: `references_verified.bib` (same folder).

## Access notes (read first)

- **DBLP could not be used.** `dblp.org`, `dblp.dagstuhl.de` and the `.bib` record URLs all return an "Making sure you're not a bot!" (Anubis) challenge page to curl and to WebFetch. `dblp.uni-trier.de` returned nothing. I did not try to get around the challenge, and the Chrome extension was not connected. Instead, USENIX, RAID, NDSS, AAAI, JMLR and the workshop papers were checked against the venue's own record (USENIX BibTeX embedded in the paper page, NDSS/doi.org, AAAI paper page and PDF, JMLR BibTeX). The preprint checks used Crossref, OpenAlex and HAL. OpenReview's mirror of DBLP confirmed only the venue string for tuor2018.
- **Semantic Scholar** returned HTTP 429 on every attempt, so I did not use it.
- **ACM DL, JSTOR and usenix.org/biblio/export** sit behind Cloudflare or JSTOR challenges. For the ACM full text (Axelsson) and the JSTOR record (Holm), I used Internet Archive snapshots.
- **Unconfirmed fields** go in the `annotation` field of each entry, marked "UNVERIFIED". biblatex-apa does not print that field.
- **Page numbers for [FACT] quotes** are the printed proceedings or journal page numbers, read from the running footers of the PDF.

---

## Per-key log

### axelsson2000 [FACT]
- Record: https://api.crossref.org/works/10.1145/357830.357849. It gives TISSEC 3(3), 186-205, Aug 2000, ACM.
- Full text: the ACM PDF via Wayback, http://web.archive.org/web/20231203032953id_/https://dl.acm.org/doi/pdf/10.1145/357830.357849. Its footer reads "ACM Transactions on Information and System Security, Vol. 3, No. 3, August 2000, Pages 186-205", which matches Crossref.
- Discrepancy: none. There is an earlier, different version: CCS 1999, "The base-rate fallacy and its implications for the difficulty of intrusion detection", doi 10.1145/319709.319710, pp. 1-7. Do not mix it up with the TISSEC article.
- FACT (abstract, p. 186): "the false alarm rate is the limiting factor for the performance of an intrusion detection system."

### sommer2010
- Record: Crossref 10.1109/sp.2010.25. It gives "2010 IEEE Symposium on Security and Privacy", pp. 305-316, IEEE. No discrepancy.

### pendlebury2019
- Record: the USENIX BibTeX embedded in https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury (entry 235493). It gives all 5 authors with full given names, the booktitle "28th USENIX Security Symposium (USENIX Security 19)", pp. 729-746, ISBN 978-1-939133-06-9, Santa Clara, CA, Aug 2019, USENIX Association.
- No DOI exists (USENIX).
- Full title: "TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time".
- The page also links an errata slip (sec19_errata_6.pdf).

### arp2022 [FACT]
- Record: USENIX BibTeX in https://www.usenix.org/conference/usenixsecurity22/presentation/arp (entry 277204). It gives 8 authors, "31st USENIX Security Symposium (USENIX Security 22)", pp. 3971-3988, ISBN 978-1-939133-31-1, Boston, MA.
- Full text: https://www.usenix.org/system/files/sec22-arp.pdf. PDF page 2 = printed p. 3971.
- FACT: names of the pitfalls (Section 2; the boxed definitions use these exact headings):
  - "P3 – Data Snooping." (p. 3973, Section 2.2 "System Design and Learning"). The box next to it reads "57% present".
  - "P7 – Inappropriate Performance Measures." (p. 3975, Section 2.3 "Performance Evaluation"). Box: "33% present".
  - "P8 – Base Rate Fallacy." (p. 3975, Section 2.3). Box: "10% present".
  - In Figure 1 (p. 3972) the short labels are "Data snooping", "Inappropriate measures" and "Base rate fallacy".
  - The headings use an en dash in the PDF. Write "P3: Data Snooping" if you need to avoid dash characters.
- FACT: prevalence of data snooping (Section 3 "Prevalence Analysis", p. 3977). Quote: "data snooping (P3), which are at least partly present in 90 % and 73 % of the papers, respectively". In context: sampling bias (P1) 90%, data snooping (P3) 73%. The set is 30 papers from ACM CCS, IEEE S&P, USENIX Security and NDSS (p. 3976).
- **Caution:** the per-pitfall box shows "57% present" for P3, while the text says 73% are "at least partly present". The two numbers measure different things. Quote the one that matches your claim.

### alahmadi2022 [FACT]
- Record: USENIX BibTeX in https://www.usenix.org/conference/usenixsecurity22/presentation/alahmadi (entry 277250). It gives "Bushra A. Alahmadi and Louise Axon and Ivan Martinovic", pp. 2783-2800, same proceedings as arp2022.
- Full text: https://www.usenix.org/system/files/sec22-alahmadi.pdf. PDF page 2 = printed p. 2783.
- FACT (Section 6 "RQ2: Definition of FPs", p. 2790). An interviewee (B3) quoted: "We know 99% of the alarms we generate are false positives, but we still have to look at them." The paper adds (p. 2791) that this 99% is "mostly benign triggers and not necessarily a measurement of the performance of the technology itself."
- FACT, alert volume (survey results, p. 2788): "45% of analysts reported that they receive less than 5K alerts daily." The survey had n = 20.

### kent2015dataset
- `curl -sI https://doi.org/10.17021/1179829` returns 302 to http://www.osti.gov/servlets/purl/1179829/.
- DataCite: https://api.datacite.org/dois/10.17021/1179829. Title "Comprehensive, Multi-Source Cyber-Security Events Data Set". Creator "Kent, Alexander" (no initial). Publisher "Los Alamos National Lab. (LANL), Los Alamos, NM (United States)". Year 2015. resourceTypeGeneral Dataset. Product number LA-UR--15-23810.
- OSTI: https://www.osti.gov/api/v1/records/1179829. Author "Kent, Alexander D.". publication_date 2015-05-21. report_number LA-UR-15-23810. product_type Dataset.
- Discrepancies:
  - The title has "Data Set" at the end, which the request left out.
  - The middle initial appears in OSTI only.

### kent2016chapter
- Chapter record: Crossref 10.1142/9781786340757_0002. pp. 37-65, World Scientific (Europe), series "Security Science and Technology". Online 2016-03-27, print 2016-05. ISBNs 9781786340740 (print) and 9781786340757 (electronic).
- Book record: Crossref 10.1142/q0022 (type edited-book). Adams, Niall and Heard, Nick are deposited with role "author" (they are the editors). It also carries an issued date of 2015-12-03.
- UNVERIFIED: series volume number.

### kent2015authgraphs
- Record: Crossref 10.1016/j.cose.2014.09.001. Kent, Alexander D.; Liebrock, Lorie M.; Neil, Joshua C. Computers & Security 48, 150-166, Feb 2015. No discrepancy.

### tuor2018
- AAAI paper page: https://aaai.org/papers/285-ws0489-aaaiw-18-17039/. Listed under AAAI Workshop Papers 2018, "Artificial Intelligence for Cyber Security" (https://aaai.org/proceeding/ws18-05/).
- PDF: https://cdn.aaai.org/ocs/ws/ws0489/17039-75960-1-PB.pdf.
  - Header: "The Workshops of the Thirty-Second AAAI Conference on Artificial Intelligence".
  - Printed authors: Aaron Tuor, Ryan Baerwolf, Nicolas Knowles, Brian Hutchinson, Nicole Nichols, Robert Jasper.
  - Page folios run 285 to 292 (8 pages).
- arXiv 1712.00557 (v1 2017-12-02, cs.NE). Comment: "To appear in proceedings of AAAI-2018 Artificial Intelligence in Cyber Security Workshop".
- OpenReview's DBLP mirror (note HJNH4yZOWH): venue "AAAI Workshops 2018". It gives the names "Aaron Randall Tuor" and "Robert Jasper".
- Discrepancies:
  - OpenAlex gives pp. 285-293, but the PDF folios end at 292.
  - The request called the workshop "Artificial Intelligence for Cyber Security (AICS)". The AAAI site uses "Artificial Intelligence for Cyber Security" and arXiv uses "Artificial Intelligence in Cyber Security".
- UNVERIFIED: publisher (AAAI Press is likely, but no record states it).
- Version cited: the peer-reviewed workshop version, not the preprint.

### brown2018
- Record: Crossref 10.1145/3217871.3217872. "Proceedings of the First Workshop on Machine Learning for Computing Systems", ACM, pp. 1-8, 2018-06-12. Event: HPDC '18.
- The request labelled it "MLCS 2018". Crossref has no "MLCS" acronym, but the full title matches.

### turcotte2016
- Record: Crossref 10.1109/isi.2016.7745472. Authors: Turcotte, Melissa; Moore, Juston; Heard, Nick; McPhall, Aaron. Venue: "2016 IEEE Conference on Intelligence and Security Informatics (ISI)", pp. 208-210, Sep 2016. No discrepancy.

### siadati2017
- Record: Crossref 10.1145/3133956.3134003. CCS '17 proceedings, pp. 1273-1284. No discrepancy.

### bowman2020
- Record: USENIX BibTeX in https://www.usenix.org/conference/raid2020/presentation/bowman (entry 259729). Authors "Benjamin Bowman and Craig Laprade and Yuede Ji and H. Howie Huang". Booktitle "23rd International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2020)", pp. 257-268, ISBN 978-1-939133-18-2, San Sebastian, Oct 2020, USENIX Association.
- No DOI.
- Discrepancy: OpenAlex gives "Craig Andrew Laprade". The USENIX form is used.

### king2022euler
- Record: Crossref 10.14722/ndss.2022.24107 and the doi.org BibTeX. Title "Euler: Detecting Network Lateral Movement via Scalable Temporal Graph Link Prediction". "Proceedings 2022 Network and Distributed System Security Symposium", Internet Society. No pages.
- The DOI resolves to https://www.ndss-symposium.org/wp-content/uploads/2022-107A-paper.pdf.
- **Discrepancy:** the PDF title is "EULER: Detecting Network Lateral Movement via Scalable Temporal Link Prediction", without "Graph". That is also the title of the TOPS version and the one in the request. The entry uses the DOI metadata title. Switch to the PDF title if you prefer it, but use one form consistently.

### king2023euler
- Record: Crossref 10.1145/3588771. ACM TOPS 26(3), pp. 1-36. Online 2023-06-27, issue 2023-08-30.
- UNVERIFIED: ACM article number (not in the record).

### xu2024argus
- Record: Crossref 10.1109/sp54263.2024.00012. "2024 IEEE Symposium on Security and Privacy (SP)", pp. 3590-3608. No discrepancy.

### paudel2022pikachu
- Record: Crossref 10.1109/noms54207.2022.9789921. "NOMS 2022-2022 IEEE/IFIP Network Operations and Management Symposium", pp. 1-7. No discrepancy.

### garchery2020adsage
- Record: arXiv API, id 2007.06985. Full title "ADSAGE: Anomaly Detection in Sequences of Attributed Graph Edges applied to insider threat detection at fine-grained level". v1 2020-07-14, cs.LG.
- Peer-reviewed version: none found. I searched Crossref (author query) and OpenAlex; DBLP was unavailable. OpenAlex does list a 2020 dissertation by the first author ("User-centered intrusion detection using heterogeneous data").

### larroche2021decades
- arXiv 2103.15708: v1 2021-03-29, v2 2022-06-28, cs.CR. Authors "Corentin Larroche; Johan Mazel; Stephan Clémençon".
- **Peer-reviewed version found.** HAL API record hal-04590157 (https://telecom-paris.hal.science/hal-04590157v1):
  - Type COMM, peerReviewing=1, proceedings=1.
  - Conference "CAID 2020 : Conference on Artificial Intelligence for Defense", starting 2020-11-19, Rennes, France, pp. 1-11.
  - Proceedings collection: "Actes de la conférence CAID 2020 / Proceedings of the Conference on Artificial Intelligence for Defence 2020" (hal-03206297, 2021-04).
- The entry cites the CAID version and keeps the arXiv eprint.
- UNVERIFIED: publisher.
- Discrepancy: given name "Stéphan" (HAL) vs "Stephan" (arXiv).

### larroche2025ultralmd
- Record: arXiv API, 2504.13527. v1 2025-04-18, cs.CR, single author.
- Peer-reviewed version: none found in Crossref (author and title queries), OpenAlex or HAL.

### larroche2026fair
- Record: arXiv API, 2607.29390. v1 2026-07-31, cs.CR. Comment: "Accepted at ANUBIS '26".
- Workshop: https://superviz.inria.fr/anubis26/ (title "ANUBIS @ESORICS 2026").
  - Full name: "The 2nd International Workshop on Assessment with New methodologies, Unified Benchmarks, and environments, of Intrusion detection and response Systems (ANUBIS)".
  - Host: "co-located with the 31st European Symposium on Research in Computer Security (ESORICS 2026)".
  - Held "at the Main Campus of Sapienza University of Rome ... on September 18".
  - The programme lists "On fair and realistic performance evaluations for graph-based lateral movement detectors, Corentin Larroche" at 15h55.
  - The page says submissions follow the LNCS template, but it does not name a proceedings publisher.
- No proceedings DOI found in Crossref as of 2026-10-21.

### zhao2026gids
- Record: arXiv API, 2609.12263. v1 2026-09-10, cs.CR. Authors "Rui Zhao; Wajih Ul Hassan". Comment: "Full version of the paper accepted to the ACM Conference on Computer and Communications Security (CCS) 2026".
- DOI: none. Crossref title search and a check of the "2026 ACM SIGSAC CCS" container both returned nothing.
- Acceptance has not been checked against the official CCS accepted-papers list.
- Related work found along the way: Wang, Zheng, Gui, Hua, Hassan, "R+R: From Claims to Crashes: A Systematic Re-evaluation of Graph-Based Network Intrusion Detection Systems", ACSAC 2025, doi 10.1109/acsac67867.2025.00050. This is from the Crossref search result only, not fully verified.

### gokstorp2024
- Record: Crossref 10.1109/noms59830.2024.10575561. Authors: Gökstorp, Simon; Nyberg, Jakob; Kim, Yeongwoo; Johnson, Pontus; Dán, György. pp. 1-9.
- Discrepancy: the Crossref container is "NOMS 2024-2024 IEEE Network Operations and Management Symposium", with no "IFIP".

### hassan2019nodoze
- Record: Crossref 10.14722/ndss.2019.23349 and doi.org BibTeX. NDSS page https://www.ndss-symposium.org/ndss-paper/nodoze-combatting-threat-alert-fatigue-with-automated-provenance-triage/ gives the same 7 authors in the same order. No pages. No discrepancy.

### apruzzese2023sok
- Record: Crossref 10.1109/eurosp57164.2023.00042. "2023 IEEE 8th European Symposium on Security and Privacy (EuroS&P)", pp. 592-614, Jul 2023. No discrepancy.

### liu2008iforest
- Record: Crossref 10.1109/icdm.2008.17. Liu, Fei Tony; Ting, Kai Ming; Zhou, Zhi-Hua. pp. 413-422. No discrepancy.

### goldstein2012hbos (mostly UNVERIFIED)
- Confirmed: title, authors and year 2012, from OpenAlex W2336788611. OpenAlex has no venue and gives "Andreas R. Dengel".
- Primary record unavailable. The DFKI publication page (https://www.dfki.de/web/forschung/projekte-publikationen/publikation/6431) returned a maintenance page (HTTP 200 with "Wartungsarbeiten") or 403, and the PDF returned 403. The Wayback availability API either rate-limited (429) or had no snapshot.
- Not citable as a record: a web-search snippet of the DFKI page reads "In: Stefan Wölfl (Hrsg.). KI-2012: Poster and Demo Track. German Conference on Artificial Intelligence (KI-2012), 35th, September 24-27, Saarbrücken, Germany, Pages 59-63, Online, 9/2012."
- No DOI in Crossref.
- Action: re-check the DFKI page once its maintenance ends (it was announced as ending at 10:00 CET).

### li2022ecod
- Record: Crossref 10.1109/tkde.2022.3159580. IEEE TKDE 35(12), 12181-12193, issue dated 2023-12-01.
- Note: an SSRN posting also exists (10.2139/ssrn.4313179).

### zhao2019pyod
- Record: JMLR page https://www.jmlr.org/papers/v20/19-011.html and BibTeX https://www.jmlr.org/papers/v20/19-011.bib. Gives 20(96):1-7, 2019.
- The arXiv preprint is 1901.01588. No DOI.

### lakhina2004
- Record: Crossref 10.1145/1015467.1015492 (proceedings). pp. 219-230. Event "SIGCOMM04: ACM SIGCOMM 2004 Conference".
- Crossref "is-identical-to" 10.1145/1030194.1015492, which is ACM SIGCOMM CCR 34(4), 219-230. Either can be cited. The proceedings version is used.

### shyu2003pca (venue UNVERIFIED)
- OpenAlex W187043655: title, 4 authors, 2003, no venue. Locations: DTIC ADA465712, CiteSeerX, and an author file named ICDM03_WS.pdf.
- DTIC PDF via Wayback (http://web.archive.org/web/20250812202052id_/https://apps.dtic.mil/sti/pdfs/ADA465712.pdf):
  - Title and authors match.
  - The Report Documentation Page gives Report Date 2003 and Performing Organization "Naval Research Laboratory, Center for High Assurance Computer Systems".
  - No venue in the supplementary notes.
- **The original venue cannot be verified from any machine-readable record.** The entry cites the DTIC copy.
- A related, later chapter with a DOI exists: 10.1007/11539827_18, Springer, pp. 311-329.

### chandola2009
- Record: Crossref 10.1145/1541880.1541882. ACM CSUR 41(3), 1-58, Jul 2009 (online 2009-07-30).
- Crossref splits title and subtitle ("Anomaly detection" / "A survey").
- UNVERIFIED: article number.

### davis2006
- Record: Crossref 10.1145/1143844.1143874. Container "Proceedings of the 23rd international conference on Machine learning - ICML '06", ACM Press, pp. 233-240. No discrepancy.

### saito2015 [FACT]
- Record: Crossref 10.1371/journal.pone.0118432. PLOS ONE 10(3): e0118432, 2015-03-04.
- Full text: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0118432&type=printable.
- FACT (Conclusion, PDF p. 19 of 21): "The results of our study strongly recommend PRC plots as the most informative visual analysis tool."
- Supporting sentence from the same paragraph: "unlike ROC plots, PRC plots express the susceptibility of classifiers to imbalanced datasets with clear visual cues".

### holm1979
- Crossref has no record. `doi.org/10.2307/4615733` returns 404, even though OpenAlex lists it and JSTOR uses it internally.
- JSTOR page (live access blocked by a "Client Challenge"), via Wayback snapshot http://web.archive.org/web/20151105081233id_/http://www.jstor.org/stable/4615733:
  - Author "Sture Holm".
  - "Scandinavian Journal of Statistics, Vol. 6, No. 2 (1979), pp. 65-70".
  - "Published by: Wiley on behalf of Board of the Foundation of the Scandinavian Journal of Statistics".
  - Stable URL http://www.jstor.org/stable/4615733.
- OpenAlex agrees on vol. 6, pp. 65-70 (it has no issue number).

### efron1993
- Crossref 10.1007/978-1-4899-4541-9: book, Efron, Bradley; Tibshirani, Robert J. Issued 1993. Publisher "Springer US". ISBNs 9780412042317 and 9781489945419.
- Crossref 10.1201/9780429246593: book. Publisher "Chapman and Hall/CRC". Dated 1994-05-15. Gives "Tibshirani, R.J.".
- Discrepancy: neither record says "Chapman & Hall, 1993" outright. The entry uses 1993 with the Springer-registered DOI, and marks publisher "Chapman & Hall" UNVERIFIED. The alternative is in the annotation.

### field2007
- Record: Crossref 10.1111/j.1467-9868.2007.00593.x. JRSS B 69(3), 369-390. Online 2007-05-22, issue 2007-06-01.
- The record gives initials only ("Field, C. A.", "Welsh, A. H.").
- The record's publisher is OUP (current publisher).

### kaufman2012leakage
- Record: Crossref 10.1145/2382577.2382579. ACM TKDD 6(4), 1-21, Dec 2012. Four authors including Stitelman, Ori.
- The KDD 2011 conference version (10.1145/2020408.2020496) has only three authors. Do not mix them up.
- UNVERIFIED: article number.

### strom2018attack
- Full text: https://attack.mitre.org/docs/ATTACK_Design_and_Philosophy_March_2020.pdf (46 pp.). The cover page gives:
  - Report number "MP180360R1", "MITRE PRODUCT", Project No. 10AOH08A-JC.
  - Title "MITRE ATT&CK®: Design and Philosophy".
  - Authors Blake E. Strom, Andy Applebaum, Doug P. Miller, Kathryn C. Nickels, Adam G. Pennington, Cody B. Thomas.
  - "McLean, VA", "Originally Published July 2018", "Revised March 2020", public release case "19-01075-28".
- The mitre.org publication page returned 403.
- I could not retrieve the original July 2018 version (presumably MP180360), so the R1 revision is cited with origdate 2018-07.

### kingma2014vae
- arXiv API 1312.6114: "Diederik P Kingma; Max Welling", stat.ML. v1 2013-12-20; latest v11 2022-12-10.
- ICLR 2014 conference-papers list (https://iclr.cc/archive/2014/conference-proceedings/) includes "Auto-Encoding Variational Bayes, Diederik P. Kingma; Max Welling", linked to arXiv.
- UNVERIFIED: edition, location and dates of ICLR 2014.

### hochreiter1997lstm
- Record: Crossref 10.1162/neco.1997.9.8.1735. Neural Computation 9(8), 1735-1780, 1997-11-01, MIT Press. No discrepancy.

### kipf2017gcn
- OpenReview search API: note SJU4ayYgl, authors "Thomas N. Kipf, Max Welling", venue "ICLR 2017 Poster".
- arXiv 1609.02907 v4: comment "Published as a conference paper at ICLR 2017".
- https://iclr.cc/archive/www/2017.html: "5th International Conference on Learning Representations", Toulon.

### grover2016node2vec
- Record: Crossref 10.1145/2939672.2939754. KDD '16, pp. 855-864. Crossref splits title and subtitle. No discrepancy.

---

## Optional search: evaluation at a fixed alert budget or analyst capacity

Result: **no entry added.** I did not find one that I could verify fully, with a primary record and a [FACT] quote, in the time available.
- Semantic Scholar was rate-limited throughout.
- DBLP was blocked.
- Searches were done in OpenAlex and Crossref.

Candidates found, all left unverified:
1. **Veeramachaneni, Arnaldo, Korrapati, Bassias, Li, "AI^2: Training a Big Data Machine to Defend"**, IEEE BigDataSecurity/HPSC/IDS 2016, pp. 49-54, doi 10.1109/bigdatasecurity-hpsc-ids.2016.79 (metadata confirmed in Crossref). It is closed access, so I extracted no quote about the analyst budget.
2. **Tuor, Kaplan, Hutchinson, Nichols, Robinson, "Deep Learning for Unsupervised Insider Threat Detection in Structured Cybersecurity Data Streams"**, arXiv 1710.00811. Comment: "Proceedings of AI for Cyber Security Workshop at AAAI 2017". The AAAI record was not retrieved, and I extracted no budget quote.
3. **Axelson and Nweke, "Alert-Budget-Aware Assessment Workflow for Evaluating Unsupervised IDS in IEC 61850 Smart-Grid Communication"**. It is on the ANUBIS 2026 programme (https://superviz.inria.fr/anubis26/), but no proceedings or preprint record was found.
4. (Excluded) OpenAlex also returned the user's own Zenodo preprint, 10.5281/zenodo.23159767. It is not peer-reviewed and was not used.


## Additions on 2026-10-21 (main session)

- **goldstein2012hbos**: venue, editor and pages now verified against the DFKI publication
  record 6431 (Internet Archive snapshot of 2026-01-16): "In: Stefan Wölfl (Hrsg.). KI-2012:
  Poster and Demo Track. German Conference on Artificial Intelligence (KI-2012), 35th,
  September 24-27, Saarbrücken, Germany, Pages 59-63, Online, 9/2012." The PDF itself
  answers HTTP 403.
- **shyu2003pca**: removed. No record names its venue; the report cites Lakhina et al.
  (2004) alone for PCA-based anomaly detection.
- **authbenchsoftware**, **authbenchpreprint**: from the DataCite records of
  10.5281/zenodo.23183808 (software, v1.0.0, issued 2026-10-06) and 10.5281/zenodo.23159768
  (preprint, v1, issued 2026-10-05).
- The `annotation` fields were moved out of `references.bib` because biblatex-apa prints
  them; the notes above and below this section are the complete record.

## Additions after review round 1 (2026-10-21)

- **mcclish1989**: Crossref record 10.1177/0272989x8900900307 (author Donna Katzman McClish; Medical Decision Making 9(3), 190-195, August 1989).
- **ho2021hopper**: USENIX presentation page (title, seven authors in order, pp. 3093-3110, ISBN 978-1-939133-24-3); abstract: "a 94.5% detection rate across over 300 realistic attack scenarios" and "an average of <9 alerts per day".
- **tuor2017**: arXiv API record 1710.00811 (five authors, published 2017-10-02; comment "Proceedings of AI for Cyber Security Workshop at AAAI 2017"). Workshop proceedings record not retrieved, so cited as the arXiv version with the comment as addendum.
- **amin2019cadence**: Crossref record 10.1145/3338501.3357368 (title "CADENCE", subtitle "Conditional Anomaly Detection for Events Using Noise-Contrastive Estimation", AISec '19, pp. 71-82, 2019-11-11).
