# Reviewer 4: reproducibility, Euler, artifact evaluation

Paper: "Detection Under an Alert Budget: A Leak-Free Case Study of Authentication Anomaly Detectors on the LANL Data, with a Reproduction of Euler" (reports/paper/main.pdf, repository HEAD 165e999).
Expertise: reproducibility studies, temporal GNNs and link prediction for security, Euler (King and Huang, NDSS 2022), artifact evaluation.

## Summary (5 lines)

1. The paper reimplements Euler (GCN + GRU link predictor), runs the authors' code (iHeartGraph/Euler at b6ab8fd) five times on the same machine, and reconciles the two test sets edge by edge.
2. The engineering and record keeping are well above the norm: every Euler number traces to a committed file, and the 518 / 514 / 525 reconciliation is correct and verified against the authors' loader.
3. The verdict "the authors' code reproduces the published AUC but not the published AP" holds statistically for the released code (t = -12.2, p = 2.6e-4 over 5 runs), but the paper's wording "on the published data" is not established: the released preprocessing does not regenerate the published Table V (nodes, events), and the pinned environment postdates the paper by 21 months.
4. The list of differences between paper, authors' code and reimplementation leaves out at least four that matter: edge weights, node features, negative sampling and the time-based validation tail. Several cheap alternative explanations for the AP gap are also left out: a handful of top-ranked positives, float32 saturation of the authors' scores, different edge sets, and worker count.
5. As an artifact, the paper's numbers regenerate byte for byte from committed outputs (I checked this), but the committed `run_lanl.sh` would not regenerate Table 10 or the authors'-code results end to end.

## Scores (1 to 5)

| Criterion | Score | Comment |
|---|---:|---|
| Soundness | 3 | Counts and arithmetic are right. Causality holds. The AP verdict is over-generalised and its alternative explanations are not examined. |
| Presentation | 3 | Dense but precise. Table 10 mixes test sets with different prevalences and leaves out half of the published Table VI columns. |
| Contribution | 2 | For a top venue the Euler part is a careful non-reproduction note, not yet a diagnosis. The harness itself is the stronger contribution. |
| Reproducibility | 4 | Exceptional provenance and regeneration from outputs. The end-to-end Euler path is not scripted, and the data copy is verified only partially. |

## Recommendation

- **Top-tier security venue (USENIX Security, CCS, NDSS, S&P; or a reproducibility/SoK track): major revision (weak reject as it stands).** Nothing is fatal. But a paper that says a published NDSS result does not reproduce has to (a) pin down which artifact it failed to reproduce, (b) rule out the cheap alternative explanations, and (c) say which component explains the reimplementation's gap of 3 AUC points. All three are affordable: the authors' code ran 5 repetitions in 1 h 52 min, and one reimplementation seed on the NTLM subset takes about 2.5 min at 5 to 7 GB peak RSS.
- **arXiv / Zenodo preprint: ready after minor corrections.** Before posting: replace "on the published data" (Section 15) and "identical test set" (abstract), state the Table V mismatch and the date of `requirements.txt`, and complete the list of undocumented code choices (M1, M2 below). These are text edits only.

## Artifact badges I would award

| Badge | Award | Reason |
|---|---|---|
| Available | Yes, conditional | Public repository, CITATION.cff, earlier version archived. Condition: a DOI-archived snapshot of the exact commit used for this version (165e999 or a tag), not only the first version. |
| Functional | Yes | Locked environment (`uv.lock`, `--locked` in CI), lint, mypy, unit and integration tests, a demo under 5 minutes, and byte-for-byte regeneration checks in CI. I reran `tests/unit/test_euler.py`: 8/8 pass. |
| Results Reproduced | Not yet | I could not rerun LANL (data form, about 17 h, 61 GB RAM, about 9 EUR). More importantly, the committed scripts would not produce Table 10 or the authors'-code row without manual steps (M6). Regenerating the paper from committed outputs works. |

## Strengths

- **Traceability.** `scripts/paper_assets.py` reads every Euler number from `reports/runs/20261022T102638Z/` and cross-checks counts (for example, the test-edge equality at paper_assets.py line 680). I ran it in a clean clone and the result was byte-identical (see "What I checked").
- **Running the original code at all**, with a minimal `paths.diff`, a pip freeze, memory logs and OOM history. Few reproducibility papers in this area do this.
- **The edge-by-edge reconciliation is correct.** I checked it against the authors' code:
  - `split.py` drops the first line of `redteam.txt` with `red_events[1:]`, but the file has no header.
  - `load_lanl.py` aligns snapshots on `te_start = 150885` and drops the partial last snapshot.
  - `PredictorEncoder.decode_all` starts at `self.is_head`, so the first test snapshot is never scored.
  - TP/TPR in `results.txt` gives 398 / 0.7743 = 514 in every run.
- **The text is fair.** Section 13 explicitly does not claim the published AP is wrong, and the run notes (REVIEW_NOTES.md) record retracted single-seed conclusions instead of overwriting them.
- **M5c is causal and row-local by construction.** Tests cover causality (`test_scores_do_not_depend_on_later_snapshots`), determinism and row-locality.

## Fatal issues

None. No issue invalidates the benchmark's main claims or the arithmetic of the Euler section.

## Major issues

### M1. "On the published data" is not established: the released preprocessing does not regenerate the published Table V, and the environment postdates the paper

- **Where:** Section 15 ("does not reach its published average precision on the published data"); abstract ("on an identical test set"); Section 9 ("with its pinned dependencies"); Table 10, "Published" row.
- **Evidence (from the authors' run notes, left out of the paper):** REVIEW_NOTES.md, "The authors' code" section:
  - The released `split.py` keeps 49,341,299 NTLM lines, 44,748,938 without self-loops. Its node map has **15,610 nodes**.
  - Table V of King and Huang reports **17,685 nodes** and **45,871,390 events**, and `load_lanl.py` hard-codes `X_DIM = 17688`.
  - 17,685 is close to the total number of computers in LANL `auth.txt`. A plausible hypothesis, to label as a hypothesis, is that the published run built its node map before the NTLM filter, or used a different filter.
  - Only the anomalous-edge count (518) matches.
- **Environment:** `requirements.txt` was added at bdbe3f0 and b6ab8fd on 2 and 6 November 2023. The paper's experiments predate the camera-ready, which was added to the repository on 29 April 2022. So torch 1.13.1 / PyG 2.2.0 / scikit-learn 1.1.1 are not demonstrably the published environment. They were also installed with pip/uv rather than the listed conda builds (`euler_original_install.sh`).
- **Workers:** the paper's script `lanl_experiments/runall_LANL.sh`, line 4 (`python run.py -t 5 -d 0.5 -e GCN -i PRED`), uses the default `-w 8`. The rerun used `-w 4`, which matches the paper's text ("4 worker nodes unless otherwise specified"). Worker count changes the DDP gradient averaging and the test-time work split. The effect is probably small, but it is a difference from the published command.
- **Fix:**
  - Say plainly that what was not reproduced is *the released code with the released preprocessing in the November 2023 environment*.
  - Add a sentence and a column to Table 10 (nodes: 15,610 vs 17,685; events: 49.3 M / 44.7 M vs 45.9 M).
  - Replace "on the published data" with "on the data its released preprocessing produces".
  - In the abstract, "identical test set" is true of the events but not of the scored edge sets (see M3). Write "the same test events".

### M2. The list of undocumented choices in the authors' code is incomplete, and two omissions bear directly on the reimplementation gap

- **Where:** Section 9, paragraph "The authors' code"; RUN.md "Euler"; docs/methodology.md lines 313-315; `src/authbench/models/euler.py` docstring lines 3-13.
- **The paper lists four choices:** 10 negatives, early stopping on AUC + AP, dropout 0.25 / DropEdge 0.8, one step per epoch. Reading b6ab8fd, these are also not described by the paper, or contradict it, and are not reproduced:
  1. **Edge weights.** `load_lanl_dist(..., ew_fn=std_edge_w)` (load_lanl.py line 33) passes `std_edge_w` to every loader. `std_edge_w` (load_utils.py) computes `sigmoid(trunc(C / sigma))`, not the paper's Eq. 13, `sigmoid((C - mu) / sigma)`. The paper's version, `standardized`, exists but `run.py` never selects it.
     - For counts [1, 1, 2, 3, 5, 50], I computed weights [0.5, 0.5, 0.5, 0.5, 0.5, 0.88] (code) against [0.38, 0.38, 0.39, 0.41, 0.43, 0.88] (paper and AuthBench).
     - In the code, almost every edge gets weight 0.5. This changes the GCN normalisation for the bulk of edges, so it is a first candidate for the 3-point AUC gap.
  2. **Node features.** `make_data_obj` uses `x = torch.eye(cl_cnt + 1)`, identity only. Section VI-A of the paper concatenates a node-type one-hot, and AuthBench follows the paper (`self.types`, euler.py line 109).
  3. **Negative sampling.** `fast_negative_sampling` (tdata.py) rejects observed edges. AuthBench draws `torch.randint` pairs with no rejection (euler.py lines 226, 245).
  4. **Validation tail.** `run_all` (spinup.py) removes the last 5% of the pre-attack *time*, `max((tr_end - tr_start)//20, 2*delta)`, from training. It uses that tail only to advance `h0` and set the cutoff, and the test starts from that `h0`. AuthBench trains on all pre-attack snapshots. The paper does state this ("remove the final 5% of snapshots for tuning the classifier"), but AuthBench's stated deviations do not mention that it drops it.
  5. Smaller differences:
     - Loss on probabilities with `EPS = 1e-6` and a factor 0.5 (euler_interface.py `bce`), against `binary_cross_entropy_with_logits` in AuthBench.
     - The authors' code is unseeded (`np.random`, `torch.rand`, `torch.randperm` with no seed). The five repetitions are therefore nondeterministic; say so.
- **AuthBench-side items that should be listed as choices:**
  - methodology.md line 290 and euler.py line 9 attribute "against as many random non-edges" to the paper. Eq. 5 does not fix the number of negatives.
  - The early-stopping criterion (hold-out loss) is a choice.
  - The first GCN layer uses ReLU (from the code, not the paper).
  - The second layer applies the bias *before* propagation: `_propagate(graph, self.second(h))` at euler.py line 124. That gives `Â(HW) + (Â1)b`, a degree-scaled bias, unlike `GCNConv`, which adds `b` after aggregation.
  - `_propagate` adds a unit self-loop even where a self-loop edge already exists. This matters for the `all` graph and the benchmark graph (15 M of the 99 M test edges are self-loops, `machine/euler_edge_sets.out`). PyG's `add_remaining_self_loops` would not do this.
- **Fix:** a three-column table (paper / authors' code b6ab8fd / AuthBench), one row per component: graph, filter, node map, features, edge weight, GCN, RNN, projection, dropout, negatives, loss, optimiser step, early stopping, validation tail, test start state, scoring, evaluation.

### M3. The AP gap has cheap alternative explanations that are not examined, and the verdict is not quantified

- **Where:** Section 9, "The authors' code"; abstract; Section 15.
- **Statistics:**
  - Over the five runs in `results.txt`: AP mean 0.00582, sd 0.00178; one-sample t against 0.0155 gives t = -12.2, df 4, p = 2.6e-4.
  - AUC mean 0.98795, sd 0.00222; t = -2.67, p = 0.056. So "reproduces the AUC closely" is right in magnitude, but it is also borderline below.
  - Report these, and note that Table VI gives no spread for the published mean of 5 unseeded runs.
- **A few positives decide AP at this prevalence.** Prevalence is 3.6e-5, so AP is roughly the mean over positives of the precision at each positive's rank.
  - One positive ranked first adds about 1/514 = 0.0019 to AP.
  - The four positives the released pipeline loses (the skipped first line, plus 3 anomalous edges in the unscored first test snapshot) could contribute up to 4/518 = 0.0077. That is most of the 0.0097 gap, if the published run scored them and ranked them near the top.
  - The skipped edge is the attack's first lateral move from C17693, a never-seen edge of exactly the kind a link predictor ranks at the top.
  - Whether the code behind the paper scored the first test snapshot cannot be known from the repository, but the sensitivity can be measured.
  - **Fix:** for the reimplementation, report AP on 525 positives and on the authors' 514 (dropping the same four); report the ranks of the 10 best-ranked positives and each one's precision contribution.
- **Score precision.**
  - The authors compute AP on `1 - sigmoid(z_u . z_v)` in float32 (`spinup.score_stats`). This saturates: every logit below about -17 maps to exactly 1.0, and the resolution near 1 is 6e-8. I checked: `1 - sigmoid(-17) = 0.99999994`, `1 - sigmoid(-20) = 1.0`.
  - AuthBench ranks float64 negated logits (euler.py line 330; euler_fidelity.py `score_edges`).
  - Ties at the top are exactly where AP is decided, so "half the AP" between AuthBench and the authors' code is partly a difference of measurement.
  - **Fix:** evaluate both with one evaluation function. Either dump the authors' per-edge logits in a local run (an output-only patch that is not redistributed), or recompute AuthBench's AP on float32 `1 - sigmoid`.
- **Different edge sets.**
  - AuthBench's AUC/AP is over 14,400,984 edges in multiples-of-1800 s alignment.
  - The authors' is over 14,414,775 edges in test-start alignment, minus the first snapshot.
  - **Fix:** add a snapshot-origin parameter to `snapshot_edges` and compare the two implementations on the authors' exact edge set.
- **The other Table VI columns.**
  - `results.txt` also gives TPR 77.4% (sd 10.8) against the published 85.49%, FPR 0.651% (SE 0.0098%) against 0.609%, and precision 398 / (398 + 93,760) = 0.0042 against 0.0050.
  - The threshold-based columns also fall short.
  - **Fix:** reporting them is fairer to the authors and more informative than AUC/AP alone.

### M4. The reimplementation gap is diagnosable now and is left untested; the vocabulary result is a red flag

- **Where:** Section 9 ("Which of them closes the gap is not tested here"); RUN.md ("needs the whole dataset, so another rented machine"); Table 10, row "authors' node vocabulary".
- **Cost is low:**
  - Reimplementation runs on the NTLM subset take about 2.5 min per seed at 5.1 to 7.0 GB peak RSS (`euler_fidelity_configs.log`). The ablations fit on a 16 GB machine.
  - A 5-repetition run of the authors' code took 1 h 52 min, at about 36 GB peak including page cache (REVIEW_NOTES.md).
- **The vocabulary result.** Giving every node its own identifier drops AuthBench to AUC 0.826 +- 0.034 and AP 1.1e-4. The authors' code uses that same vocabulary and reaches 0.988. So some AuthBench-specific component interacts badly with untrained node rows. Candidates are the degree-scaled bias, non-rejected negatives, the type embedding, and truncated BPTT. Until this is explained, the reader cannot tell whether M5c is "Euler" or "a GCN+GRU predictor inspired by Euler".
- **Fix:**
  - One-at-a-time ablations of AuthBench toward the code: `std_edge_w`, identity features, rejection negatives, `nratio = 10`, DropEdge 0.8 with dropout 0.25, a full-sequence step, AUC+AP early stopping, the 5% time tail. Five seeds each.
  - Two authors'-code variants: `ew_fn = standardized` (paper Eq. 13), and `-w 8`.
  - If the gap closes, M5c becomes a faithful Euler. If not, rename M5c "GCN-GRU link predictor (Euler-style)" throughout the benchmark tables.

### M5. How Euler is used in the benchmark: causal, but not equally comparable to the other detectors

- **Where:** Section 5, paragraph on line 179 of main.tex; Table 3 (catalogue); Sections 7.4 and 7.7; Section 12 "Fitting variance".
- **Causal: yes.**
  - `prepare` runs the trained model forward over `[train, val, test]` (train_eval.py line 338).
  - `score` reads the embeddings of snapshot s-1 (euler.py line 319).
  - The vocabulary comes from training rows only.
  - I found no look-ahead. Warm-up days feed features only (build_features.py line 190), so the timeline frames are contiguous and do not overlap.
  - The information set differs from the other detectors, though: an event is judged on history up to the end of the previous half hour, not "every earlier event". State this in Section 4.2.
- **Shared unseen node.** Every computer first seen after training collapses into one node: `unseen = len(self._nodes)`, euler.py line 174. That node is a hub in message passing and a shared embedding in the decoder. In the harmonised protocol (41 hours of training) this may cover a large share of test computers, possibly including red-team targets. Euler's harmonised result (best AUC-PR, 51.2% of campaigns at 500 user-days) may partly reflect this artefact.
  - **Fix:** report the share of test events and of malicious test events that touch the unseen node, per protocol. Ablate it (per-node rows over the full node set as in the authors' code, or type-only features for unseen nodes).
- **Input graph.** The input graph is not Euler's: remote LogOn only (harmonised) or all events with self-loops (temporal), against NTLM-only for Euler. The fidelity numbers therefore do not certify M5c as used. Say so next to Table 3.
- **Fitted once, untuned.** M5c is fitted once with one seed (`SEEDED` in pipeline/tuning.py excludes it), while PCA, Isolation Forest and others are refitted with five seeds and tuned with 20 candidates in the temporal protocol. The fidelity runs show seed sd up to 0.034 AUC and a 2x spread in AP. The paper acknowledges this (Section 12), but it decides Euler's place in the "first detection" and "user-day" conclusions.
  - **Fix:** add M5c to the five-seed fit-variance table. The harmonised Euler fit is a single model on 41 hours of training; give its cost per seed.
- **Ties.** M5c gives every event of an (edge, snapshot) the same score, so malicious events can tie with benign events on the same edge and half hour. RUN.md reports zero-width tie brackets for Euler, which is plausible (red-team edges are rare edges), but the paper should state it as a measured fact.

### M6. The artifact does not regenerate the Euler results end to end

- **Where:** Section 14; `scripts/run_lanl.sh`; `reports/runs/20261022T102638Z/machine/fidelity_rerun.sh`; `euler_original/*.sh`.
- **Problems:**
  - `run_lanl.sh` calls `scripts/euler_fidelity.py` with no `--seeds` (default `42`). Run as committed, it gives single-seed results, not Table 10's five.
  - `run_lanl.sh` also starts the fidelity check in parallel with both evaluations, which is what OOM-killed it at 50.8 GB in this run. The script exited 1 and `run.json` was written by hand.
  - Table 10 comes from `machine/fidelity_rerun.sh` at commit **6c12582** (`machine/authbench_commit.txt`). Section 14 names only `bd833eb`.
  - The authors'-code pipeline (`euler_original_install.sh`, `_split.sh`, `_run.sh`, `_after_eval.sh`) hard-codes `/root/...` paths and is not orchestrated.
  - `euler_original_split.sh` writes `zcat` output to `auth.fifo` with no `mkfifo` in the committed scripts. If that file was a regular file, `split.py` could race the writer. The recount shows the split was complete this time, but the script is not safe to rerun.
  - The evaluations ran while `euler.py` changed on disk. The "imported before 90ab443" argument (RUN.md) is convincing, but a third party cannot rebuild that state from one commit.
  - Data: the LANL server refused connections, so an Imperial College mirror was used, and its compressed SHA-256 differs from the official one. Content equality was checked on the total row count and on days 0, 5 and 13 only (`raw_sha256.txt`, RUN.md).
- **Fix:**
  - (a) A single `scripts/run_euler_reproduction.sh` that runs the fidelity check sequentially (all configs, `--seeds 42,...,46`) and the authors' code: clone at b6ab8fd, apply `paths.diff`, install the pinned environment, `mkfifo` or decompress, split, run, then the recount scripts. Use parameterised paths and cgroup memory caps.
  - (b) Sequential steps in `run_lanl.sh`, or explicit `systemd-run -p MemoryMax=` caps.
  - (c) A per-result commit table in Section 14: evaluations at bd833eb, fidelity at 6c12582, recounts ad hoc, authors' code at b6ab8fd.
  - (d) Order-independent content hashes for all 58 days and for `redteam.txt`, committed, so that anyone can verify their copy.
  - (e) In the paper, the full cost: about 17.1 h and 8.9 EUR billed, 61 GB RAM, and about 36 GB for the authors' code (two OOM failures at 16 GB and 30 GB caps).

## Minor issues

1. **Table 10 does not show prevalence.** "All events" AP 0.00035 against NTLM AP 0.0027 mostly reflects prevalence: 525 / 99.4 M = 5.3e-6 against 525 / 14.4 M = 3.6e-5, so the AP lifts are 66 and 74. Add prevalence and AP/prevalence columns.
2. **Table 10, "Authors' code" row, Events "NTLM, as above".** Their split keeps 49,341,299 lines, and self-loops are dropped at load time. Give that number.
3. **Section 9, "with its pinned dependencies".** Write "with the dependencies listed in requirements.txt (added November 2023), installed with pip/uv rather than conda".
4. **euler.py docstring (line 7) and methodology.md (line 289) say "two GCN layers of 32 units, tanh".** The code uses ReLU then tanh. State the ReLU as taken from the authors' code.
5. **No contact with the original authors is mentioned.** Before publishing a non-reproduction of an NDSS result, it is standard practice (and an ACM reproducibility-paper expectation) to contact King and Huang, ask for the published environment, node map and score files, and report their answer. Doing so would also resolve M1.
6. **Section 9 says DropEdge is "0.8" without explaining.** `DropEdge(0.8)` *drops* 80% of edges in training (`mask > p` keeps 20%), in addition to feature dropout of 0.25 in the GCN and before the GRU. Say "keeps 20% of edges" to avoid misreading.
7. **CI diffs `reports/paper/generated/*.tex` only.** Figures are not checked, although I found them byte-identical. Hash them in CI too.
8. **Stale comment in `dvc.yaml` (lines 86-87):** it says `reports/paper/` does not exist. Also, the DVC pipeline does not cover the fidelity check or the authors'-code run; say so in the README.
9. **Unbiased vs biased std.** AuthBench uses `std(unbiased=False)` and zero weights (sigmoid 0.5) when `sigma = 0`. The authors use the unbiased std (NaN for single-edge snapshots in `std_edge_w`). This is negligible but belongs in the M2 table for completeness.
10. **Missing p-values in the abstract.** The abstract says the AUC is reproduced "closely" and the AP "not". Add the two p-values from M3 (or "within 0.003, p = 0.06" and "p < 0.001").

## What I checked or reran, and the result

| Check | Result |
|---|---|
| Cloned iHeartGraph/Euler (b6ab8fd) into the scratchpad. Read `run.py`, `spinup.py`, `models/*`, `loaders/*`, `utils.py`, `runall_LANL.sh`, and the git history with dates. | `requirements.txt` dates from Nov 2023. The LANL code was last changed 10 Oct 2022 (path hotfix). Published command uses `-w 8`. Edge weights come from `std_edge_w`, features are identity only, negatives are rejection-sampled, the 5% time tail is held out, and the code is unseeded (M1, M2). |
| Read the paper (`EulerNDSS-Paper.pdf` in that repository): Sections IV-C, V-A, VI-A/B, Tables V and VI. | GCN+GRU predictor: AUC 0.9906, AP 0.0155, TPR 85.49, FPR 0.6088, P 0.0050. Eq. 13 edge weights, type one-hot, 4 workers, patience 10, 5 runs. Number of negatives and learning rate for LANL not stated. |
| Compared `src/authbench/models/euler.py` line by line with the paper and the code. | Faithful to the paper where the paper is explicit. Deviations as in M2 (degree-scaled bias, no rejection negatives, extra self-loops, shared unseen node). |
| Recomputed `results.txt` statistics, plus one-sample t-tests against the published values. | AP 0.00582 +- 0.00178, t = -12.2, p = 2.6e-4. AUC 0.98795 +- 0.00222, t = -2.67, p = 0.056. TP/TPR = 514 in each run. |
| Checked the reconciliation logic (`authors_test_edges.awk`, `authors_test_load.py`, `redteam_matches.py`) and outputs against the authors' loader semantics. | Correct: 14,416,149 - 1,374 = 14,414,775 edges; 517 - 3 = 514; 524 + 1 = 525; 715 red-team events of which 14 are unmatched. |
| Numerically checked `std_edge_w` against Eq. 13, and float32 `1 - sigmoid` saturation. | As in M2.1 and M3. |
| Checked causality and timeline construction of M5c in `train_eval.py` and `build_features.py`. | Causal; frames contiguous; shared unseen node (M5). |
| Fresh `git clone` of the repository (HEAD 165e999) into `scratchpad/review2/repo_copy_r4`, then `scripts/paper_assets.py` with the project venv and `PYTHONPATH` pointing to the copy. | Exit 0. `git status` clean: every file in `reports/paper/generated/` (numbers.tex, all tables, both figures in PDF and PNG) regenerated **byte-identical** from committed outputs. |
| `pytest tests/unit/test_euler.py` in the same copy. | 8 passed. |
| Not done | No LANL rerun (no data access or compute here). I did not execute the authors' code. |

The original repository was not modified: all clones and runs were in the scratchpad.

## The three changes that would most raise acceptance odds

1. **Run the ablations that turn the non-reproduction into a diagnosis (M4, M2).** Run them one at a time with five seeds on the NTLM subset: `std_edge_w`, identity features, rejection negatives, 10 negatives, DropEdge/dropout, a full-sequence step, AUC+AP stopping, the time tail. Add two authors'-code variants (Eq. 13 weights, `-w 8`). Either M5c becomes a faithful Euler, or the paper can say exactly which undocumented choice carries the published performance. That is a contribution a top venue values.
2. **Evaluate both implementations with one evaluation function on one edge set, and quantify AP's sensitivity (M3).** Use the same alignment, the same positives (with and without the four lost ones), and the same score precision. Report per-positive precision contributions and the remaining Table VI columns. If the AP gap survives, the claim becomes robust. If it does not, the paper has found why.
3. **Rewrite the claim around what was actually reproduced, and make that path one command (M1, M6).** State the Table V mismatch and the environment date. Replace "on the published data". Contact the original authors and report their answer. Commit a single script that regenerates Table 10 and the authors'-code row from raw data, with memory caps and per-result commits.
