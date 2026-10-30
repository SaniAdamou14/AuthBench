# Review notes, LANL run 20261022T102638Z (draft)

Working notes written while the run was still going, to become RUN.md. Every number below
was read from a file of this run; the file is named next to it. Dated entries are appended,
never rewritten.

## Machine and provenance

- Hetzner CCX43, Nuremberg, Ubuntu 24.04, 16 vCPU, 61 GB RAM; booted about 08:42 UTC on
  22 October 2026, billed at 0.52 EUR/h as displayed in the console.
- Raw data from the Imperial College mirror (`lanl.ma.ic.ac.uk/data/cyber1/`), csr.lanl.gov
  refusing connections. The compressed files differ from the official SHA-256, the content
  does not: 1,051,430,459 rows, and days 0, 5 and 13 have the same row counts and
  order-independent content hashes as a conversion of the official files (`raw_sha256.txt`,
  `convert.log`).
- `run_lanl.sh` will exit 1 at the end: its first Euler fidelity check was OOM-killed (see
  below), so `run.json` and the archive are written by hand.

## Euler fidelity

Attempts, in order (`run.log`, `euler_fidelity*.log`):

1. In `run_lanl.sh`: OOM-killed at 50.8 GB (one `group_by` over all 1.05 billion events).
2. After 90ab443 (per-day snapshot edges): OOM-killed at its 24 GB cgroup cap (test edges
   still one `unique`, and 108 million edges scored at once). A first relaunch had not run at
   all (no HOME in the unit) although `run.log` logged exit 0; corrected in `run.log` below it.
3. After c4892b0 (test edges and scoring a day at a time): AUC 0.92955, AP 0.0000989
   (`euler_fidelity.json`, all events, first test snapshot dropped).
4. After ef40bdf (NTLM variant): OOM-killed at 24 GB building the NTLM test edges.
5. After c05b615 (NTLM filtered a day at a time): one seed per variant
   (`euler_fidelity_ntlm_c05b615.json`, `euler_fidelity_all_c05b615.json`).
6. After 6c12582: four configurations over seeds 42 to 46 (`euler_fidelity_<config>.json`).

| Configuration | Events | Test edges (anomalous) | AUC mean +- sd | AP mean +- sd |
|---|---|---|---|---|
| all | 1,051,430,459 | 99,430,341 (525) | 0.9521 +- 0.0137 | 0.00035 +- 0.00024 |
| ntlm | 44,748,939 | 14,400,984 (525) | 0.9570 +- 0.0028 | 0.0027 +- 0.0011 |
| ntlm_vocabulary | 44,748,939 | 14,400,984 (525) | 0.8261 +- 0.0341 | 0.00011 +- 0.00003 |
| ntlm_code | 44,748,939 | 14,400,984 (525) | 0.9590 +- 0.0145 | 0.0013 +- 0.0006 |
| published (table VI) | 45,871,390 | (518) | 0.9906 | 0.0155 |

Seed 42 of `ntlm` reproduces the single-seed run of step 5 to the last digit. A correction to
what was said during the run from one seed: on five seeds the NTLM filter multiplies AP by
about 8 (not 20) and moves AUC by 0.005, within seed noise (not 2.6 points).

### The authors' code (iHeartGraph/Euler, commit b6ab8fd, no licence)

Run on the same machine, outside this repository, edited only where its README asks for paths
(`/root/euler_original_data/paths.diff`). Facts established on its prepared files:

- `loaders/split.py` keeps lines containing "NTLM": 49,341,299 events, 44,748,938 without
  self-loops. AuthBench's `ntlm` variant keeps 44,748,939: the one line apart is the first
  line of auth.txt, which `split.py` skips as a header although the file has none.
- It skips the first line of redteam.txt the same way, so the first red-team event
  (t=150885, C17693 to C1003) is labelled benign in its files (`150885,9660,44,0`).
- 15,610 nodes in its node map, against 17,685 in the paper's table V and `X_DIM = 17688`
  in `load_lanl.py`; 45,871,390 events in table V matches neither count.
- Not in the paper, read in the code: 10 negatives per edge in training (`nratio`),
  early stopping on validation AUC + AP, dropout 0.25 and DropEdge(0.8), one optimiser step
  per epoch over the whole training sequence, learning rate 0.005 and patience 5 by default.
- Running it beside the evaluations failed twice for memory (16 GB cap, then a global OOM
  with a 30 GB cap during test loading, which killed only its process, `oom_score_adj` 1000).
  It is queued to start alone once both evaluations finish (`euler-original-run3`).

## temporal protocol (finished 18:58 UTC, 28,567 s)

- No model skipped (`tables/skipped_models.json`).
- R1, test days 11 to 13, 52 campaigns over 48 accounts (`tables/budget_summary.json`,
  `metrics_summary.json`): at 10 alerts a day no model detects a campaign. M1 detects 1, 2, 3
  of 52 at 50, 100, 500 a day, first detection at rank 42 (random ranking as good: 0.001).
  PCA reaches 9.6 % at 500. Euler: ROC-AUC 0.985 and third AUC-PR, but first detection at
  15,945, not better than random (0.203).
- M1u ROC-AUC 0.2441 is not a bug (`diagnostics/m1u_rules_temporal.txt`, script beside it):
  on all 282 malicious test events and a 1-in-25 benign sample, R5 (0.19), R3 (0.23), R7
  (0.32) and R4 (0.38) rank the red team below benign traffic; 61.7 % of benign events and
  0 % of malicious ones come from machine accounts. R1 (0.72) carries the signal.
- `first_detection` `ci_low` equals the point value for all 13 models in both regimes: a
  property of the percentile bootstrap of a minimum, now limitation 13 in
  `docs/limitations.md`.

## Added 23 October 2026, after the run

### Code provenance of the evaluations

Both evaluations started at commit bd833eb (`machine.txt`; its two "dirty" paths are the demo
figure rewritten by the smoke run and the untracked `reports/runs/`, `machine/smoke`). The
Euler fidelity commits 90ab443 to 6c12582 were pulled on the machine while they ran, and the
only source file they change is `src/authbench/models/euler.py`. Each evaluation imported that
module when it built its first catalog (temporal about 11:02 UTC, harmonised about 11:08 UTC,
`evaluate_*.log`), before 90ab443 existed (committed 11:39:45 UTC); Python does not reload an
imported module and the pipeline starts no subprocess. Every number in `temporal/` and
`harmonised/` is therefore from bd833eb.

### harmonised protocol (finished 21:57 UTC, 38,951 s)

- R1 only, by design (`conf/split/harmonised.yaml`: no validation partition). No model skipped.
- 702 malicious test events in 162 campaigns over 98 accounts (`tables/positives_ledger.csv`,
  `tables/budget_summary.json`).
- At 10 alerts a day no model detects a campaign. First detection: PCA 20 alerts a day
  (random ranking as good: 0.003), VAE 27 (0.003), autoencoder 59 (0.007), Euler 203 (0.022),
  pair rarity 405 (0.040); these are per-model values, not adjusted for 13 comparisons. At 500
  a day the VAE reaches 4.3 % of campaigns (7 of 162) and PCA 2.5 %.
- Euler has the best AUC-PR (0.00072, ROC-AUC 0.9615). M1 and M1u are identical, as expected
  without validation labels, at ROC-AUC 0.2435.
- M1u: the same reading as on temporal (`diagnostics/m1u_rules_harmonised.txt`, 702 malicious
  and a 1-in-25 benign sample): R3 0.15, R5 0.17, R4 0.38 rank the red team below benign
  traffic, R1 0.70 carries the signal, machine accounts are 66.8 % of benign events and 0 % of
  malicious ones.
- `first_detection` `ci_low` equals the point for all 13 models here too; limitation 13 now
  says so.

### The authors' code, five repetitions (`euler_original/`)

Run alone after the evaluations, 21:57 to 23:49 UTC, exit 0 (`results.txt`,
`run_gcn_gru_pred.log`). Highest sample of its cgroup's `memory.current` (page cache
included, every 15 s): 36,715 MiB, about 35.9 GiB (`memory.log`).

| | AUC | AP |
|---|---|---|
| Authors' code on this machine, mean +- sd over 5 | 0.9879 +- 0.0022 | 0.0058 +- 0.0018 |
| Published (table VI) | 0.9906 | 0.0155 |
| AuthBench `ntlm`, 5 seeds | 0.9570 +- 0.0028 | 0.0027 +- 0.0011 |

`results.txt` gives the standard error; the sd above is it times the square root of 5. The
authors' own code does not reach its published AP (0.0040 to 0.0080 over the five runs). Our
implementation is a further 3 points of AUC and about half the AP below their code.

### Test set reconciliation (`euler_original/authors_test_*`, `redteam_matches.*`)

- Red-team events: 715 distinct rows, 701 with an authentication event at the same second and
  computers (all of them NTLM), 14 without any, which no pipeline can label.
- In AuthBench's snapshot alignment (multiples of 1,800 s), the authors' prepared files give
  14,400,984 test edges and 524 anomalous, 525 with the first red-team line they skip: exactly
  AuthBench's `ntlm` test set. Both pipelines read the same data.
- Their loader aligns snapshots on the test start (t=150885): 14,416,149 edges, of which 1,374
  in the last, partial snapshot that it drops, leaving 14,414,775; 517 anomalous, 518 with the
  skipped first line, which is the paper's table V figure.
- Their predictor never scores the first test snapshot (`decode_all` starts at 1 on the head
  worker), which holds 3 anomalous edges: 517 - 3 = 514 evaluated, what TP/TPR gives in every
  repetition.

### Machine

Kernel OOM kills (`machine/kernel_oom.log`): five, matching the attempts above (50.8 GB global,
two at the 24 GB cap, the authors' code at 16 GB, then the global OOM at 17:42 that killed the
authors' process, not an evaluation). Correction, 23 October 2026: only the authors' code ran with `oom_score_adj` 1000 (shown in
its kill line); the fidelity reruns were at 0 and confined by their cgroup caps. The server
was deleted on 23 October 2026 after
everything was copied back and pushed; `run.json` was written by hand. The LANL form e-mail is
redacted from `machine/console_logs/download.log`.

## Still to do

- Done 23 October 2026: `RUN.md`; the stale-parts fix in `ingest/to_parquet.py` (refuses an
  output directory holding partitions, `--overwrite` replaces them). The fix does not concern
  this run: its conversion rewrote, with the same file and block size (547 blocks), every part
  of an identical earlier conversion (`machine/console_logs/convert.log`), and the fidelity
  check read exactly 1,051,430,459 events from the result.
- The archive: the run folder is in git, a tar.gz would only duplicate it.
- Euler gap to the authors' code: their undocumented choices (10 negatives per edge, early
  stopping on AUC + AP, dropout, DropEdge) are the leads; no machine is needed to read them, one
  is needed to test them.
