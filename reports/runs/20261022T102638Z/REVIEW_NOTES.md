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

## Euler fidelity (S6-09)

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

## Still to do

- harmonised: finish, then the same review.
- run.json (duration and cost) and the archive, by hand.
- Authors' Euler code: five repetitions alone on the machine, then compare.
- Fix `ingest/to_parquet.py` stale parts (latent duplication on reconversion).
- Delete the server once everything is copied back.
