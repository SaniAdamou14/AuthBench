# LANL run, all 58 days, two protocols (22 October 2026)

Run `20261022T102638Z`: the first AuthBench run on the whole LANL dataset, with the code of
22 October 2026 (equal tuning budgets, both training regimes, the autoencoders and Euler). It
evaluates two protocols: `temporal`, the split of the wider-window run of 15 September 2026
([`../../lanl/wider_window_results/RUN.md`](../../lanl/wider_window_results/RUN.md)), and
`harmonised`, the protocol of Larroche (2026). It also checks the Euler reimplementation
against the paper and runs the authors' own Euler code on the same machine.

Which run the report is built on is decided with the paper update, not here. Every number
below is read from a file of this directory, named next to it. The working notes written during
the run, with every attempt and correction in order, are in [`REVIEW_NOTES.md`](REVIEW_NOTES.md).

## Provenance

| | |
|---|---|
| Code | commit `bd833eb` for both evaluations (see "Code during the run" below) |
| Machine | Hetzner CCX43, 16 vCPU, 61 GB RAM, Ubuntu 24.04 (`machine.txt`) |
| Duration and cost | 11.52 h from start to the last protocol, 5.99 EUR at 0.52 EUR/h; about 17.1 h and 8.9 EUR for the machine in all, invoice authoritative (`run.json`) |
| Source | Imperial College mirror of `cyber1` (csr.lanl.gov refused connections on 22 October 2026) |
| Integrity | the compressed files differ from the official SHA-256, the content does not: 1,051,430,459 rows, and days 0, 5 and 13 have the official files' row counts and order-independent content hashes (`raw_sha256.txt`, `run.log`) |
| Conversion | 1,051,430,459 of 1,051,430,459 rows, 0 dropped, 547 blocks, 703 s, peak RSS 2.17 GB (`convert.log`) |
| Features | `temporal` 1,455 s, `harmonised` 1,805 s (`run.log`) |
| Evaluation | `temporal` 28,567 s (both regimes), `harmonised` 38,951 s (`run.log`) |
| Bootstrap | 10,000 resamples over accounts, Holm correction over 78 pairs (`evaluate_*.log`) |
| Tuning | 20 candidates per tunable model on validation AUC-PR (`temporal/tables/tuning.json`); none on `harmonised`, which has no validation partition |
| Skipped models | none (`*/tables/skipped_models.json`) |

`run_lanl.sh` exited 1 at the end because its first Euler fidelity step had been OOM-killed;
it therefore wrote neither `run.json` nor the archive. `run.json` was written by hand with the
script's definitions; the archive is not needed, the directory is in git.

### Code during the run

Both evaluations started at `bd833eb`. While they ran, the Euler fidelity fixes `90ab443` to
`6c12582` were pulled on the machine; the only source file they change is
`src/authbench/models/euler.py`. Each evaluation imported that module when it built its first
catalog (`temporal` about 11:02 UTC, `harmonised` about 11:08 UTC, `evaluate_*.log`), before
`90ab443` was committed (11:39:45 UTC). Python does not reload an imported module and the
pipeline starts no subprocess, so every number in `temporal/` and `harmonised/` comes from
`bd833eb`. The two "dirty" paths in `machine.txt` are the demo figure rewritten by the smoke run
and the untracked `reports/runs/`.

## temporal protocol

Train days 0 to 7, validation 8 to 10, test 11 to 13, two warm-up days per partition, all
events. Test: 282 red-team events, 52 campaigns over 48 accounts
(`temporal/tables/budget_summary.json`). Budgets are alerts per day; "1st detection" is the
daily budget at which the first campaign is caught, "p random" the probability that a random
ranking does at least as well.

### R1, unsupervised (reported regime)

| Model | AUC-PR [95% CI] | ROC-AUC | Camp. @10 | @50 | @100 | @500 | 1st detection | p random |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M4a autoencoder | 2.17e-3 [1.27e-3, 3.65e-3] | 0.988 | 0.0% | 0.0% | 0.0% | 1.9% | 494 | 0.007 |
| M2b PCA | 1.64e-3 [6.14e-4, 6.23e-3] | 0.955 | 0.0% | 0.0% | 0.0% | 9.6% | 246 | 0.003 |
| M5c Euler | 8.12e-4 [4.81e-4, 1.71e-3] | 0.985 | 0.0% | 0.0% | 0.0% | 0.0% | 15,945 | 0.203 |
| M4b VAE | 7.80e-4 [3.49e-4, 2.65e-3] | 0.972 | 0.0% | 0.0% | 0.0% | 0.0% | 5,845 | 0.077 |
| M2a pair rarity | 6.30e-4 [2.63e-4, 2.12e-3] | 0.940 | 0.0% | 0.0% | 0.0% | 0.0% | 7,436 | 0.097 |
| M3b ECOD | 2.52e-4 [8.27e-5, 1.26e-3] | 0.966 | 0.0% | 0.0% | 0.0% | 0.0% | 5,123 | 0.066 |
| M1 rules (tuned weights) | 2.13e-4 [1.84e-5, 1.35e-3] | 0.496 | 0.0% | 1.9% | 3.8% | 5.8% | 42 | 0.001 |
| M2c source rarity | 9.09e-5 [3.29e-5, 4.67e-4] | 0.925 | 0.0% | 0.0% | 0.0% | 0.0% | 5,010 | 0.065 |
| M3b HBOS | 6.16e-5 [3.44e-5, 1.33e-4] | 0.930 | 0.0% | 0.0% | 0.0% | 0.0% | 8,572 | 0.111 |
| M3a Isolation Forest | 5.46e-5 [2.46e-5, 2.09e-4] | 0.937 | 0.0% | 0.0% | 0.0% | 0.0% | 98,938 | 0.757 |
| M0a random *(floor)* | 5.17e-6 [3.02e-6, 1.07e-5] | 0.500 | 0.0% | 0.0% | 0.0% | 0.0% | 48,891 | 0.492 |
| M0b always-fail *(floor)* | 5.16e-6 [3.04e-6, 1.17e-5] | 0.496 | 0.0% | 0.0% | 0.0% | 0.0% | 5,505,419 | 1.000 |
| M1u rules (uniform) | 3.08e-6 [1.84e-6, 6.91e-6] | 0.244 | 0.0% | 0.0% | 0.0% | 0.0% | 529,646 | 0.999 |

60 of the 78 pairwise AUC-PR differences are significant after Holm (`evaluate_temporal.log`,
`temporal/tables/pairwise_comparisons.json`).

### R2, semi-supervised (red-team events removed from the fitted rows)

The 50 training positives are removed from what the models fit on; everything else is as in
R1 (`temporal/r2_semi_supervised/`, `temporal/tables/r2_semi_supervised/`). The models whose
AUC-PR or first detection moves by more than 1 %:

| Model | AUC-PR [95% CI] | ROC-AUC | Camp. @500 | 1st detection | p random |
|---|---|---:|---:|---:|---:|
| M4a autoencoder | 1.74e-3 [9.67e-4, 3.12e-3] | 0.881 | 0.0% | 1,565 | 0.020 |
| M5c Euler | 1.22e-3 [7.15e-4, 2.51e-3] | 0.985 | 0.0% | 8,481 | 0.110 |
| M4b VAE | 7.86e-4 [3.69e-4, 2.48e-3] | 0.971 | 0.0% | 5,005 | 0.065 |
| M3a Isolation Forest | 4.97e-5 [2.47e-5, 1.56e-4] | 0.938 | 0.0% | 55,094 | 0.537 |
| M3b HBOS | 6.15e-5 [3.42e-5, 1.33e-4] | 0.930 | 0.0% | 8,987 | 0.116 |

The others change their AUC-PR by less than 1 % and their first detection by less than 1 %
(PCA, ECOD, M1, M1u) or not at all (M0, M2a, M2c); 60 of 78 pairs are significant here too.

### Reading the temporal results

- **No model catches a campaign at 10 alerts a day**, under any tie order (every bracket's upper
  end at 10 is 0 except M2a, below).
- **M1 has the cheapest first detection**, 42 alerts a day (p = 0.001), and catches 1, 2 and 3
  of 52 campaigns at 50, 100 and 500, while its ROC-AUC is 0.496. Its weights are tuned on
  validation labels (`tuning.json`); with uniform weights (M1u) the same rules rank the red
  team below benign traffic (below).
- **PCA reaches 9.6 % of campaigns (5 of 52) at 500 a day**, first detection at 246. Tuning
  chose `n_components = 0.5` (validation AUC-PR 0.0055) over the default 0.95 (6.7e-5), the
  value the September run used (`tuning.json`, `models/stats.py`).
- **Euler** has ROC-AUC 0.985 and the third AUC-PR, but its first detection (15,945 a day) is
  not better than a random ranking (p = 0.203).
- **M2a's zeros depend on the tie order.** Its rarity is anchored to the start of the day since
  22 October (limitation 12), so equally new events tie: about 118,500 events compete for the
  last slots over the three test days. Breaking ties in arrival order (`(time, event_id)`, the
  stated convention) gives 0 %; the most favourable order would give 32.7 % at 10 and 67.3 % at
  50 to 500 a day (`campaign_recall_tie_bracket`, `tie_exposure` in `metrics_summary.json`).
  That upper end needs an oracle to choose among the ties, so it bounds what the tie hides
  rather than measuring M2a; every other model's bracket has zero width.
- **ROC-AUC does not order the models operationally**: the autoencoder and Euler have the two
  highest ROC-AUC values, M1 is at chance on it and has the best first detection.

### Against the run of 15 September (same split, earlier code)

Both runs test the same 282 events in 52 campaigns. Between them the code changed in ways
recorded in `docs/limitations.md` (items 10 to 12) and `docs/methodology.md` ("Labels"): equal
tuning budgets, M1's normalisation and ECOD fitted on training data, rarity anchored to the
start of the day; the autoencoders and Euler were added and ECOD is no longer excluded. The
largest differences: PCA's AUC-PR from 3.50e-5 to 1.64e-3 and its first detection from 1,569 to
246 (with the tuning choice above); M1's first detection from 999 to 42 and its AUC-PR from
4.69e-4 to 2.13e-4; M2a's AUC-PR from 7.75e-4 to 6.30e-4 with a tie bracket that is no longer
`[0, 0]`. This run does not separate the effect of each change.

## harmonised protocol

Train on hours 0 to 41, test on hours 41 to 1392 (days 1 to 57), remote LogOn events only,
every event compared with all earlier ones (`conf/split/harmonised.yaml`). R1 only: there is no
validation partition, so nothing is tuned, no label is read, and M1 keeps uniform weights
(identical to M1u). Test: 702 red-team events, 162 campaigns over 98 accounts
(`harmonised/tables/positives_ledger.csv`, `budget_summary.json`).

| Model | AUC-PR [95% CI] | ROC-AUC | Camp. @10 | @50 | @100 | @500 | 1st detection | p random |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M5c Euler | 7.19e-4 [4.75e-4, 1.24e-3] | 0.961 | 0.0% | 0.0% | 0.0% | 0.6% | 203 | 0.022 |
| M2a pair rarity | 4.21e-4 [2.59e-4, 8.78e-4] | 0.949 | 0.0% | 0.0% | 0.0% | 1.2% | 405 | 0.040 |
| M3b ECOD | 1.71e-4 [1.07e-4, 3.60e-4] | 0.972 | 0.0% | 0.0% | 0.0% | 0.0% | 1,691 | 0.157 |
| M3a Isolation Forest | 7.37e-5 [4.43e-5, 1.31e-4] | 0.957 | 0.0% | 0.0% | 0.0% | 0.0% | 718 | 0.072 |
| M3b HBOS | 5.92e-5 [4.20e-5, 9.21e-5] | 0.979 | 0.0% | 0.0% | 0.0% | 0.0% | 1,262 | 0.120 |
| M4b VAE | 5.53e-5 [3.52e-5, 9.50e-5] | 0.959 | 0.0% | 1.2% | 1.9% | 4.3% | 27 | 0.003 |
| M4a autoencoder | 3.94e-5 [2.55e-5, 6.85e-5] | 0.935 | 0.0% | 0.0% | 0.6% | 1.9% | 59 | 0.007 |
| M2c source rarity | 1.58e-5 [1.08e-5, 2.83e-5] | 0.926 | 0.0% | 0.0% | 0.0% | 0.0% | 3,737 | 0.309 |
| M2b PCA | 1.23e-5 [7.83e-6, 1.78e-5] | 0.888 | 0.0% | 0.6% | 0.6% | 2.5% | 20 | 0.003 |
| M0a random *(floor)* | 1.96e-6 [1.36e-6, 3.18e-6] | 0.504 | 0.0% | 0.0% | 0.0% | 0.0% | 16,718 | 0.817 |
| M0b always-fail *(floor)* | 1.95e-6 [1.38e-6, 3.20e-6] | 0.498 | 0.0% | 0.0% | 0.0% | 0.0% | 20,636 | 0.876 |
| M1 rules (uniform, no validation) | 1.16e-6 [8.35e-7, 1.78e-6] | 0.244 | 0.0% | 0.0% | 0.0% | 0.0% | 41,065 | 0.985 |
| M1u rules (uniform) | 1.16e-6 [8.35e-7, 1.78e-6] | 0.244 | 0.0% | 0.0% | 0.0% | 0.0% | 41,065 | 0.985 |

69 of the 78 pairwise AUC-PR differences are significant after Holm (`evaluate_harmonised.log`).

- **No model catches a campaign at 10 alerts a day** under the arrival-order tie-break.
- **The cheapest first detections are PCA (20 a day), the VAE (27) and the autoencoder (59)**,
  each with p below 0.01; the VAE catches 7 of 162 campaigns at 500 a day. These p-values are
  per model, not adjusted for 13 comparisons.
- **AUC-PR and the operational view disagree again**: Euler has the best AUC-PR and a first
  detection at 203; PCA has the lowest AUC-PR above the floors and the cheapest first
  detection.
- **Ties**: M2a's bracket is [0, 32.1 %] at 10 and up to [0, 61.7 %] at 500 (about 216,500
  contenders over 57 days), M0b's [0, 0.6 %]; every other bracket has zero width.

## M1u: why uniform rules rank the red team below benign traffic

Scored rule by rule on every malicious test event and a seeded 1-in-25 hash sample of benign
ones, which leaves ROC-AUC unbiased (`diagnostics/`, script beside the outputs):

| Rule | temporal | harmonised |
|---|---:|---:|
| R1 new pair | 0.72 | 0.70 |
| R8 new source | 0.56 | 0.55 |
| R2 host burst | 0.54 | 0.59 |
| R6 new auth type | 0.52 | 0.51 |
| R4 off hours | 0.38 | 0.38 |
| R7 lateral chain | 0.32 | 0.49 |
| R3 fail then success | 0.23 | 0.15 |
| R5 machine account anomaly | 0.19 | 0.17 |
| M1u, all rules (pipeline) | 0.244 | 0.244 |

Machine accounts make 61.7 % (temporal) and 66.8 % (harmonised) of benign events and none of
the malicious ones, so R5 points the wrong way; R3 and R4 do too, and R7 on temporal. Summing
the eight rules with equal weights, three or four of them ranking the red team low, puts the
combination below chance. This is the rules' behaviour on LANL, not a bug: the per-rule values
come from the same `rule_scores` that M1u sums (`models/rules.py`).

## First-detection intervals are one-sided

For all 13 models, in both regimes of `temporal` and in `harmonised`, the bootstrap interval of
the first detection has its lower end equal to the point value. A minimum over campaigns cannot
be beaten by resampling accounts, only kept or lost; the interval is an upper bound on what the
first detection could need. This is limitation 13 in `docs/limitations.md`; the quantile budgets
and the log-area are not affected in this way.

## Euler

### Fidelity check of the reimplementation

`scripts/euler_fidelity.py` runs AuthBench's Euler under the paper's protocol: training before
the first red-team event, edges scored per 30-minute snapshot, edge-level AUC and AP against
table VI of King and Huang (2022). Five seeds (42 to 46) per configuration
(`euler_fidelity_<config>.json`):

| Configuration | Events | Test edges (anomalous) | AUC mean +- sd | AP mean +- sd |
|---|---|---|---|---|
| all events | 1,051,430,459 | 99,430,341 (525) | 0.9521 +- 0.0137 | 0.00035 +- 0.00024 |
| NTLM only, as the authors' loader | 44,748,939 | 14,400,984 (525) | 0.9570 +- 0.0028 | 0.0027 +- 0.0011 |
| NTLM, authors' node vocabulary | 44,748,939 | 14,400,984 (525) | 0.8261 +- 0.0341 | 0.00011 +- 0.00003 |
| NTLM, vocabulary and their defaults (lr 0.005, patience 5) | 44,748,939 | 14,400,984 (525) | 0.9590 +- 0.0145 | 0.0013 +- 0.0006 |
| published (table VI) | 45,871,390 | (518) | 0.9906 | 0.0155 |

Five attempts came before this table: three were OOM-killed, two were single-seed passes since
superseded, and one relaunch did not run at all; `REVIEW_NOTES.md` lists them in order. A conclusion drawn from one seed during
the run (NTLM multiplying AP by 20 and adding 2.6 points of AUC) did not survive five seeds:
about 8 times the AP, and 0.005 of AUC, within seed noise.

### The authors' code on the same machine

iHeartGraph/Euler at commit `b6ab8fd` (no licence; run, not redistributed), edited only where
its README asks for paths (`euler_original/paths.diff`), Python 3.10 with torch 1.13.1+cpu and
PyG 2.2.0 (`euler_original/euler_original_pip_freeze.txt`), `run.py -w 4 -t 5 -d 0.5 -e GCN -i
PRED`: five repetitions, alone on the machine after the evaluations, exit 0
(`euler_original/results.txt`).

| | AUC | AP |
|---|---|---|
| Authors' code, this machine, mean +- sd over 5 | 0.9879 +- 0.0022 | 0.0058 +- 0.0018 |
| Published (table VI) | 0.9906 | 0.0155 |
| AuthBench, NTLM only, 5 seeds | 0.9570 +- 0.0028 | 0.0027 +- 0.0011 |

`results.txt` reports the standard error; the sd is it times the square root of 5. **The
authors' own code reproduces the published AUC closely but not the published AP**: 0.0040 to
0.0080 over the five runs, against 0.0155. **AuthBench's reimplementation stays below their
code**, by about 3 points of AUC and half the AP, on the same data (next section). Choices
present in their code and absent from the paper, which AuthBench does not reproduce: 10
negative samples per edge in training, early stopping on validation AUC plus AP, dropout 0.25
and DropEdge(0.8), one optimiser step per epoch over the whole training sequence. Testing which
of them closes the gap needs the whole dataset, so another rented machine.

### The test sets, reconciled edge by edge

- **Red-team events**: 715 distinct rows; 701 have an authentication event at the same second
  with the same computers (all of them NTLM), 14 have none, so no pipeline can label them
  (`euler_original/redteam_matches.out`).
- **Same data on both sides**: recounted in AuthBench's snapshot alignment (multiples of
  1,800 s), the authors' prepared files give 14,400,984 test edges and 524 anomalous, 525 with
  the first red-team line their `split.py` skips as a header: exactly AuthBench's NTLM test set
  (`euler_original/authors_test_edges.out`).
- **The paper's 518**: their loader aligns snapshots on the test start (t = 150,885 s), giving
  14,416,149 edges, of which it drops the 1,374 of the last, partial snapshot; 517 anomalous,
  518 with the skipped first line, which is table V's figure.
- **The 514 they evaluate**: their predictor never scores the first test snapshot
  (`decode_all` starts at 1 on the head worker), which holds 3 anomalous edges. Their own loader,
  run on the machine, hands the workers 14,414,775 edges and 517 anomalous edges
  (`euler_original/authors_test_load.out`); 517 - 3 = 514, which TP / TPR gives in each
  repetition.
- Their first red-team event (t = 150,885 s, C17693 to C1003) is labelled benign in their files
  because of the skipped line.

## Machine incidents

The kernel's OOM killer acted five times (`machine/kernel_oom.log`): the first fidelity check
at 50.8 GB (one aggregation over the whole timeline), two fidelity reruns at their 24 GB cgroup
cap, the authors' code at its 16 GB cap, and a global OOM at 17:42 UTC while the authors' code
ran beside the evaluations with a 30 GB cap. The authors' code ran with `oom_score_adj` 1000
(printed in the kill line), so that last kill took its process and not an evaluation; the
fidelity reruns were confined by their cgroup caps instead. Both evaluations finished without
error. One line of `run.log` reported a fidelity rerun as exit 0
when it had not run at all (no HOME in its unit); the correction is logged below it. The LANL
data-use form e-mail is redacted from `machine/console_logs/download.log`.

## Read before quoting

- **One intrusion**: almost all test red-team events start from one host (limitation 9), so
  campaign intervals describe one operation's accounts and phases, not many attacks.
- **First-detection intervals are one-sided** (above, limitation 13).
- **p random** values are per model; with 13 models a value near 0.05 is weak evidence.
- **Ties**: read M2a's recall with its bracket.
- **ROC-AUC** is reported for comparison with the literature only (`docs/limitations.md`).

## Files

| Path | Content |
|---|---|
| `temporal/`, `harmonised/` | metrics, tables, figures, MLflow database per protocol; R2 under `r2_semi_supervised/` |
| `run.log`, `run.json`, `machine.txt`, `raw_sha256.txt`, `*.log` | run log, duration and cost, machine, source hashes, step logs |
| `euler_fidelity_*.json`, `euler_fidelity_*.log` | the fidelity check, every attempt |
| `euler_original/` | the authors' code: results, logs, environment, path diff, scripts and the test-set recount |
| `diagnostics/` | M1u rule by rule, script and outputs |
| `machine/` | kernel OOM log, systemd log, console logs, smoke run, package list, helper scripts |
| `REVIEW_NOTES.md` | working notes, in the order things happened |
