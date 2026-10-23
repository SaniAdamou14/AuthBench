# Review round 3 — Artifact evaluation and code review (R3)

Reviewer role: independent artifact-evaluation committee member (ACM/USENIX style) and code reviewer.
Scope: repository at `cf6f6f8` (tag `v2.0.0`), accompanying report `reports/paper/main.tex`.
Earlier review rounds were not read. Every finding is marked **[verified]** (I ran it or read the
code path end to end) or **[suspected]** (plausible from reading, not demonstrated).

---

## 1. What the artifact does

AuthBench is a Python harness (Polars, scikit-learn/PyOD, PyTorch) that turns the LANL
authentication log into a day-partitioned feature store with causal features, fits thirteen
anomaly detectors on a strictly earlier partition, and scores the test partition one day at a time.
Instead of only aggregate AUC-PR/ROC-AUC, it reduces every detector's scores to the per-day queue
rank of each red-team event, from which it derives campaign recall at daily alert budgets, the
per-campaign first-detection budget, a tie-break bracket, edge-hour and user-day units, an exact
random-ranking null for the first detection, account-level block-bootstrap BCa intervals and
Holm-corrected pairwise tests. A synthetic demo sample exercises the protocol in ~40 s; a
generator rebuilds every number of the report from committed run outputs.

## 2. Commands run and results

Environment: Windows 11, existing `.venv` (Python 3.11.3), uv 0.11.32. Nothing tracked was modified; the
demo and the asset generator were run in a scratch `git clone` of `HEAD`.

| Command | Result |
|---|---|
| `.venv\Scripts\python -m pytest` (repo `addopts`: `-ra -q --cov=authbench`) | **376 passed, 30 warnings in 146.57 s**. 0 failed, 0 skipped. Warnings are all sklearn `max_samples (256/1024) is greater than the total number of samples (160)` from IForest on 160-row test fixtures (harmless). Total coverage 84%. |
| `coverage report --include=evaluate/*,features/*,ingest/*,models/* --fail-under=85` (the CI gate) | **95%, pass** (exit 0). |
| `ruff check src tests scripts` (ruff 0.16.10) | `All checks passed!` |
| `ruff format --check src tests scripts` | `101 files already formatted` |
| `mypy src` (mypy 2.4.0, `strict = true`, `disallow_any_generics = false`) | `Success: no issues found in 53 source files` |
| `uv lock --check --offline` | `Resolved 200 packages` — lockfile consistent with `pyproject.toml`. |
| `python -m authbench.cli demo` (in clone) | Completed in 34.5 s (39 s wall). `reports/demo/tables/*.json` and `data_quality.json`: **byte-identical** to the committed copies. Four CSVs flagged by `git diff` are identical modulo CRLF (clone checked out with `core.autocrlf=true`; no `.gitattributes`). The demo PNG differs (not claimed reproducible; CI does not check it). |
| `python scripts/paper_assets.py` (in clone) | 12 s; **every file in `reports/paper/generated/` byte-identical** (`git status` clean for that directory). |
| `gh run list` | Last 5 CI runs on `master` / PR branches: all `success` (≈2.3 min each). |
| Own probe 1: 200 random tie-heavy frames (1-3 days, 4 distinct score values, repeated timestamps), comparing `SplitAccumulator` → `budget_curve`, the whole-frame `evaluate/budget.py`, and my own brute-force per-day queue | **0 mismatches** in campaign recall, first-detection budget, tie bracket min/max; `roc_auc` and `average_precision` equal scikit-learn's. |
| Own probe 2: `random_first_detection_survival` vs 10⁵ Monte Carlo draws | Agrees to MC error (e.g. b=10: exact 0.3494, MC 0.3506). |
| Own probe 3: `build_store(..., history="cumulative")` on the demo with the harmonised hour-41 cut; permute `dst_computer` and `success` of every event after the cut (including same-day events) | **Training-partition features bit-identical** before and after. No future-to-past leakage through F1–F4, the carried state, or the train-fitted encodings. |

Not run: the LANL pipeline (data not present; ~55 GB disk, ~19 GB RAM, ≈17 machine-hours).

## 3. Badge assessment

| Badge | Assessment | Reasons |
|---|---|---|
| **Available** | **Awarded** | Public GitHub repo, Apache-2.0, Zenodo DOI for the exact release (`v2.0.0`, 10.5281/zenodo.23588281), `CITATION.cff`. Data cannot be redistributed; acquisition is scripted (`ingest/download.py`) and documented. Run outputs, logs and provenance for the paper run are committed (`reports/runs/20261022T102638Z/`). |
| **Functional** | **Awarded** | Documented install (`uv sync --locked`), pinned lockfile, all tests/lint/types pass, demo runs end to end in < 1 min and reproduces its committed results byte for byte on a second OS, the paper's numbers regenerate byte for byte from committed outputs. Tests assert properties (bit-exact equality with whole-window references, sklearn and brute-force cross-checks, exhaustive enumeration of the null), not just "it runs". |
| **Reusable** | **Borderline — not awarded as is; likely after fixing M3–M5** | Clean `AnomalyScorer` protocol, strict typing, extensive docstrings, runtime row-locality enforcement. But: the configuration surface is partly dead or contradicts the code (M4), the DVC pipeline and Makefile do not reproduce the paper or do not run off Windows (M3), adding a detector means editing four hard-coded places with no registry or guide (M5), and only LANL has an ingest adapter (`conf/dataset/cert.yaml` has no implementation). |
| **Results Reproduced** | **Not awarded on this evidence** | I verified the *last* step (committed outputs → every table/number) and the demo. The LANL run itself could not be replayed here. It is replayable in principle by `scripts/run_lanl.sh` (≈17 h, ≈€9), but the published run exited 1, `run.json` was written by hand, the raw files came from a mirror whose SHA-256 differ from `conf/dataset/lanl.yaml` (content checked by total row count plus 3 of 58 days), and the Euler fidelity/authors'-code results are "not yet one command" (main.tex:561). The orchestration code that produced the paper numbers has no automated test (M2). |

## 4. Verdicts

**(a) Master's thesis software deliverable: clear pass, top band.** Well above what I would expect of
a Master's thesis: a typed, linted, 376-test code base with a 95% coverage gate on the core,
bit-exact determinism engineering (`models/rowwise.py`), a self-checking evaluation accumulator,
documented provenance including the failures, and a byte-for-byte reproducibility check in CI. The
problems below are the kind an examiner would list as corrections, not as reasons to fail.

**(b) Artifact accompanying a PhD-student paper at a peer-reviewed security venue: acceptable;
Available + Functional now, Reusable within reach, Results Reproduced not yet.** It is above the
median artifact at USENIX Security/CCS/NDSS AE in test quality and honesty of provenance. Two things
should be fixed before submitting it there: the paper describes a leakage guard that is not on the
code path that produced its numbers (M1), and the paper-producing pipeline is untested and its
documented entry points (DVC, Makefile) do not reproduce the paper (M2, M3). Neither invalidates a
reported number as far as I can tell (my leakage probe and metric probes passed).

## 5. Major issues (ranked)

### M1. The leakage guard the paper describes is not on the path that produced any paper number [verified]

- **Where.** Paper `reports/paper/main.tex:137`: "A guard raises an error unless the last training
  timestamp precedes the first validation timestamp … A static architecture test fails if any
  module under `models/` or `features/` imports the only function that loads the test partition."
  That guard is `split/temporal.py:57-85` (`verify_temporal_order`) and `get_test_split`
  (`split/temporal.py:47-54`), enforced by `tests/unit/test_no_test_leakage.py:13-50`.
  Their only production caller is the demo (`cli.py:559-562`). The LANL path never calls them:
  partitions are defined by `pipeline/build_features.py:103-120` (`split_ranges`) and written by
  half-open filters (`build_features.py:154-161`); `pipeline/train_eval.py:296-299` loads the test
  partition directly with `pl.scan_parquet(version_dir / "test" / "*.parquet")`; and models do
  receive test frames, by design, through `prepare(timeline)` (`train_eval.py:338, 395-398`;
  Euler, `models/euler.py:289-313`).
- **Why it matters.** The actual separation is sound: `split_ranges` rejects overlapping or
  unordered config ranges, `tests/integration/test_build_store.py` checks written partitions
  bit for bit, the Euler causality test (`tests/unit/test_euler.py:52-60`) and my perturbation
  probe pass. But the paper's stated mechanism, and the test that "fails the build", protect a
  function the evaluated pipeline does not use; the AST test only matches the identifier
  `get_test_split`. A reviewer who checks will find the claim does not describe the code.
- **Fix.** Either (i) make `train_eval.main` assert on data, not config: after scanning, check
  `max(train.time) < min(val.time) ≤ … < min(test.time)` with `verify_temporal_order`, and assert in
  `tune_and_fit`/`fit` callers that every fitted frame ends before the test start; add a test that
  `fit` is never called with a frame reaching into `ranges["test"]` (e.g. a spy scorer); or
  (ii) rewrite main.tex:137 to describe `split_ranges` + half-open filtering + the bit-exact
  `build_store` test, and say that `prepare` deliberately sees the unlabelled test timeline.

### M2. The code that produced the paper's numbers has no end-to-end test; the demo is a separate implementation [verified]

- **Where.** Coverage report: `pipeline/train_eval.py` 47% (missing 279-330, 374-540: `main` and
  `evaluate_regime` entirely), `pipeline/build_features.py` 74% (missing 201-258: `main`),
  `evaluate/ledger.py` missing 282-310 (`write_ledger_outputs`). `authbench demo`
  (`cli.py:493-…`) re-implements the pipeline: it featurises each split with no warm-up and no
  cumulative history, scores with `statistics_from_frame` rather than `score_split`, never calls
  `prepare`, `calibrate_alert_thresholds` or the Hydra entry points, and never runs the harmonised
  split, the deep models or Euler.
- **Why it matters.** CI's byte-for-byte check covers the demo, not the code that produced the
  report. The components are well tested in isolation (the row-locality and statistics
  equivalence tests are strong), but wiring errors in `evaluate_regime` (regime loop, R2 timeline,
  thresholds source, block mapping, output paths) would only show up on a 17-hour run.
- **Fix.** Add an integration test that composes the Hydra config (`hydra.compose`) and runs
  `build_features.main` + `train_eval.main` on the demo data in `tmp_path` for both `split=temporal`
  (demo days) and `split=harmonised`, with `runtime.deep_models`/`euler` on and small epochs; assert
  the output files exist and that the R1 table equals the one from `statistics_from_frame` on the
  same scores. Better: make `authbench demo` call the same functions.

### M3. Documented reproduction entry points do not reproduce the paper; the one that did failed and was patched by hand [verified]

- **Where.**
  - DVC: `params.yaml:23` `to_parquet_days: "0:13"`, `split` temporal only; `conf/config.yaml`
    `runtime.deep_models: false`, `euler: false`; `dvc.lock` records only `download` and
    `to_parquet` (`--days "0:13"`), so `dvc repro` has never been run to completion for any
    published result. Stage deps omit the stage's own code and configs:
    `dvc.yaml:41-58` (no `src/authbench/pipeline`, `src/authbench/parse`,
    `conf/split/harmonised.yaml`), `dvc.yaml:60-66` (no `src/authbench/pipeline`, no
    `conf/config.yaml`). `dvc.yaml:86-88` still says `reports/paper/` does not exist.
  - Makefile: `Makefile:4,13,17,20…` hard-code the Windows launcher `py -3.11 -m uv`; `make
    install`/`make demo` (README:231-232) fail on Linux/macOS.
  - The paper run used `scripts/run_lanl.sh`, which exited 1 (Euler fidelity OOM);
    `reports/runs/20261022T102638Z/run.json` is marked `written_by_hand`; fidelity configs and the
    authors' Euler code were driven by separate machine-specific scripts (main.tex:561).
  - Data provenance: `raw_sha256.txt` (`eee54f…`, `e8ffba…`) ≠ `conf/dataset/lanl.yaml` `sha256`
    (`9c6b0c…`, `606635…`); RUN.md establishes content identity by total row count and order-free
    hashes of days 0, 5 and 13 only.
- **Why it matters.** "From README to the paper's tables" today means: trust the committed
  outputs, or run a script that is known to fail at the end and then hand-assemble the rest. An AE
  committee would ask for one command per table.
- **Fix.** Declare `run_lanl.sh` the supported path; split the Euler fidelity step into its own
  script so the evaluation exits 0; script the authors'-code run; either delete DVC or make
  `dvc.yaml` mirror `run_lanl.sh` (both protocols, `runtime.*` as params, full deps); make the
  Makefile portable (`uv run …`); record per-day content hashes for all 58 days so a mirror can be
  verified completely.

### M4. Dead and contradictory configuration [verified]

- **Where.**
  - `conf/features/base.yaml` declares F2 `windows_hours: [1, 24, 168]`,
    `entities: [src_user, src_computer, dst_computer]`, per-family `enabled` flags,
    `categorical_encoding`, `new_pair_sentinel_days`, `night_window_calibration`. Production
    featurisation ignores all of them: `features/incremental.py:254-260` hard-codes
    `DEFAULT_F2_WINDOWS_HOURS = [1, 24]` and two entities (`features/history.py:47-48`), and
    `build_store` (`build_features.py:139-190`) never passes the config to `featurize_day`, so the
    `enabled` flags are honoured only by the test reference `featurize` (`build_features.py:66-86`).
    Only `universe` and `history_warmup_days` are read. The dead keys still enter the
    feature-store hash (`build_features.py:58-63`), are mirrored in `params.yaml:31-34`, and are
    "protected" by `tests/unit/test_params_match_conf.py`, which checks params against conf but not
    conf against code.
  - `conf/model/rules.yaml`: 7 rules, `requires_labels: false`, `n_trials: 200`; code:
    8 rules (R8), `requires_labels = True` (`models/rules.py:95`), 20 candidates
    (`pipeline/tuning.py`). `conf/model/{iforest,pca,ecod,lof,…}.yaml` are not read
    (`_model_params` is used only for `ae`, `vae`, `euler`, `train_eval.py:124-146`);
    `conf/config.yaml` default `model: rules` is unused; `conf/dataset/cert.yaml` has no
    implementation.
- **Why it matters.** For a harness whose selling point is reuse, a user who sets
  `f2_history.windows_hours` or disables a family gets a new feature-store hash and identical
  features, silently.
- **Fix.** Delete the dead keys/files or wire them through `featurize_day`/`build_store`; add a
  test that loads each config and fails on unread keys.

### M5. Adding a detector requires editing several hard-coded places; no registry or guide [verified]

- **Where.** Catalog list `train_eval.py:107-118`; demo catalog in `cli.py`; tuning by name and
  `isinstance` chains in `pipeline/tuning.py:35-51` (`GRIDS`), `:72-99` (`_variant`),
  `:102-109` (`_config_of`), `:113-136` (`SEEDED`, `reseed`); `scripts/paper_assets.py:51-80`
  (`LABELS`, `CODES`). The parametrised row-locality test builds its list from
  `build_model_catalog(...)` without `euler=True` (`tests/unit/test_row_locality.py:55-60`); Euler's
  row-locality is tested separately (`tests/unit/test_euler.py:63-71`), so a new detector added
  behind a flag can escape the parametrised test.
- **Why it matters.** The protocol itself (`models/base.py`) is clean, and `score_split` refuses
  non-row-local models at runtime (`train_eval.py:212-219`), which is the right design. But the
  "harness" claim (main.tex:565, "cheap to evaluate") needs an extension point.
- **Fix.** A small registry keyed by model id carrying constructor, grid, seeded flag and display
  label; a `docs/adding_a_detector.md` with a 30-line example; parametrise the row-locality test
  over every registered model including flagged ones.

## 6. Minor issues

1. **Stale docstring contradicts code** [verified]. `models/base.py:29-34` says `scores_row_locally`
   "is False for M1"; `models/rules.py:98` sets it True, and `train_eval.score_split` would refuse M1
   otherwise.
2. **Stale README claim** [verified]. `README.md:159-162`: "the LANL runs predate the change" (F3
   source-side relations). Commit `035a7e9` (F3) is an ancestor of the run commit `bd833eb`
   (`git merge-base --is-ancestor`), and `user_src_rarity`/`src_dst_rarity` are in
   `MODEL_FEATURE_COLUMNS`.
3. **Harmonised fixed thresholds are calibrated per calendar-day bucket, not per 24 h** [verified].
   `calibrate_alert_thresholds` counts `n_unique(day)` = 2 on the 41-hour training range
   (`train_eval.py:272-273`) and takes the `k·n_days`-th score (`evaluate/statistics.py:612-619`),
   i.e. ≈1.17 k alerts per 24 h, not k as main.tex:223 states. It does not change the conclusion
   (PCA raises 149/day against a nominal 10), but the sentence should match the code, or divide by
   hours/24.
4. **Partial boundary day** [verified]. The harmonised first test day holds hours 41-47 only, and the
   per-day budget applies to it unchanged. The paper and README disclose that this drives the
   autoencoder's 59; consider reporting results with that day pro-rated or excluded as a
   first-class variant rather than a footnote.
5. **Version metadata** [verified]. `pyproject.toml:3` `version = "0.1.0"` (installed metadata
   reports 0.1.0) while the release, `CITATION.cff` and `.zenodo.json` say 2.0.0.
6. **Model ids** [verified]. HBOS and ECOD are both `M3b_*` in code (`models/classical.py:73,124`)
   and relabelled M3b/M3c only in `paper_assets.py`; raw JSON outputs and the paper disagree on
   ids.
7. **No `.gitattributes`** [verified]. On Windows with `core.autocrlf=true`, CI's `git diff
   --exit-code` reproducibility check reports four demo CSVs as changed although their content is
   identical. Add `*.csv text eol=lf` (and the same for `.json`, `.tex`).
8. **"Exact null" wording** [verified]. README:78-80 calls the 3.0% user-day campaign-recall null
   "exact"; `paper_assets.py:565-616` estimates it by simulation (only the first-detection null is
   exact).
9. **Multiplicity on the first-detection p-values** [verified as a reporting choice]. `P(random ≤
   it)` is exact per detector, but the headline values (0.002, 0.003) are minima over 12-13
   detectors. Say so next to the table, or give a family-wise version.
10. **`sample_for_fit` materialises the whole training set before sampling** (`models/base.py:77-82`)
    [verified]. Memory, not correctness. Sample lazily (hash on `event_id`) to keep the RAM figure
    in the README honest for larger datasets.
11. **`random_ablation_split`** (`split/temporal.py:99-113`) uses `hash % n` as a rank, which is
    not a permutation, so partition sizes are approximate; untested (lines uncovered)
    [verified]. Not used in the paper.
12. **Cross-machine determinism of torch models** [suspected]. Euler and the autoencoders enable
    deterministic algorithms and seed everything, and the tests check same-machine repeatability.
    CPU matmul/GRU results can still differ with thread count or CPU ISA, so LANL AE/VAE/Euler
    numbers may not reproduce bit for bit on different hardware. Document the expected tolerance,
    or pin `torch.set_num_threads`.
13. **`hypothesis` is a dev dependency but unused** (no `@given` anywhere) [verified]. Either use it
    (the accumulator is a natural target, as my probe 1 shows) or drop it.
14. **Account block key** [suspected, negligible on LANL]. Campaigns are grouped by `user@domain`
    (`label/redteam_join.py:80-84`) but bootstrap blocks and gap regrouping use `src_user` only
    (`evaluate/ledger.py:313-320`, `:224-230`).

### What is done well (for calibration, briefly)

- Row-locality to the last bit, by construction (`models/rowwise.py`) and checked at runtime on
  every test day (`evaluate/statistics.py:287-292`), with a test that the check fires
  (`tests/unit/test_statistics.py:134`).
- Tie handling is explicit, label-free and bracketed. My independent brute force agrees on
  tie-heavy data.
- The exact null (`evaluate/budget.py:299-312`) is correct, `C(N−b,p)/C(N,p)` per day with the
  boundary handled, and tested by exhaustive enumeration (`tests/unit/test_budget_metrics.py:235`).
- ECOD was re-implemented to remove PyOD's batch dependence (`models/classical.py:61-113`); the
  formula matches PyOD's skewness-corrected tails.
- Provenance is unusually candid: mid-run code changes, OOM kills and the hand-written `run.json`
  are all recorded (`reports/runs/20261022T102638Z/RUN.md`).

## 7. What would move it up one level

- **From Functional to Reusable:** fix M4 (one source of truth for configuration, with a test that
  every key is consumed), M5 (detector registry + "add a detector" guide + row-locality test over
  all registered models), the portable Makefile from M3, and a second dataset adapter or at least a
  documented adapter interface in place of the dead `cert.yaml`.
- **From "derivation verified" to Results Reproduced:** M2 (Hydra-level integration test of
  `build_features.main` + `train_eval.main` on both splits), and M3: one command that runs on a
  stock Ubuntu VM, exits 0, and produces `reports/runs/<id>/` including the Euler fidelity and
  authors'-code tables, plus per-day content hashes so a mirror copy can be verified in full. Then
  a committee could reproduce the paper for about €10 and a day.
- **For the paper:** align the leakage paragraph (main.tex:137) with the code (M1), and fix the
  calibration sentence (minor 3).
