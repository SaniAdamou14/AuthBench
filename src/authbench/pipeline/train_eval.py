"""DVC `train_eval` stage: feature store -> fitted models -> evaluation tables/figures.

Trains the dependency-light catalog (M0 floors, M1 rules, M2 stats, M3
classical) end to end and writes the comparison tables. Deep (M4) and graph
(M5) models are heavier, optional-extra runs — invoke them individually via
`authbench train model=ae` (Hydra model group) once `authbench[deep]` /
`authbench[graph]` are installed, rather than bundling them into the default
stage that every contributor's machine must be able to run.

Scoring is separated from evaluation on purpose. Every model scores the test
split first, all scores land on one slim frame, and a single
campaign-stratified resampling pass then produces both the per-model
confidence intervals and every pairwise comparison — see
`evaluate.stats_tests.paired_campaign_bootstrap`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from authbench.evaluate.metrics import ROC_AUC_WARNING, auc_pr
from authbench.evaluate.plots import CAMPAIGN_RECALL_FIGURE, plot_campaign_recall_vs_budget
from authbench.evaluate.stats_tests import paired_campaign_bootstrap
from authbench.evaluate.summary import (
    EVAL_COLUMNS,
    evaluate_model,
    pairwise_comparisons,
    score_column,
)
from authbench.features import MODEL_FEATURE_COLUMNS
from authbench.models.base import DEFAULT_FIT_SAMPLE_SIZE
from authbench.models.classical import ECODScorer, HBOSScorer, IsolationForestScorer
from authbench.models.floors import AlwaysFailScorer, RandomScorer
from authbench.models.rules import RulesScorer
from authbench.models.stats import PairRarityScorer, PCAReconstructionScorer
from authbench.pipeline.build_features import CONF_DIR, feature_store_version
from authbench.pipeline.tracking import open_tracker

logger = logging.getLogger(__name__)

#: Rows scored at once by the row-local models.
#:
#: Peak memory for the design matrix is this times 21 float64 columns times the
#: estimators' copy factor — about 1 GB at two million rows, and independent of
#: how large the split is. Lower it on a smaller machine; it cannot change any
#: score, only how many are computed at a time.
DEFAULT_SCORING_CHUNK_ROWS = 2_000_000

# The design matrix is `features.MODEL_FEATURE_COLUMNS`, shared with
# `authbench demo`. It used to be redeclared here, and the two copies had
# drifted: this stage was fitting M2b/M3a/M3b on 14 columns while the demo
# used 21, silently dropping every F2 history feature and all three F1
# frequency encodings — i.e. benchmarking different models under the same
# names in the two places that are supposed to agree.


# `RLIMIT_AS` (a hard cap on the process's virtual address space, tried
# in an earlier version of this file) does not work as an OOM guard for this
# workload and was removed: on the real run it rejected even the ~450 MB
# RandomScorer needs and a 417 MB bootstrap array while RSS was ~8 GB, because
# numpy/Polars/PyArrow routinely reserve far more virtual address space than
# they ever touch (the same run's OOM `dmesg` entry, before this change, read
# `total-vm: 114 GB` against `anon-rss: 30 GB` -- a >3x ratio). Linux also does
# not enforce `RLIMIT_RSS`, so there is no in-process rlimit that tracks the
# resident memory an OOM kill actually depends on. `exclude` below is the
# honest alternative: it costs nothing, unlike an address-space cap that is
# either too tight (kills cheap models as collateral damage, as above) or too
# loose to catch the real one before the kernel does.
def build_model_catalog(
    fit_sample_size: int | None = DEFAULT_FIT_SAMPLE_SIZE,
    *,
    exclude: set[str] | None = None,
) -> list[object]:
    """The catalog, *unfitted*. `main` is the single place that fits — an
    earlier version fitted M1 here and then refit every model in the loop,
    which at LANL scale meant paying for M1's per-user aggregation twice and
    left the calibrated weights one stray `fit()` away from being discarded.

    `fit_sample_size` bounds the rows the vector-space models estimate their
    distributions from. It is a published parameter rather than a hidden
    constant, because it is a real methodological choice: `None` fits on
    everything and is what the tests and the demo use.

    `exclude` drops named models before fitting even starts -- for a model
    known in advance to be unaffordable on the machine at hand (M3b_ecod's
    `decision_function` on the full LANL test split needs far more than a
    30 GB box has, and is transductive so it cannot be chunked -- see its
    class docstring), this is the honest alternative to attempting it and
    losing the run: `fit_and_score_all`'s `except (MemoryError, OSError)`
    only catches a graceful allocation failure, not a kernel OOM kill, which
    is what a model this size actually gets.
    """
    catalog = [
        RandomScorer(),
        AlwaysFailScorer(),
        PairRarityScorer(),
        PCAReconstructionScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size),
        IsolationForestScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size),
        ECODScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size),
        HBOSScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size),
        RulesScorer(),
    ]
    if exclude:
        catalog = [m for m in catalog if m.name not in exclude]  # type: ignore[attr-defined]
    return catalog


def fit_and_score_all(
    models: list[object],
    train: pl.LazyFrame,
    val: pl.LazyFrame,
    test: pl.LazyFrame,
    *,
    chunk_rows: int | None = DEFAULT_SCORING_CHUNK_ROWS,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Fit every model and return the slim evaluation frame carrying one
    score column per model.

    Holding all scores on a single frame is what lets the bootstrap below
    resample once for every model instead of once per model.

    Row-local models (`AnomalyScorer.scores_row_locally`) are scored
    `chunk_rows` at a time, so peak memory is a tunable constant rather than a
    property of the dataset. Chunking by *day* was the first attempt and is not
    fine-grained enough: a single LANL day is ~18 million events, which as a
    21-column float64 matrix in three estimator copies is 9 GB — more than the
    machine this was meant to fit on.

    Models with cross-row state — M1, whose R6 and R7 look backwards across the
    whole split — are scored whole, because chunking them would silently reset
    that state at each boundary.
    """
    n_rows = int(test.select(pl.len()).collect().item())
    skipped: dict[str, str] = {}

    for model in models:
        name: str = model.name  # type: ignore[attr-defined]
        logger.info("Fitting %s", name)
        model.fit(train)  # type: ignore[attr-defined]
        # M1 alone has a second, label-aware fitting step. It runs after
        # `fit` (which re-derives the thresholds its rules are built on)
        # and against the validation split only — never train, never test.
        if isinstance(model, RulesScorer):
            val_frame = val.collect()
            model.calibrate_weights(val_frame.lazy(), val_frame["is_malicious"], n_trials=200)
            del val_frame

    chunkable = [m for m in models if getattr(m, "scores_row_locally", False)]
    whole_split = [m for m in models if not getattr(m, "scores_row_locally", False)]

    if chunk_rows and chunkable:
        n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
        logger.info(
            "Scoring %d row-local model(s) over %d chunks of %d rows; "
            "%d model(s) need the whole split.",
            len(chunkable),
            n_chunks,
            chunk_rows,
            len(whole_split),
        )
        eval_parts: list[pl.DataFrame] = []
        chunk_scores: dict[str, list[np.ndarray]] = {
            m.name: []  # type: ignore[attr-defined]
            for m in chunkable
        }
        for offset in range(0, n_rows, chunk_rows):
            chunk = test.slice(offset, chunk_rows).collect()
            eval_parts.append(chunk.select(EVAL_COLUMNS))
            for model in chunkable:
                name = model.name  # type: ignore[attr-defined]
                if name in skipped:
                    continue
                try:
                    scores = model.score(chunk.lazy())  # type: ignore[attr-defined]
                except MemoryError as exc:
                    skipped[name] = f"{type(exc).__name__}: {exc}"
                    logger.error("%s failed on a chunk and is EXCLUDED: %s", name, exc)
                    continue
                chunk_scores[name].append(scores.to_numpy())
            del chunk

        # Concatenated in the order the slices were taken, which is the frame's
        # own order — so position still identifies the event, and every caller
        # attaches scores by position rather than by key.
        scored = pl.concat(eval_parts, how="vertical")
        scored = scored.with_columns(
            [
                pl.Series(score_column(name), np.concatenate(parts))
                for name, parts in chunk_scores.items()
                if name not in skipped
            ]
        )
        # Slicing preserves order, so the whole-split models see exactly the
        # same row order the chunked ones produced.
        ordered_test = test
    else:
        scored = test.select(EVAL_COLUMNS).collect()
        whole_split = list(models)
        ordered_test = test

    for model in whole_split:
        name = model.name  # type: ignore[attr-defined]
        logger.info("Scoring %s over the whole split", name)
        try:
            scores = model.score(ordered_test)  # type: ignore[attr-defined]
        except (MemoryError, OSError) as exc:
            # One model that cannot be scored must not take the other seven
            # with it. This is not hypothetical: M3b_ecod is transductive, so
            # PyOD concatenates the fitted sample onto the frame being scored
            # and argsorts all 21 columns of the result — 3.89 GB of int64
            # indices for a 20-million-row split, which killed a run that had
            # already spent twenty minutes building features on a smaller
            # split, and on the full 55-million-row test split climbed past
            # 30 GB RSS plus 20 GB of swap before the kernel OOM-killed the
            # whole process outright -- not a catchable exception, so this
            # block alone cannot save that run; `main` below is expected to
            # pass `exclude={"M3b_ecod"}` to `build_model_catalog` on a
            # machine this size instead. This except stays, uncapped, as a
            # cheap safety net for whichever whole-split model turns out to
            # be the next one that's merely close to the ceiling rather than
            # far past it -- for those, a graceful `MemoryError`/`OSError` is
            # plausible and worth catching.
            #
            # The model is recorded as unevaluated, with the reason, and the
            # report says so. A benchmark that silently drops a competitor is
            # worse than one that loses a run.
            skipped[name] = f"{type(exc).__name__}: {exc}"
            logger.error(
                "%s could not be scored on this split and is EXCLUDED from the results: %s",
                name,
                exc,
            )
            continue
        scored = scored.with_columns(pl.Series(score_column(name), scores))

    evaluated = [m for m in models if m.name not in skipped]  # type: ignore[attr-defined]
    if not evaluated:
        raise RuntimeError("Every model failed to score; there is nothing to report.")
    return (
        scored.select([*EVAL_COLUMNS, *[score_column(m.name) for m in evaluated]]),  # type: ignore[attr-defined]
        skipped,
    )


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    stratify_by = str(cfg.eval.bootstrap.stratify_by)
    if stratify_by != "campaign":
        raise ValueError(
            f"eval.bootstrap.stratify_by={stratify_by!r} is not supported. Only 'campaign' is: "
            "events within a campaign are strongly dependent, and resampling them "
            "independently would understate every confidence interval in the report."
        )

    processed_dir = Path(cfg.paths.processed_dir) / "features"
    version = feature_store_version(cfg.features)
    version_dir = processed_dir / version

    # Scanned, not read. `pl.read_parquet` on a real split materializes every
    # column of every row before a single model has been fitted; the lazy
    # frames below let each stage pull only what it needs — a sampled fit, a
    # day of scoring — and are the other half of why this stage now fits in
    # memory.
    train = pl.scan_parquet(version_dir / "train.parquet")
    val = pl.scan_parquet(version_dir / "val.parquet")
    test = pl.scan_parquet(version_dir / "test.parquet")

    tracker = open_tracker(str(cfg.mlflow.tracking_uri), str(cfg.mlflow.experiment_name))

    tables_dir = Path(cfg.paths.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    budgets = list(cfg.eval.budgets)
    fpr_targets = list(cfg.eval.fpr_targets)
    report_roc_auc = bool(cfg.eval.roc_auc.report)
    if report_roc_auc and bool(cfg.eval.roc_auc.warn):
        logger.warning("%s", str(cfg.eval.roc_auc.warning_message).strip() or ROC_AUC_WARNING)

    # A machine-specific affordability call, not a methodological default:
    # unset almost everywhere, so `dvc repro` on a big-enough machine still
    # evaluates all eight models. Set on a box too small for M3b_ecod's
    # whole-split `decision_function` (see `build_model_catalog`'s docstring)
    # to get a complete run for the other seven instead of losing the run.
    exclude_models = {
        name.strip()
        for name in os.environ.get("AUTHBENCH_EXCLUDE_MODELS", "").split(",")
        if name.strip()
    }
    if exclude_models:
        logger.warning("Excluding from this run (AUTHBENCH_EXCLUDE_MODELS): %s", sorted(exclude_models))
    models = build_model_catalog(
        fit_sample_size=int(cfg.runtime.fit_sample_size) or None, exclude=exclude_models
    )
    model_names: list[str] = [m.name for m in models]  # type: ignore[attr-defined]
    scored, skipped = fit_and_score_all(models, train, val, test)
    if skipped:
        model_names = [n for n in model_names if n not in skipped]
        logger.error(
            "%d of %d models were not evaluated: %s",
            len(skipped),
            len(models),
            "; ".join(f"{k} ({v})" for k, v in skipped.items()),
        )

    logger.info(
        "Campaign-stratified bootstrap: %d resamples x %d models over %d test events",
        cfg.eval.bootstrap.n_resamples,
        len(models),
        scored.height,
    )
    bootstrap = paired_campaign_bootstrap(
        scored,
        lambda frame, col: auc_pr(frame["is_malicious"].to_numpy(), frame[col].to_numpy()),
        {name: score_column(name) for name in model_names},
        n_resamples=int(cfg.eval.bootstrap.n_resamples),
        confidence=float(cfg.eval.bootstrap.confidence),
        seed=int(cfg.eval.bootstrap.seed),
        # Same estimator, computed from a pre-sorted rank order instead of by
        # re-gathering and re-sorting the split once per resample per model.
        # `tests/unit/test_fast_auc_pr.py` pins the two to agree.
        fast_auc_pr=True,
        n_jobs=int(cfg.runtime.bootstrap_workers),
    )

    summary_rows = []
    curves = []
    for name in model_names:
        evaluation = evaluate_model(
            scored,
            score_column(name),
            name,
            budgets=budgets,
            fpr_targets=fpr_targets,
            report_roc_auc=report_roc_auc,
            # The caveat is logged once above, from the config's own text —
            # repeating it per model would bury it.
            warn_roc_auc=False,
            auc_pr_ci=bootstrap.ci(name),
        )
        curves.append(evaluation.curve)
        summary_rows.append(evaluation.to_dict())

        with tracker.run(name):
            tracker.log_param("feature_store_version", version)
            tracker.log_param("seed", cfg.seed)
            tracker.log_metric("auc_pr", evaluation.auc_pr)
            assert evaluation.auc_pr_ci is not None
            tracker.log_metric("auc_pr_ci_low", evaluation.auc_pr_ci.ci_low)
            tracker.log_metric("auc_pr_ci_high", evaluation.auc_pr_ci.ci_high)
            if evaluation.roc_auc is not None:
                tracker.log_metric("roc_auc", evaluation.roc_auc)
            for k, recall in zip(
                evaluation.curve.budgets, evaluation.curve.campaign_recall, strict=True
            ):
                tracker.log_metric(f"campaign_recall_at_{k}", recall)
            for ttd in evaluation.time_to_detection:
                tracker.log_metric(
                    f"campaigns_never_detected_at_{ttd.budget}", ttd.n_never_detected
                )
                if ttd.median_delay_seconds is not None:
                    tracker.log_metric(
                        f"median_ttd_seconds_at_{ttd.budget}", ttd.median_delay_seconds
                    )

        logger.info(
            "%s: AUC-PR=%.4f [%.4f, %.4f]",
            name,
            evaluation.auc_pr,
            bootstrap.ci(name).ci_low,
            bootstrap.ci(name).ci_high,
        )

    # Beside reports/tables/, not inside it: DVC will not accept a metric
    # nested in a tracked output directory.
    Path(cfg.paths.reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.paths.reports_dir) / "metrics_summary.json").write_text(
        json.dumps(summary_rows, indent=2)
    )

    comparisons = pairwise_comparisons(
        scored,
        model_names,
        bootstrap,
        method=str(cfg.eval.pairwise_test.method),
        alpha=float(cfg.eval.pairwise_test.alpha),
        n_permutations=int(cfg.eval.pairwise_test.n_permutations),
        seed=int(cfg.eval.pairwise_test.seed),
    )
    # Published, not just logged: a reader must be able to see which
    # competitors are missing from the comparison and why.
    (tables_dir / "skipped_models.json").write_text(json.dumps(skipped, indent=2))

    (tables_dir / "pairwise_comparisons.json").write_text(
        json.dumps([c.to_dict() for c in comparisons], indent=2)
    )
    n_significant = sum(1 for c in comparisons if c.significant)
    logger.info(
        "%d/%d pairwise AUC-PR differences significant at alpha=%s after %s correction.",
        n_significant,
        len(comparisons),
        cfg.eval.pairwise_test.alpha,
        cfg.eval.pairwise_test.correction,
    )

    n_campaigns = scored.filter(pl.col("campaign_id").is_not_null())["campaign_id"].n_unique()
    figure_path = plot_campaign_recall_vs_budget(
        curves,
        figures_dir / CAMPAIGN_RECALL_FIGURE,
        subtitle=(
            f"{cfg.dataset.name} test split — {n_campaigns} campaigns, {scored.height:,} events."
        ),
    )
    logger.info("Wrote headline figure to %s", figure_path)


if __name__ == "__main__":
    main()
