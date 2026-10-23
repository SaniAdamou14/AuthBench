"""DVC `train_eval` stage: feature store -> fitted models -> evaluation tables/figures.

Trains the dependency-light catalog (M0 floors, M1 rules, M2 stats, M3
classical) end to end and writes the comparison tables. The autoencoders
(M4a, M4b) join the same catalog, scored under the same protocol, when
`runtime.deep_models` is true and the `deep` extra is installed; they are off
by default so the stage still runs on a machine without PyTorch. M4c and M5
need the F6/F5 feature families, which the feature store does not carry yet.

The test split is scored one day at a time and each day is folded into the
sufficient statistics of `evaluate.statistics`, so memory is bounded by the
largest day whatever the length of the split. The positives are scored first,
on their own; every model is row-local to the last bit, and each day checks it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf

from authbench.evaluate.ledger import campaign_users, write_ledger_outputs
from authbench.evaluate.metrics import ROC_AUC_WARNING
from authbench.evaluate.plots import CAMPAIGN_RECALL_FIGURE, plot_campaign_recall_vs_budget
from authbench.evaluate.staffing import staffing_table
from authbench.evaluate.statistics import (
    POSITIVE_KEYS,
    ModelStatistics,
    SplitAccumulator,
    alert_thresholds,
)
from authbench.evaluate.stats_tests import (
    cluster_bootstrap,
    cluster_bootstrap_record,
    paired_campaign_bootstrap,
)
from authbench.evaluate.summary import (
    EVAL_COLUMNS,
    evaluate_model,
    pairwise_comparisons,
    score_column,
)
from authbench.features import MODEL_FEATURE_COLUMNS
from authbench.models.base import DEFAULT_FIT_SAMPLE_SIZE, AnomalyScorer
from authbench.models.classical import ECODScorer, HBOSScorer, IsolationForestScorer
from authbench.models.floors import AlwaysFailScorer, RandomScorer
from authbench.models.rules import RulesScorer, UniformRulesScorer
from authbench.models.stats import (
    PairRarityScorer,
    PCAReconstructionScorer,
    SourceRarityScorer,
)
from authbench.pipeline.build_features import CONF_DIR, feature_store_version
from authbench.pipeline.tracking import open_tracker
from authbench.pipeline.tuning import TuningRecord, fit_variance, tune_and_fit
from authbench.split.temporal import verify_temporal_order

logger = logging.getLogger(__name__)

#: Rows scored at once by the row-local models.
#:
#: Peak memory for the design matrix is this times 21 float64 columns times the
#: estimators' copy factor — about 1 GB at two million rows, and independent of
#: how large the split is. Lower it on a smaller machine; it cannot change any
#: score, only how many are computed at a time.
DEFAULT_SCORING_CHUNK_ROWS = 2_000_000

# Descriptive keys of `conf/model/*.yaml` that are not constructor arguments.
_MODEL_CONFIG_METADATA = frozenset({"name", "id", "family", "requires_labels"})

# The design matrix is `features.MODEL_FEATURE_COLUMNS`, shared with
# `authbench demo`. It used to be redeclared here, and the two copies had
# drifted: this stage was fitting M2b/M3a/M3b on 14 columns while the demo
# used 21, silently dropping every F2 history feature and all three F1
# frequency encodings — i.e. benchmarking different models under the same
# names in the two places that are supposed to agree.


def build_model_catalog(
    fit_sample_size: int | None = DEFAULT_FIT_SAMPLE_SIZE,
    *,
    exclude: set[str] | None = None,
    deep_models: bool = False,
    euler: bool = False,
    cache_dir: str | None = None,
) -> list[AnomalyScorer]:
    """The catalog, *unfitted*. `main` is the single place that fits — an
    earlier version fitted M1 here and then refit every model in the loop,
    which at LANL scale meant paying for M1's per-user aggregation twice and
    left the calibrated weights one stray `fit()` away from being discarded.

    `fit_sample_size` bounds the rows the vector-space models estimate their
    distributions from. It is a published parameter rather than a hidden
    constant, because it is a real methodological choice: `None` fits on
    everything and is what the tests and the demo use.

    `exclude` drops named models before fitting even starts, for a model known
    to be unaffordable on the machine at hand: `score_split` catches a graceful
    `MemoryError`, not a kernel OOM kill.
    """
    catalog: list[AnomalyScorer] = [
        RandomScorer(),
        AlwaysFailScorer(),
        PairRarityScorer(),
        SourceRarityScorer(),
        PCAReconstructionScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size),
        IsolationForestScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size),
        ECODScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size),
        HBOSScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size),
        RulesScorer(),
        UniformRulesScorer(),
    ]
    if deep_models:
        catalog.extend(_deep_catalog(fit_sample_size))
    if euler:
        from authbench.models.euler import EulerScorer

        catalog.append(EulerScorer(**_model_params("euler"), cache_dir=cache_dir))
    if exclude:
        catalog = [m for m in catalog if m.name not in exclude]
    return catalog


def _deep_catalog(fit_sample_size: int | None) -> list[AnomalyScorer]:
    """M4a and M4b, configured from `conf/model/ae.yaml` and `conf/model/vae.yaml`."""
    from authbench.models.deep import AutoencoderScorer, VAEScorer

    return [
        AutoencoderScorer(
            MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size, **_model_params("ae")
        ),
        VAEScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size, **_model_params("vae")),
    ]


def _model_params(name: str) -> dict[str, Any]:
    """Constructor arguments from `conf/model/<name>.yaml`."""
    cfg = OmegaConf.to_container(OmegaConf.load(CONF_DIR / "model" / f"{name}.yaml"))
    assert isinstance(cfg, dict)
    return {str(k): v for k, v in cfg.items() if k not in _MODEL_CONFIG_METADATA}


R1 = "r1_unsupervised"
R2 = "r2_semi_supervised"


def regime_train(train: pl.LazyFrame, regime: str) -> pl.LazyFrame:
    """The training rows of a regime (spec section 6.2).

    R1 keeps the red-team events inside the training window, as a defender who does not
    know they are compromised would; R2 removes them, as most published work does
    silently. Features are computed with them in either case: the history an event is
    compared against cannot be cleaned by a defender who has not found the attack.
    """
    if regime == R1:
        return train
    if regime == R2:
        return train.filter(~pl.col("is_malicious"))
    raise ValueError(f"Unknown regime {regime!r}; expected {R1!r} or {R2!r}.")


def fit_models(
    models: list[AnomalyScorer],
    train: pl.LazyFrame,
    val: pl.LazyFrame | None,
    *,
    n_configs: int = 20,
    seed: int = 42,
) -> tuple[list[AnomalyScorer], list[TuningRecord]]:
    """Fit every model on train, each tunable one choosing among `n_configs` candidates on validation.

    Without a validation partition nothing is tuned and no label is read (`pipeline.tuning`).
    """
    return tune_and_fit(models, train, val, n_configs=n_configs, seed=seed)


def _score(model: AnomalyScorer, frame: pl.DataFrame, chunk_rows: int) -> np.ndarray:
    parts = [
        model.score(frame.slice(offset, chunk_rows).lazy()).to_numpy()
        for offset in range(0, frame.height, chunk_rows)
    ]
    return np.concatenate(parts) if parts else np.empty(0)


def score_split(
    models: list[AnomalyScorer],
    test: pl.LazyFrame,
    *,
    budgets: list[int],
    top_k: int,
    chunk_rows: int = DEFAULT_SCORING_CHUNK_ROWS,
    names: list[str] | None = None,
    alert_thresholds: dict[str, np.ndarray] | None = None,
    cluster_column: str | None = None,
    with_units: bool = False,
) -> tuple[dict[str, ModelStatistics], dict[str, str]]:
    """Score `test` one day at a time into each model's statistics; return them and the skipped models.

    The positives are scored first, alone, which gives every threshold the benign events
    of each day are counted against; each day then checks that it reproduces those
    scores bit for bit. A model that runs out of memory is excluded with its reason,
    and the report says so: a benchmark that silently drops a competitor is worse
    than one that loses a run. `names` labels the models when several share a class,
    as tuning candidates do.
    """
    labels = names or [m.name for m in models]
    not_row_local = [
        label
        for label, m in zip(labels, models, strict=True)
        if not getattr(m, "scores_row_locally", False)
    ]
    if not_row_local:
        raise ValueError(f"{not_row_local} are not row-local and cannot be scored by day.")
    skipped: dict[str, str] = {}
    positives = test.filter(pl.col("is_malicious")).collect()
    positive_scores = positives.select(
        *POSITIVE_KEYS, *([cluster_column] if cluster_column is not None else [])
    )
    for label, model in zip(labels, models, strict=True):
        try:
            positive_column = _score(model, positives, chunk_rows)
        except MemoryError as exc:
            skipped[label] = f"{type(exc).__name__}: {exc}"
            continue
        positive_scores = positive_scores.with_columns(
            pl.Series(score_column(label), positive_column)
        )
    active = {label: m for label, m in zip(labels, models, strict=True) if label not in skipped}
    accumulator = SplitAccumulator(
        positive_scores,
        {label: score_column(label) for label in active},
        budgets=budgets,
        top_k=top_k,
        alert_thresholds=alert_thresholds,
        cluster_column=cluster_column,
        with_units=with_units,
    )
    days = test.select(pl.col("day").unique().sort()).collect().to_series().to_list()
    for day in days:
        frame = test.filter(pl.col("day") == day).collect()
        scores: dict[str, np.ndarray] = {}
        for label, model in list(active.items()):
            try:
                scores[label] = _score(model, frame, chunk_rows)
            except MemoryError as exc:
                skipped[label] = f"{type(exc).__name__}: {exc}"
                logger.error("%s failed on day %s and is EXCLUDED: %s", label, day, exc)
                accumulator.drop(label)
                del active[label]
        accumulator.add_day(frame.select(EVAL_COLUMNS), scores)
        logger.info("Scored day %s: %d events, %d models.", day, frame.height, len(active))
        del frame
    if not active:
        raise RuntimeError("Every model failed to score; there is nothing to report.")
    return accumulator.finish(), skipped


def calibrate_alert_thresholds(
    models: list[AnomalyScorer], calibration: pl.LazyFrame, budgets: list[int]
) -> dict[str, np.ndarray]:
    """Per model, the thresholds giving each budget's alerts a day on `calibration` (S6-25).

    Volume, not labels: the thresholds read no label, so the training partition serves when
    there is no validation one.
    """
    n_days = int(calibration.select(pl.col("day").n_unique()).collect().item())
    stats, _ = score_split(models, calibration, budgets=[1], top_k=max(budgets) * n_days)
    return {name: alert_thresholds(s, budgets) for name, s in stats.items()}


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    block = str(cfg.eval.bootstrap.block)
    if block not in ("account", "campaign"):
        raise ValueError(
            f"eval.bootstrap.block={block!r} is not supported; use 'account' or 'campaign'. "
            "Events within a campaign are strongly dependent, and resampling them "
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
    train, val, test = (
        pl.scan_parquet(version_dir / name / "*.parquet") if (version_dir / name).exists() else None
        for name in ("train", "val", "test")
    )
    if train is None or test is None:
        raise FileNotFoundError(
            f"No train or test day files under {version_dir}; run build_features."
        )
    # On the written partitions, not their configured ranges: every timestamp of a partition
    # precedes every timestamp of the next.
    verify_temporal_order(train, val, test)

    tracker = open_tracker(str(cfg.mlflow.tracking_uri), str(cfg.mlflow.experiment_name))
    report_roc_auc = bool(cfg.eval.roc_auc.report)
    if report_roc_auc and bool(cfg.eval.roc_auc.warn):
        logger.warning("%s", str(cfg.eval.roc_auc.warning_message).strip() or ROC_AUC_WARNING)

    # A machine-specific affordability call, not a methodological default.
    exclude_models = {
        name.strip()
        for name in os.environ.get("AUTHBENCH_EXCLUDE_MODELS", "").split(",")
        if name.strip()
    }
    if exclude_models:
        logger.warning(
            "Excluding from this run (AUTHBENCH_EXCLUDE_MODELS): %s", sorted(exclude_models)
        )
    users = campaign_users(
        test.filter(pl.col("is_malicious"))
        .select("campaign_id", "src_user", "is_malicious")
        .collect()
    )
    regimes = [str(r) for r in cfg.split.get("regimes", [R1])]
    for index, regime in enumerate(regimes):
        # The first regime is the reported one and keeps the paths DVC tracks; the others sit beside it.
        suffix = Path() if index == 0 else Path(regime)
        logger.info("Regime %s", regime)
        evaluate_regime(
            cfg,
            regime,
            regime_train(train, regime),
            val,
            test,
            users,
            # Temporal models observe every event up to the one they score, whatever they trained on.
            timeline=[f for f in (train, val, test) if f is not None],
            tables_dir=Path(cfg.paths.tables_dir) / suffix,
            figures_dir=Path(cfg.paths.figures_dir) / suffix,
            metrics_path=Path(cfg.paths.reports_dir) / suffix / "metrics_summary.json",
            block_of=dict(users.iter_rows()) if block == "account" else None,
            exclude_models=exclude_models,
            tracker=tracker,
            version=version,
            # The fit variance of the reported regime only: each seed refits and rescores.
            fit_variance_seeds=(
                [int(x) for x in cfg.eval.get("fit_variance", {}).get("seeds", [])]
                if index == 0
                else None
            ),
        )


def evaluate_regime(
    cfg: DictConfig,
    regime: str,
    train: pl.LazyFrame,
    val: pl.LazyFrame | None,
    test: pl.LazyFrame,
    users: pl.DataFrame,
    *,
    tables_dir: Path,
    figures_dir: Path,
    metrics_path: Path,
    block_of: dict[int, str] | None,
    exclude_models: set[str],
    tracker: Any,
    version: str,
    fit_variance_seeds: list[int] | None = None,
    timeline: list[pl.LazyFrame] | None = None,
) -> None:
    """Fit a fresh catalog on `train`, score `test` by day, and write every table and the figure."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    budgets = list(cfg.eval.budgets)
    models = build_model_catalog(
        fit_sample_size=int(cfg.runtime.fit_sample_size) or None,
        exclude=exclude_models,
        deep_models=bool(cfg.runtime.get("deep_models", False)),
        euler=bool(cfg.runtime.get("euler", False)),
        # Per split and regime: protocols may run side by side on one machine.
        cache_dir=str(Path(cfg.paths.processed_dir) / "euler" / str(cfg.split.name) / regime),
    )
    tuning_cfg = cfg.eval.get("tuning", {})
    models, tuning = fit_models(
        models,
        train,
        val,
        n_configs=int(tuning_cfg.get("n_configs", 20)),
        seed=int(tuning_cfg.get("seed", 42)),
    )
    (tables_dir / "tuning.json").write_text(json.dumps([r.to_dict() for r in tuning], indent=2))
    for model in models:
        prepare = getattr(model, "prepare", None)
        if prepare is not None:
            prepare(timeline or [train, test])
    thresholds = calibrate_alert_thresholds(models, val if val is not None else train, budgets)
    statistics, skipped = score_split(
        models,
        test,
        budgets=budgets,
        top_k=max(budgets),
        alert_thresholds=thresholds,
        cluster_column="src_user",
        with_units=True,
    )
    if fit_variance_seeds:
        variance = fit_variance(models, train, test, seeds=fit_variance_seeds, budgets=budgets)
        (tables_dir / "fit_variance.json").write_text(json.dumps(variance, indent=2))
    model_names = list(statistics)
    if skipped:
        logger.error(
            "%d of %d models were not evaluated: %s",
            len(skipped),
            len(models),
            "; ".join(f"{k} ({v})" for k, v in skipped.items()),
        )
    bootstrap = paired_campaign_bootstrap(
        statistics,
        n_resamples=int(cfg.eval.bootstrap.n_resamples),
        confidence=float(cfg.eval.bootstrap.confidence),
        seed=int(cfg.eval.bootstrap.seed),
        block_of=block_of,
    )

    by_user = cluster_bootstrap(
        statistics,
        n_resamples=int(cfg.eval.bootstrap.n_resamples),
        confidence=float(cfg.eval.bootstrap.confidence),
        seed=int(cfg.eval.bootstrap.seed),
    )
    (tables_dir / "bootstrap_user_clusters.json").write_text(
        json.dumps(
            cluster_bootstrap_record(by_user, alpha=float(cfg.eval.pairwise_test.alpha)), indent=2
        )
    )

    summary_rows = []
    curves = []
    for name in model_names:
        evaluation = evaluate_model(
            statistics[name],
            budgets=budgets,
            fpr_targets=list(cfg.eval.fpr_targets),
            report_roc_auc=bool(cfg.eval.roc_auc.report),
            # The caveat is logged once, from the config's own text; per model it would bury it.
            warn_roc_auc=False,
            auc_pr_ci=bootstrap.ci(name),
        )
        curves.append(evaluation.curve)
        summary_rows.append({"regime": regime, **evaluation.to_dict()})

        with tracker.run(f"{regime}/{name}"):
            tracker.log_param("feature_store_version", version)
            tracker.log_param("seed", cfg.seed)
            tracker.log_param("regime", regime)
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
            "%s %s: AUC-PR=%.4f [%.4f, %.4f]",
            regime,
            name,
            evaluation.auc_pr,
            evaluation.auc_pr_ci.ci_low,
            evaluation.auc_pr_ci.ci_high,
        )

    # Beside reports/tables/, not inside it: DVC will not accept a metric
    # nested in a tracked output directory.
    metrics_path.write_text(json.dumps(summary_rows, indent=2))

    comparisons = pairwise_comparisons(
        bootstrap,
        method=str(cfg.eval.pairwise_test.method),
        alpha=float(cfg.eval.pairwise_test.alpha),
    )
    # Published, not just logged: a reader must be able to see which
    # competitors are missing from the comparison and why.
    excluded = {
        name: "excluded before fitting (AUTHBENCH_EXCLUDE_MODELS)"
        for name in sorted(exclude_models)
    }
    (tables_dir / "skipped_models.json").write_text(json.dumps({**excluded, **skipped}, indent=2))
    (tables_dir / "pairwise_comparisons.json").write_text(
        json.dumps([c.to_dict() for c in comparisons], indent=2)
    )
    logger.info(
        "%s: %d/%d pairwise AUC-PR differences significant at alpha=%s after %s correction.",
        regime,
        sum(1 for c in comparisons if c.significant),
        len(comparisons),
        cfg.eval.pairwise_test.alpha,
        cfg.eval.pairwise_test.correction,
    )

    (tables_dir / "staffing.json").write_text(json.dumps(staffing_table(budgets), indent=2))
    ledger_cfg = cfg.eval.ledger
    write_ledger_outputs(
        statistics,
        users,
        tables_dir,
        k_max=int(ledger_cfg.k_max),
        n_resamples=int(ledger_cfg.n_resamples),
        n_simulations=int(ledger_cfg.n_simulations),
        confidence=float(ledger_cfg.confidence),
        seed=int(ledger_cfg.seed),
        gaps_hours=[float(g) for g in ledger_cfg.get("campaign_gap_hours", [])],
        budgets=budgets,
    )

    reference = statistics[model_names[0]]
    n_campaigns = reference.positives["campaign_id"].n_unique()
    figure_path = plot_campaign_recall_vs_budget(
        curves,
        figures_dir / CAMPAIGN_RECALL_FIGURE,
        subtitle=(
            f"{cfg.dataset.name} test split, {regime}: {n_campaigns} campaigns, "
            f"{reference.n_events:,} events."
        ),
    )
    logger.info("Wrote headline figure to %s", figure_path)


if __name__ == "__main__":
    main()
