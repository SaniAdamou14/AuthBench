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
from pathlib import Path

import hydra
import mlflow
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
from authbench.models.classical import ECODScorer, HBOSScorer, IsolationForestScorer
from authbench.models.floors import AlwaysFailScorer, RandomScorer
from authbench.models.rules import RulesScorer
from authbench.models.stats import PairRarityScorer, PCAReconstructionScorer
from authbench.pipeline.build_features import CONF_DIR, feature_store_version

logger = logging.getLogger(__name__)

NUMERIC_FEATURE_COLUMNS = [
    "is_success",
    "auth_type_is_null",
    "logon_type_is_null",
    "src_user_is_machine",
    "src_dst_user_same",
    "src_dst_computer_same",
    "domain_crossing",
    "pair_is_new",
    "pair_global_rarity",
    "user_new_host_count_24h",
    "host_new_user_count_24h",
    "hour_sin",
    "hour_cos",
    "hour_deviation_from_profile",
]


def build_model_catalog() -> list[object]:
    """The catalog, *unfitted*. `main` is the single place that fits — an
    earlier version fitted M1 here and then refit every model in the loop,
    which at LANL scale meant paying for M1's per-user aggregation twice and
    left the calibrated weights one stray `fit()` away from being discarded.
    """
    return [
        RandomScorer(),
        AlwaysFailScorer(),
        PairRarityScorer(),
        PCAReconstructionScorer(NUMERIC_FEATURE_COLUMNS),
        IsolationForestScorer(NUMERIC_FEATURE_COLUMNS),
        ECODScorer(NUMERIC_FEATURE_COLUMNS),
        HBOSScorer(NUMERIC_FEATURE_COLUMNS),
        RulesScorer(),
    ]


def fit_and_score_all(
    models: list[object], train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame
) -> pl.DataFrame:
    """Fit every model and return the slim evaluation frame carrying one
    score column per model.

    Holding all scores on a single frame is what lets the bootstrap below
    resample once for every model instead of once per model.
    """
    scored = test.select(EVAL_COLUMNS)
    for model in models:
        name: str = model.name  # type: ignore[attr-defined]
        logger.info("Fitting %s", name)
        model.fit(train.lazy())  # type: ignore[attr-defined]
        # M1 alone has a second, label-aware fitting step. It runs after
        # `fit` (which re-derives the thresholds its rules are built on)
        # and against the validation split only — never train, never test.
        if isinstance(model, RulesScorer):
            model.calibrate_weights(val.lazy(), val["is_malicious"], n_trials=200)

        scores = model.score(test.lazy())  # type: ignore[attr-defined]
        scored = scored.with_columns(pl.Series(score_column(name), scores))

    return scored


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

    train = pl.read_parquet(version_dir / "train.parquet")
    val = pl.read_parquet(version_dir / "val.parquet")
    test = pl.read_parquet(version_dir / "test.parquet")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    tables_dir = Path(cfg.paths.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    budgets = list(cfg.eval.budgets)
    fpr_targets = list(cfg.eval.fpr_targets)
    report_roc_auc = bool(cfg.eval.roc_auc.report)
    if report_roc_auc and bool(cfg.eval.roc_auc.warn):
        logger.warning("%s", str(cfg.eval.roc_auc.warning_message).strip() or ROC_AUC_WARNING)

    models = build_model_catalog()
    model_names: list[str] = [m.name for m in models]  # type: ignore[attr-defined]
    scored = fit_and_score_all(models, train, val, test)

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

        with mlflow.start_run(run_name=name):
            mlflow.log_param("feature_store_version", version)
            mlflow.log_param("seed", cfg.seed)
            mlflow.log_metric("auc_pr", evaluation.auc_pr)
            assert evaluation.auc_pr_ci is not None
            mlflow.log_metric("auc_pr_ci_low", evaluation.auc_pr_ci.ci_low)
            mlflow.log_metric("auc_pr_ci_high", evaluation.auc_pr_ci.ci_high)
            if evaluation.roc_auc is not None:
                mlflow.log_metric("roc_auc", evaluation.roc_auc)
            for k, recall in zip(
                evaluation.curve.budgets, evaluation.curve.campaign_recall, strict=True
            ):
                mlflow.log_metric(f"campaign_recall_at_{k}", recall)
            for ttd in evaluation.time_to_detection:
                mlflow.log_metric(f"campaigns_never_detected_at_{ttd.budget}", ttd.n_never_detected)
                if ttd.median_delay_seconds is not None:
                    mlflow.log_metric(
                        f"median_ttd_seconds_at_{ttd.budget}", ttd.median_delay_seconds
                    )

        logger.info(
            "%s: AUC-PR=%.4f [%.4f, %.4f]",
            name,
            evaluation.auc_pr,
            bootstrap.ci(name).ci_low,
            bootstrap.ci(name).ci_high,
        )

    (tables_dir / "metrics_summary.json").write_text(json.dumps(summary_rows, indent=2))

    comparisons = pairwise_comparisons(
        scored,
        model_names,
        bootstrap,
        method=str(cfg.eval.pairwise_test.method),
        alpha=float(cfg.eval.pairwise_test.alpha),
        n_permutations=int(cfg.eval.pairwise_test.n_permutations),
        seed=int(cfg.eval.pairwise_test.seed),
    )
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

    n_campaigns = test.filter(pl.col("campaign_id").is_not_null())["campaign_id"].n_unique()
    figure_path = plot_campaign_recall_vs_budget(
        curves,
        figures_dir / CAMPAIGN_RECALL_FIGURE,
        subtitle=(
            f"{cfg.dataset.name} test split — {n_campaigns} campaigns, {test.height:,} events."
        ),
    )
    logger.info("Wrote headline figure to %s", figure_path)


if __name__ == "__main__":
    main()
