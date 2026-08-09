"""DVC `train_eval` stage: feature store -> fitted models -> evaluation tables/figures.

Trains the dependency-light catalog (M0 floors, M1 rules, M2 stats, M3
classical) end to end and writes the comparison tables. Deep (M4) and graph
(M5) models are heavier, optional-extra runs — invoke them individually via
`authbench train model=ae` (Hydra model group) once `authbench[deep]` /
`authbench[graph]` are installed, rather than bundling them into the default
stage that every contributor's machine must be able to run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import mlflow
import polars as pl
from omegaconf import DictConfig

from authbench.evaluate.budget import compute_budget_curve
from authbench.evaluate.metrics import auc_pr
from authbench.evaluate.plots import CAMPAIGN_RECALL_FIGURE, plot_campaign_recall_vs_budget
from authbench.evaluate.stats_tests import bootstrap_ci
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


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    processed_dir = Path(cfg.paths.processed_dir) / "features"
    version = feature_store_version(cfg.features)
    version_dir = processed_dir / version

    train = pl.read_parquet(version_dir / "train.parquet")
    val = pl.read_parquet(version_dir / "val.parquet")
    test = pl.read_parquet(version_dir / "test.parquet")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    models = build_model_catalog()
    y_test = test["is_malicious"].to_numpy()

    tables_dir = Path(cfg.paths.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    curves = []
    for model in models:
        with mlflow.start_run(run_name=model.name):  # type: ignore[attr-defined]
            mlflow.log_param("feature_store_version", version)
            mlflow.log_param("seed", cfg.seed)

            model.fit(train.lazy())  # type: ignore[attr-defined]
            # M1 alone has a second, label-aware fitting step. It runs after
            # `fit` (which re-derives the thresholds its rules are built on)
            # and against the validation split only — never train, never test.
            if isinstance(model, RulesScorer):
                model.calibrate_weights(val.lazy(), val["is_malicious"], n_trials=200)

            scores = model.score(test.lazy())  # type: ignore[attr-defined]

            ap = auc_pr(y_test, scores.to_numpy())
            scored = test.with_columns(pl.Series("_score", scores))
            curve = compute_budget_curve(
                scored,
                "_score",
                model.name,  # type: ignore[attr-defined]
                budgets=list(cfg.eval.budgets),
            )
            curves.append(curve)

            ci = bootstrap_ci(
                scored,
                lambda f: auc_pr(f["is_malicious"].to_numpy(), f["_score"].to_numpy()),
                n_resamples=cfg.eval.bootstrap.n_resamples,
                confidence=cfg.eval.bootstrap.confidence,
                seed=cfg.eval.bootstrap.seed,
            )

            mlflow.log_metric("auc_pr", ap)
            for k, recall in zip(curve.budgets, curve.campaign_recall, strict=True):
                mlflow.log_metric(f"campaign_recall_at_{k}", recall)

            summary_rows.append(
                {
                    "model": model.name,  # type: ignore[attr-defined]
                    "auc_pr": ap,
                    "auc_pr_ci_low": ci.ci_low,
                    "auc_pr_ci_high": ci.ci_high,
                    "budgets": curve.budgets,
                    "campaign_recall": curve.campaign_recall,
                    "event_recall": curve.event_recall,
                }
            )
            logger.info("%s: AUC-PR=%.4f [%.4f, %.4f]", model.name, ap, ci.ci_low, ci.ci_high)  # type: ignore[attr-defined]

    (tables_dir / "metrics_summary.json").write_text(json.dumps(summary_rows, indent=2))

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
