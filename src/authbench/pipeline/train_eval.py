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
from authbench.evaluate.statistics import POSITIVE_KEYS, ModelStatistics, SplitAccumulator
from authbench.evaluate.stats_tests import paired_campaign_bootstrap
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
from authbench.models.rules import RulesScorer
from authbench.models.stats import (
    PairRarityScorer,
    PCAReconstructionScorer,
    SourceRarityScorer,
)
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
    ]
    if deep_models:
        catalog.extend(_deep_catalog(fit_sample_size))
    if exclude:
        catalog = [m for m in catalog if m.name not in exclude]
    return catalog


def _deep_catalog(fit_sample_size: int | None) -> list[AnomalyScorer]:
    """M4a and M4b, configured from `conf/model/ae.yaml` and `conf/model/vae.yaml`."""
    from authbench.models.deep import AutoencoderScorer, VAEScorer

    def params(name: str) -> dict[str, Any]:
        cfg = OmegaConf.to_container(OmegaConf.load(CONF_DIR / "model" / f"{name}.yaml"))
        assert isinstance(cfg, dict)
        return {str(k): v for k, v in cfg.items() if k not in _MODEL_CONFIG_METADATA}

    return [
        AutoencoderScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size, **params("ae")),
        VAEScorer(MODEL_FEATURE_COLUMNS, fit_sample_size=fit_sample_size, **params("vae")),
    ]


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


def fit_models(models: list[AnomalyScorer], train: pl.LazyFrame, val: pl.LazyFrame | None) -> None:
    """Fit every model on train; M1 then calibrates its rule weights on validation labels.

    Without a validation partition M1 keeps uniform weights and no label is read.
    """
    for model in models:
        logger.info("Fitting %s", model.name)
        model.fit(train)
        if isinstance(model, RulesScorer) and val is not None:
            val_frame = val.collect()
            model.calibrate_weights(val_frame.lazy(), val_frame["is_malicious"], n_trials=200)
            del val_frame


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
) -> tuple[dict[str, ModelStatistics], dict[str, str]]:
    """Score `test` one day at a time into each model's statistics; return them and the skipped models.

    The positives are scored first, alone, which gives every threshold the benign events
    of each day are counted against; each day then checks that it reproduces those
    scores bit for bit. A model that runs out of memory is excluded with its reason,
    and the report says so: a benchmark that silently drops a competitor is worse
    than one that loses a run.
    """
    not_row_local = [m.name for m in models if not getattr(m, "scores_row_locally", False)]
    if not_row_local:
        raise ValueError(f"{not_row_local} are not row-local and cannot be scored by day.")
    skipped: dict[str, str] = {}
    positives = test.filter(pl.col("is_malicious")).collect()
    positive_scores = positives.select(POSITIVE_KEYS)
    for model in models:
        try:
            positive_column = _score(model, positives, chunk_rows)
        except MemoryError as exc:
            skipped[model.name] = f"{type(exc).__name__}: {exc}"
            continue
        positive_scores = positive_scores.with_columns(
            pl.Series(score_column(model.name), positive_column)
        )
    active = [m for m in models if m.name not in skipped]
    accumulator = SplitAccumulator(
        positive_scores,
        {m.name: score_column(m.name) for m in active},
        budgets=budgets,
        top_k=top_k,
    )
    days = test.select(pl.col("day").unique().sort()).collect().to_series().to_list()
    for day in days:
        frame = test.filter(pl.col("day") == day).collect()
        scores: dict[str, np.ndarray] = {}
        for model in list(active):
            try:
                scores[model.name] = _score(model, frame, chunk_rows)
            except MemoryError as exc:
                skipped[model.name] = f"{type(exc).__name__}: {exc}"
                logger.error("%s failed on day %s and is EXCLUDED: %s", model.name, day, exc)
                accumulator.drop(model.name)
                active.remove(model)
        accumulator.add_day(frame.select(EVAL_COLUMNS), scores)
        logger.info("Scored day %s: %d events, %d models.", day, frame.height, len(active))
        del frame
    if not active:
        raise RuntimeError("Every model failed to score; there is nothing to report.")
    return accumulator.finish(), skipped


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
            tables_dir=Path(cfg.paths.tables_dir) / suffix,
            figures_dir=Path(cfg.paths.figures_dir) / suffix,
            metrics_path=Path(cfg.paths.reports_dir) / suffix / "metrics_summary.json",
            block_of=dict(users.iter_rows()) if block == "account" else None,
            exclude_models=exclude_models,
            tracker=tracker,
            version=version,
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
    )
    fit_models(models, train, val)
    statistics, skipped = score_split(models, test, budgets=budgets, top_k=max(budgets))
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
