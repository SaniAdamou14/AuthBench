"""Equal hyperparameter budgets, chosen on validation by one criterion for every model.

The published runs tuned one model: M1's rule weights, by 200 draws scored on validation
labels, while every other model kept its library defaults. That asymmetry favoured M1.
Here every model with something to choose gets the same budget of `n_configs`
candidates, always including its default, and keeps the one with the highest AUC-PR on
validation: M1 its rule weights, PCA its retained variance, Isolation Forest its forest
size and sample, HBOS its bins and tolerance. Validation is scored by day, like the test
split, so tuning costs no more memory than evaluation.

Without a validation partition nothing is tuned and no label is read. Models with no
hyperparameter (M0, M2a, M2c, ECOD, M1u) are fitted once; the autoencoders are not tuned,
and the report says so.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from authbench.evaluate.statistics import average_precision
from authbench.models.base import AnomalyScorer
from authbench.models.classical import ECODScorer, HBOSScorer, IsolationForestScorer
from authbench.models.rules import RULES, NoPositivesInValidationError, RulesScorer
from authbench.models.stats import PCAReconstructionScorer

logger = logging.getLogger(__name__)

# The default first, so tuning cannot do worse on validation than not tuning.
GRIDS: dict[str, list[dict[str, Any]]] = {
    "M2b_pca_reconstruction": [{"n_components": v} for v in (0.95, 0.5, 0.7, 0.8, 0.9, 0.99)],
    "M3a_iforest": [{"n_estimators": 200, "max_samples": 256}]
    + [
        {"n_estimators": n, "max_samples": m}
        for n in (100, 200, 400)
        for m in (64, 256, 1024)
        if (n, m) != (200, 256)
    ],
    "M3b_hbos": [{"n_bins": 10, "tol": 0.5}]
    + [
        {"n_bins": b, "tol": t}
        for b in (5, 10, 20, 50)
        for t in (0.1, 0.5, 0.9)
        if (b, t) != (10, 0.5)
    ],
}


@dataclass(frozen=True)
class TuningRecord:
    model: str
    configs: list[dict[str, Any]]
    validation_auc_pr: list[float]
    chosen: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "chosen": self.configs[self.chosen],
            "candidates": [
                {"config": c, "validation_auc_pr": ap}
                for c, ap in zip(self.configs, self.validation_auc_pr, strict=True)
            ],
        }


def _variant(
    model: AnomalyScorer, config: dict[str, Any], seed: int | None = None
) -> AnomalyScorer:
    seed = model.seed if seed is None else seed  # type: ignore[attr-defined]
    if isinstance(model, PCAReconstructionScorer):
        return PCAReconstructionScorer(
            model.feature_cols,
            n_components=float(config["n_components"]),
            seed=seed,
            fit_sample_size=model.fit_sample_size,
        )
    if isinstance(model, IsolationForestScorer):
        return IsolationForestScorer(
            model.feature_cols,
            n_estimators=int(config["n_estimators"]),
            max_samples=int(config["max_samples"]),
            seed=seed,
            fit_sample_size=model.fit_sample_size,
        )
    if isinstance(model, HBOSScorer):
        return HBOSScorer(
            model.feature_cols,
            fit_sample_size=model.fit_sample_size,
            n_bins=int(config["n_bins"]),
            tol=float(config["tol"]),
            seed=seed,
        )
    raise TypeError(f"No grid for {model.name}.")


def _config_of(model: AnomalyScorer) -> dict[str, Any]:
    if isinstance(model, PCAReconstructionScorer):
        return {"n_components": model.n_components}
    if isinstance(model, IsolationForestScorer):
        return {"n_estimators": model.n_estimators, "max_samples": model.max_samples}
    if isinstance(model, HBOSScorer):
        return {"n_bins": model.n_bins, "tol": model.tol}
    raise TypeError(f"No hyperparameters for {model.name}.")


#: The models whose fit draws on a seed: a fitting sample, a forest, rule normalisation.
SEEDED = (
    "M1_rules",
    "M2b_pca_reconstruction",
    "M3a_iforest",
    "M3b_ecod",
    "M3b_hbos",
)


def reseed(model: AnomalyScorer, seed: int) -> AnomalyScorer:
    """An unfitted copy of a tuned model with the same configuration and another seed."""
    if isinstance(model, RulesScorer):
        copy = type(model)(
            chain_window_seconds=model.chain_window_seconds,
            percentile_threshold=model.percentile_threshold,
            normalization_days=model.normalization_days,
            normalization_sample_size=model.normalization_sample_size,
            seed=seed,
        )
        copy.weights = dict(model.weights)
        return copy
    if isinstance(model, ECODScorer):
        return ECODScorer(model.feature_cols, fit_sample_size=model.fit_sample_size, seed=seed)
    return _variant(model, _config_of(model), seed)


def _candidates(
    model: AnomalyScorer, train: pl.LazyFrame, n_configs: int, rng: np.random.Generator
) -> tuple[list[AnomalyScorer], list[dict[str, Any]]]:
    """Fitted candidates and their configurations, the default first."""
    if isinstance(model, RulesScorer):
        model.fit(train)
        uniform = np.full(len(RULES), 1.0 / len(RULES))
        draws = [uniform] + [rng.dirichlet(np.ones(len(RULES))) for _ in range(n_configs - 1)]
        configs: list[dict[str, Any]] = [
            {"weights": dict(zip([r.id for r in RULES], w.tolist(), strict=True))} for w in draws
        ]
        return [model.with_weights(w) for w in draws], configs
    grid = GRIDS[model.name]
    if len(grid) > n_configs:
        picked = rng.choice(len(grid) - 1, size=n_configs - 1, replace=False) + 1
        grid = [grid[0]] + [grid[i] for i in sorted(picked)]
    variants = [_variant(model, config) for config in grid]
    for variant in variants:
        variant.fit(train)
    return variants, grid


def is_tunable(model: AnomalyScorer) -> bool:
    if isinstance(model, RulesScorer):
        return bool(model.requires_labels)
    return model.name in GRIDS


def tune_and_fit(
    models: list[AnomalyScorer],
    train: pl.LazyFrame,
    val: pl.LazyFrame | None,
    *,
    n_configs: int,
    seed: int,
) -> tuple[list[AnomalyScorer], list[TuningRecord]]:
    """Fit every model; with validation, keep each tunable model's best of `n_configs` candidates."""
    from authbench.pipeline.train_eval import score_split

    if val is not None and int(val.select(pl.col("is_malicious").sum()).collect().item()) == 0:
        raise NoPositivesInValidationError(
            "Tuning needs malicious events in the validation period: AUC-PR is identically 0 "
            "there, and every model would keep its first candidate while looking tuned. Check "
            "the validation window of the split against the red-team timestamps."
        )
    fitted: list[AnomalyScorer] = []
    records: list[TuningRecord] = []
    for model in models:
        if val is None or not is_tunable(model):
            model.fit(train)
            fitted.append(model)
            continue
        # Seeded by name, so a model's candidates do not depend on the catalog's order.
        rng = np.random.default_rng([seed, zlib.crc32(model.name.encode())])
        variants, configs = _candidates(model, train, n_configs, rng)
        labels = [f"{model.name}#{i}" for i in range(len(variants))]
        stats, skipped = score_split(variants, val, budgets=[1], top_k=1, names=labels)
        scores = [
            average_precision(stats[label]) if label not in skipped else float("-inf")
            for label in labels
        ]
        best = int(np.argmax(scores))
        logger.info(
            "Tuned %s on validation: candidate %d of %d, AUC-PR %.4f (default %.4f).",
            model.name,
            best,
            len(variants),
            scores[best],
            scores[0],
        )
        fitted.append(variants[best])
        records.append(TuningRecord(model.name, configs, scores, best))
    return fitted, records


def fit_variance(
    models: list[AnomalyScorer],
    train: pl.LazyFrame,
    test: pl.LazyFrame,
    *,
    seeds: list[int],
    budgets: list[int],
) -> dict[str, dict[str, Any]]:
    """Each seeded model refitted with every seed, its configuration fixed, and scored on test.

    The spread is the variance the fit itself adds, which no bootstrap over the test split sees:
    a forest grown again, another fitting sample, another normalisation sample for M1.
    """
    from authbench.evaluate.statistics import budget_curve
    from authbench.pipeline.train_eval import score_split

    seeded = [m for m in models if m.name in SEEDED]
    variants, labels = [], []
    for model in seeded:
        for seed in seeds:
            variant = reseed(model, seed)
            variant.fit(train)
            variants.append(variant)
            labels.append(f"{model.name}@{seed}")
    if not variants:
        return {}
    stats, skipped = score_split(variants, test, budgets=budgets, top_k=max(budgets), names=labels)
    result: dict[str, dict[str, Any]] = {}
    for model in seeded:
        runs = []
        for seed in seeds:
            label = f"{model.name}@{seed}"
            if label in skipped:
                continue
            curve = budget_curve(stats[label], budgets)
            runs.append(
                {
                    "seed": seed,
                    "auc_pr": average_precision(stats[label]),
                    "budget_for_first_detection": curve.budget_for_first_detection,
                    "campaign_recall": dict(
                        zip(map(str, budgets), curve.campaign_recall, strict=True)
                    ),
                }
            )
        values = np.array([r["auc_pr"] for r in runs])
        firsts = np.array([r["budget_for_first_detection"] or 0 for r in runs], dtype=float)
        result[model.name] = {
            "runs": runs,
            "auc_pr": {
                "min": float(values.min()),
                "max": float(values.max()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            },
            "budget_for_first_detection": {"min": float(firsts.min()), "max": float(firsts.max())},
        }
    return result
