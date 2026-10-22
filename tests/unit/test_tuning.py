"""Every tunable model gets the same budget, its default among the candidates, chosen on validation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from tests.unit.test_row_locality import _featurized

from authbench.features import MODEL_FEATURE_COLUMNS
from authbench.models.classical import HBOSScorer, IsolationForestScorer
from authbench.models.rules import RulesScorer, UniformRulesScorer
from authbench.models.stats import PairRarityScorer, PCAReconstructionScorer
from authbench.pipeline.tuning import GRIDS, is_tunable, tune_and_fit


def _catalog() -> list:  # type: ignore[type-arg]
    return [
        RulesScorer(),
        UniformRulesScorer(),
        PCAReconstructionScorer(MODEL_FEATURE_COLUMNS),
        IsolationForestScorer(MODEL_FEATURE_COLUMNS),
        HBOSScorer(MODEL_FEATURE_COLUMNS),
        PairRarityScorer(),
    ]


def test_every_tunable_model_gets_the_same_budget_with_its_default_first() -> None:
    frame = _featurized()
    fitted, records = tune_and_fit(_catalog(), frame.lazy(), frame.lazy(), n_configs=5, seed=1)
    assert [r.model for r in records] == [
        "M1_rules",
        "M2b_pca_reconstruction",
        "M3a_iforest",
        "M3b_hbos",
    ]
    assert all(len(r.configs) == 5 for r in records)
    assert records[0].configs[0]["weights"] == pytest.approx(
        {k: 1 / 8 for k in records[0].configs[0]["weights"]}
    )
    for record in records[1:]:
        assert record.configs[0] == GRIDS[record.model][0]
    # The chosen candidate is the best on validation, and is the model returned.
    for record in records:
        assert record.validation_auc_pr[record.chosen] == max(record.validation_auc_pr)
    assert [type(m) for m in fitted] == [type(m) for m in _catalog()]


def test_without_validation_nothing_is_tuned_and_m1_keeps_uniform_weights() -> None:
    frame = _featurized()
    fitted, records = tune_and_fit(_catalog(), frame.lazy(), None, n_configs=5, seed=1)
    assert records == []
    rules = next(m for m in fitted if type(m) is RulesScorer)
    assert np.allclose(list(rules.weights.values()), 1 / 8)


def test_the_label_free_rules_are_never_tuned() -> None:
    assert not is_tunable(UniformRulesScorer())
    assert is_tunable(RulesScorer())
    assert not is_tunable(PairRarityScorer())


def test_candidates_do_not_depend_on_the_catalog_order() -> None:
    frame = _featurized()
    _, forward = tune_and_fit(_catalog(), frame.lazy(), frame.lazy(), n_configs=4, seed=3)
    _, backward = tune_and_fit(_catalog()[::-1], frame.lazy(), frame.lazy(), n_configs=4, seed=3)
    by_name = {r.model: r.configs for r in backward}
    for record in forward:
        assert record.configs == by_name[record.model]


def test_a_tuned_hbos_scores_with_its_chosen_bins() -> None:
    frame = _featurized()
    fitted, records = tune_and_fit(
        [HBOSScorer(MODEL_FEATURE_COLUMNS)], frame.lazy(), frame.lazy(), n_configs=12, seed=0
    )
    chosen = records[0].configs[records[0].chosen]
    assert (fitted[0].n_bins, fitted[0].tol) == (chosen["n_bins"], chosen["tol"])
    assert fitted[0].score(frame.lazy()).len() == frame.height
    assert pl.Series(records[0].validation_auc_pr).is_finite().all()
