"""M3 — classical ML anomaly detectors via PyOD (US-120, `authbench[classical]` extra)."""

from __future__ import annotations

import numpy as np
import polars as pl

from authbench.models.base import BaseAnomalyScorer


class _PyODScorer(BaseAnomalyScorer):
    """Shared plumbing for PyOD-backed models: a feature matrix in, a score out."""

    def __init__(self, feature_cols: list[str]) -> None:
        self.feature_cols = feature_cols
        self._model: object | None = None

    def _matrix(self, data: pl.LazyFrame) -> np.ndarray:
        return data.select(self.feature_cols).collect().to_numpy().astype(np.float64)

    def fit(self, train: pl.LazyFrame) -> None:
        assert self._model is not None, "subclass must set self._model before fit()"
        self._model.fit(self._matrix(train))  # type: ignore[attr-defined]

    def score(self, data: pl.LazyFrame) -> pl.Series:
        assert self._model is not None
        raw_scores = self._model.decision_function(self._matrix(data))  # type: ignore[attr-defined]
        return pl.Series(self.name, raw_scores)


class IsolationForestScorer(_PyODScorer):
    """M3a — Isolation Forest. The literature's reference classical baseline."""

    name = "M3a_iforest"
    requires_labels = False

    def __init__(
        self,
        feature_cols: list[str],
        n_estimators: int = 200,
        max_samples: int | str = 256,
        seed: int = 42,
    ) -> None:
        super().__init__(feature_cols)
        from pyod.models.iforest import IForest

        self._model = IForest(n_estimators=n_estimators, max_samples=max_samples, random_state=seed)


class ECODScorer(_PyODScorer):
    """M3b — ECOD. Hyperparameter-free, robust; no assumptions to tune wrong."""

    name = "M3b_ecod"
    requires_labels = False

    def __init__(self, feature_cols: list[str]) -> None:
        super().__init__(feature_cols)
        from pyod.models.ecod import ECOD

        self._model = ECOD()


class HBOSScorer(_PyODScorer):
    """M3b (alternate) — HBOS, the histogram-based hyperparameter-free sibling of ECOD."""

    name = "M3b_hbos"
    requires_labels = False

    def __init__(self, feature_cols: list[str]) -> None:
        super().__init__(feature_cols)
        from pyod.models.hbos import HBOS

        self._model = HBOS()


class LOFScorer(_PyODScorer):
    """M3c — Local Outlier Factor on a fixed subsample. LOF does not scale to
    LANL volumes, so the subsample size is a first-class, published parameter
    — its effect on variance is one of the required sensitivity analyses
    (spec section 6.5).
    """

    name = "M3c_lof"
    requires_labels = False

    def __init__(
        self,
        feature_cols: list[str],
        n_neighbors: int = 35,
        subsample_size: int = 100_000,
        seed: int = 42,
    ) -> None:
        super().__init__(feature_cols)
        from pyod.models.lof import LOF

        self._model = LOF(n_neighbors=n_neighbors)
        self.subsample_size = subsample_size
        self.seed = seed

    def fit(self, train: pl.LazyFrame) -> None:
        n = train.select(pl.len()).collect().item()
        if n > self.subsample_size:
            train = train.collect().sample(n=self.subsample_size, seed=self.seed).lazy()
        super().fit(train)
