"""M3 — classical ML anomaly detectors via PyOD (US-120, `authbench[classical]` extra)."""

from __future__ import annotations

import numpy as np
import polars as pl

from authbench.models.base import BaseAnomalyScorer, sample_for_fit


class _PyODScorer(BaseAnomalyScorer):
    """Shared plumbing for PyOD-backed models: a feature matrix in, a score out."""

    # Every PyOD estimator here scores each row against a fitted model and
    # nothing else, so a partition scored piecewise is bit-identical to the
    # whole — which is what makes the day-chunked path in `train_eval` safe.
    scores_row_locally = True

    def __init__(self, feature_cols: list[str], fit_sample_size: int | None = None) -> None:
        self.feature_cols = feature_cols
        self.fit_sample_size = fit_sample_size
        self.seed = 42
        self._model: object | None = None

    def _matrix(self, data: pl.LazyFrame) -> np.ndarray:
        return data.select(self.feature_cols).collect().to_numpy().astype(np.float64)

    def fit(self, train: pl.LazyFrame) -> None:
        assert self._model is not None, "subclass must set self._model before fit()"
        self._model.fit(self._matrix(sample_for_fit(train, self.fit_sample_size, self.seed)))  # type: ignore[attr-defined]

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
        fit_sample_size: int | None = None,
    ) -> None:
        super().__init__(feature_cols, fit_sample_size)
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        from pyod.models.iforest import IForest

        self._model = IForest(n_estimators=n_estimators, max_samples=max_samples, random_state=seed)


class ECODScorer(BaseAnomalyScorer):
    """M3b — ECOD (Li et al., 2022), with its empirical distributions fitted on training data.

    PyOD's `ECOD.decision_function` recomputes the empirical CDFs from the training
    sample concatenated with the matrix it scores, so an event's score depended on the
    other events scored with it (10.64 alone, 10.81 among four hundred others) and on
    the test distribution. Here the CDFs and the skewness signs come from the training
    sample alone, and an event is scored as PyOD would score it if it were the only
    event scored: the tail counts include the event itself, (count + 1) / (n + 1).
    The score is then a function of the row and the fit, in any batch.
    """

    name = "M3b_ecod"
    requires_labels = False
    scores_row_locally = True

    def __init__(
        self, feature_cols: list[str], fit_sample_size: int | None = None, seed: int = 42
    ) -> None:
        self.feature_cols = feature_cols
        self.fit_sample_size = fit_sample_size
        self.seed = seed
        self._sorted: np.ndarray | None = None
        self._tail_log: np.ndarray = np.empty(0)
        self._skew_sign: np.ndarray = np.empty(0)

    def _matrix(self, data: pl.LazyFrame) -> np.ndarray:
        return data.select(self.feature_cols).collect().to_numpy().astype(np.float64)

    def fit(self, train: pl.LazyFrame) -> None:
        from scipy.stats import skew

        x = self._matrix(sample_for_fit(train, self.fit_sample_size, self.seed))
        n = x.shape[0]
        self._sorted = np.sort(x, axis=0)
        # -log((c + 1) / (n + 1)) for every tail count c, looked up rather than recomputed per batch.
        self._tail_log = -np.log(np.arange(1, n + 2, dtype=np.float64) / (n + 1))
        self._skew_sign = np.sign(np.nan_to_num(skew(x, axis=0)))

    def score(self, data: pl.LazyFrame) -> pl.Series:
        if self._sorted is None:
            raise RuntimeError(f"{self.name} must be fitted before scoring.")
        x = self._matrix(data)
        n = self._sorted.shape[0]
        total = np.zeros(x.shape[0])
        for j, sign in enumerate(self._skew_sign):
            column = self._sorted[:, j]
            left = self._tail_log[np.searchsorted(column, x[:, j], side="right")]
            right = self._tail_log[n - np.searchsorted(column, x[:, j], side="left")]
            # PyOD's skewness-corrected tail: left for a negative skew, right for a positive one, both at zero.
            skewed = left * -np.sign(sign - 1) + right * np.sign(sign + 1)
            total += np.maximum(np.maximum(left, right), skewed)
        return pl.Series(self.name, total)


class HBOSScorer(_PyODScorer):
    """M3b (alternate) — HBOS, the histogram-based sibling of ECOD.

    The histograms are built at `fit` time and only looked up at scoring, so a matrix
    split in two scores identically to the whole (`tests/unit/test_row_locality.py`).
    `n_bins` and `tol` are its two hyperparameters (`pipeline.tuning`).
    """

    name = "M3b_hbos"
    requires_labels = False
    scores_row_locally = True

    def __init__(
        self,
        feature_cols: list[str],
        fit_sample_size: int | None = None,
        n_bins: int = 10,
        tol: float = 0.5,
    ) -> None:
        super().__init__(feature_cols, fit_sample_size)
        from pyod.models.hbos import HBOS

        self.n_bins = n_bins
        self.tol = tol
        self._model = HBOS(n_bins=n_bins, tol=tol)


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
