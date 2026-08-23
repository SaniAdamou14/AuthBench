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
        from pyod.models.iforest import IForest

        self._model = IForest(n_estimators=n_estimators, max_samples=max_samples, random_state=seed)


class ECODScorer(_PyODScorer):
    """M3b — ECOD. Hyperparameter-free, robust; no assumptions to tune wrong.

    **ECOD is transductive, and that is a property of the model, not of this
    wrapper.** PyOD's `ECOD.decision_function` recomputes the empirical
    cumulative distribution from the matrix it is handed, so an event's score
    depends on which other events are scored alongside it: the same row scores
    10.64 alone and 10.81 among four hundred others, and splitting a matrix in
    two changes every score in it by up to 0.44.

    Two consequences, both real:

    - it cannot be scored in day chunks, hence the override below — the
      equivalence test in `tests/unit/test_chunked_scoring.py` is what caught
      this, having been written to catch exactly the opposite mistake;
    - its test scores depend on the *test* distribution. No label crosses the
      boundary, so this is not leakage in the sense US-107 guards against, but
      it does mean M3b is not a pure "fit here, apply there" model, and a
      benchmark that claims a strictly temporal protocol owes the reader that
      sentence. See `docs/methodology.md`.
    """

    name = "M3b_ecod"
    requires_labels = False
    scores_row_locally = False

    def __init__(self, feature_cols: list[str], fit_sample_size: int | None = None) -> None:
        super().__init__(feature_cols, fit_sample_size)
        from pyod.models.ecod import ECOD

        self._model = ECOD()


class HBOSScorer(_PyODScorer):
    """M3b (alternate) — HBOS, the histogram-based hyperparameter-free sibling of ECOD.

    Unlike ECOD it *is* row-local: the histograms are built at `fit` time and
    only looked up at scoring, so a matrix split in two scores identically to
    the whole. Verified, not assumed — see `test_chunked_scoring.py`.
    """

    name = "M3b_hbos"
    requires_labels = False
    scores_row_locally = True

    def __init__(self, feature_cols: list[str], fit_sample_size: int | None = None) -> None:
        super().__init__(feature_cols, fit_sample_size)
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
