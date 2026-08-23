"""M2 — single-parameter statistical baselines (US-119)."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from authbench.models.base import BaseAnomalyScorer, sample_for_fit


class PairRarityScorer(BaseAnomalyScorer):
    """M2a — score is simply `pair_global_rarity` (-log frequency of the
    (user, destination) pair), already computed causally by F3. A one-line
    model with no fitting: the point is to show how far a single well-chosen
    statistic already gets you.
    """

    name = "M2a_pair_rarity"
    requires_labels = False
    scores_row_locally = True

    def fit(self, train: pl.LazyFrame) -> None:
        return None

    def score(self, data: pl.LazyFrame) -> pl.Series:
        return data.select(pl.col("pair_global_rarity").alias(self.name)).collect().to_series()


class PCAReconstructionScorer(BaseAnomalyScorer):
    """M2b — PCA reconstruction error on the numeric feature columns. The
    linear baseline every nonlinear (autoencoder) model must beat to justify
    its extra complexity.
    """

    name = "M2b_pca_reconstruction"
    requires_labels = False
    scores_row_locally = True

    def __init__(
        self,
        feature_cols: list[str],
        n_components: float = 0.95,
        seed: int = 42,
        fit_sample_size: int | None = None,
    ) -> None:
        self.feature_cols = feature_cols
        self.n_components = n_components
        self.seed = seed
        self.fit_sample_size = fit_sample_size
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)

    def _matrix(self, data: pl.LazyFrame) -> np.ndarray:
        return data.select(self.feature_cols).collect().to_numpy().astype(np.float64)

    def fit(self, train: pl.LazyFrame) -> None:
        # A covariance and a set of principal axes are distribution estimates;
        # five million rows pins them far tighter than any downstream interval
        # can resolve. Scoring still covers every event — see `sample_for_fit`.
        x = self._matrix(sample_for_fit(train, self.fit_sample_size, self.seed))
        x_scaled = self.scaler.fit_transform(x)
        self.pca.fit(x_scaled)

    def score(self, data: pl.LazyFrame) -> pl.Series:
        x = self._matrix(data)
        x_scaled = self.scaler.transform(x)
        x_reconstructed = self.pca.inverse_transform(self.pca.transform(x_scaled))
        error = np.mean((x_scaled - x_reconstructed) ** 2, axis=1)
        return pl.Series(self.name, error)
