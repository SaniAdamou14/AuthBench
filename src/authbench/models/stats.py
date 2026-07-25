"""M2 — single-parameter statistical baselines (US-119)."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from authbench.models.base import BaseAnomalyScorer


class PairRarityScorer(BaseAnomalyScorer):
    """M2a — score is simply `pair_global_rarity` (-log frequency of the
    (user, destination) pair), already computed causally by F3. A one-line
    model with no fitting: the point is to show how far a single well-chosen
    statistic already gets you.
    """

    name = "M2a_pair_rarity"
    requires_labels = False

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

    def __init__(self, feature_cols: list[str], n_components: float = 0.95, seed: int = 42) -> None:
        self.feature_cols = feature_cols
        self.n_components = n_components
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)

    def _matrix(self, data: pl.LazyFrame) -> np.ndarray:
        return data.select(self.feature_cols).collect().to_numpy().astype(np.float64)

    def fit(self, train: pl.LazyFrame) -> None:
        x = self._matrix(train)
        x_scaled = self.scaler.fit_transform(x)
        self.pca.fit(x_scaled)

    def score(self, data: pl.LazyFrame) -> pl.Series:
        x = self._matrix(data)
        x_scaled = self.scaler.transform(x)
        x_reconstructed = self.pca.inverse_transform(self.pca.transform(x_scaled))
        error = np.mean((x_scaled - x_reconstructed) ** 2, axis=1)
        return pl.Series(self.name, error)
