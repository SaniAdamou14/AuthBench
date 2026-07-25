"""M0 — comparison floors (US-117). Every other model must beat these, or it
contributes nothing to the benchmark's conclusions.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from authbench.models.base import BaseAnomalyScorer


class RandomScorer(BaseAnomalyScorer):
    """M0a — uniform random score. The absolute floor."""

    name = "M0a_random"
    requires_labels = False

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, train: pl.LazyFrame) -> None:
        return None

    def score(self, data: pl.LazyFrame) -> pl.Series:
        n = data.select(pl.len()).collect().item()
        rng = np.random.default_rng(self.seed)
        return pl.Series(self.name, rng.uniform(size=n))


class AlwaysFailScorer(BaseAnomalyScorer):
    """M0b — alert on every authentication failure. The realistic naive floor:
    any model that cannot beat "alert on every failed login" has learned
    nothing beyond what a one-line rule already captures.
    """

    name = "M0b_always_fail"
    requires_labels = False

    def fit(self, train: pl.LazyFrame) -> None:
        return None

    def score(self, data: pl.LazyFrame) -> pl.Series:
        return (
            data.select((~pl.col("success")).cast(pl.Float64).alias(self.name))
            .collect()
            .to_series()
        )
