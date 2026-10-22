"""M0 — comparison floors (US-117). Every other model must beat these, or it
contributes nothing to the benchmark's conclusions.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from authbench.models.base import BaseAnomalyScorer

_MASK64 = (1 << 64) - 1


def _splitmix64_uniform(keys: np.ndarray, seed: int) -> np.ndarray:
    """Uniform [0, 1) doubles from 64-bit keys through SplitMix64 (Steele et al., 2014)."""
    z = keys.astype(np.uint64) + np.uint64((seed * 0x9E3779B97F4A7C15) & _MASK64)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * 2.0**-53


class RandomScorer(BaseAnomalyScorer):
    """M0a — uniform random score. The absolute floor.

    The score is a hash of `event_id` and the seed rather than a draw from a generator
    run over the frame, so an event scores the same in any batch, in any order.
    """

    name = "M0a_random"
    requires_labels = False
    scores_row_locally = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, train: pl.LazyFrame) -> None:
        return None

    def score(self, data: pl.LazyFrame) -> pl.Series:
        keys = data.select("event_id").collect().to_series().to_numpy()
        return pl.Series(self.name, _splitmix64_uniform(keys, self.seed))


class AlwaysFailScorer(BaseAnomalyScorer):
    """M0b — alert on every authentication failure. The realistic naive floor:
    any model that cannot beat "alert on every failed login" has learned
    nothing beyond what a one-line rule already captures.
    """

    name = "M0b_always_fail"
    requires_labels = False
    scores_row_locally = True

    def fit(self, train: pl.LazyFrame) -> None:
        return None

    def score(self, data: pl.LazyFrame) -> pl.Series:
        return (
            data.select((~pl.col("success")).cast(pl.Float64).alias(self.name))
            .collect()
            .to_series()
        )
