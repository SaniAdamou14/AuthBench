"""The single contract every model in the catalog must satisfy (spec section 3.5).

A model that does not conform does not enter the results table. Evaluation
depends only on the **rank** of `score()`'s output, never its absolute scale
— this is what makes distances, probabilities, reconstruction errors, and
negative log-likelihoods directly comparable without arbitrary calibration.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class AnomalyScorer(Protocol):
    name: str
    requires_labels: bool

    def fit(self, train: pl.LazyFrame) -> None:
        """Fit on the training partition. `train` never contains data beyond
        the caller's split boundary — enforced upstream by
        `authbench.split.temporal`, not by this interface.
        """
        ...

    def score(self, data: pl.LazyFrame) -> pl.Series:
        """Return one score per event in `data`, same row order. Higher is
        more anomalous. No constraint on scale — only rank matters.
        """
        ...


class BaseAnomalyScorer:
    """Convenience base class: stores `name`/`requires_labels`, leaves
    `fit`/`score` to subclasses. Not required — anything satisfying
    `AnomalyScorer` structurally works — but avoids repeating the
    boilerplate in every model module.
    """

    name: str = "base"
    requires_labels: bool = False

    def fit(self, train: pl.LazyFrame) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def score(self, data: pl.LazyFrame) -> pl.Series:  # pragma: no cover - overridden
        raise NotImplementedError
