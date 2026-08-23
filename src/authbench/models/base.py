"""The single contract every model in the catalog must satisfy (spec section 3.5).

A model that does not conform does not enter the results table. Evaluation
depends only on the **rank** of `score()`'s output, never its absolute scale
— this is what makes distances, probabilities, reconstruction errors, and
negative log-likelihoods directly comparable without arbitrary calibration.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import polars as pl

logger = logging.getLogger(__name__)


@runtime_checkable
class AnomalyScorer(Protocol):
    name: str
    requires_labels: bool
    #: True when `score()` depends only on the current row, so scoring a
    #: partition of the data and concatenating gives exactly the same answer as
    #: scoring it whole. That is what lets `train_eval` score a split one day
    #: at a time instead of materializing hundreds of millions of rows as a
    #: float64 matrix.
    #:
    #: It is False for M1, and the distinction is not a technicality: R6 asks
    #: "has this user ever used this auth type *before*?" and R7 asks "did this
    #: user's previous event end where this one starts?". Chunking those by day
    #: would reset both at every midnight — R6 would call an auth type new once
    #: a day, R7 would lose every chain crossing a day boundary — and the
    #: scores would still look completely plausible.
    scores_row_locally: bool

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


#: Rows a vector-space model fits on when no explicit limit is given.
#:
#: Every estimator in the catalog estimates a *distribution* — a covariance, an
#: empirical CDF, a histogram, an ensemble of random splits. None of them needs
#: 550 million rows to do that, and M3a already subsamples internally to 256.
#: What 550 million rows does need is 21 float64 columns times three estimator
#: copies of working memory, which is several hundred gigabytes and the single
#: reason `train_eval` could not run on real data.
#:
#: Fitting on a bounded, seeded sample and scoring every event is standard
#: practice and, unlike the alternatives, changes nothing about what is
#: measured: the scores are still produced for the whole test split.
DEFAULT_FIT_SAMPLE_SIZE = 5_000_000


def sample_for_fit(train: pl.LazyFrame, sample_size: int | None, seed: int) -> pl.LazyFrame:
    """Bound the number of rows a model fits on, deterministically.

    `sample_size=None` means "no limit" and is what the demo and the tests use:
    on a sample smaller than the bound this is a no-op either way, and being
    explicit about it keeps small runs exactly reproducible against the old
    behaviour.
    """
    if sample_size is None:
        return train

    n_rows = train.select(pl.len()).collect().item()
    if n_rows <= sample_size:
        return train

    logger.info("Fitting on a seeded sample of %d/%d training rows.", sample_size, n_rows)
    return train.collect().sample(n=sample_size, seed=seed, shuffle=False).lazy()


class BaseAnomalyScorer:
    """Convenience base class: stores `name`/`requires_labels`, leaves
    `fit`/`score` to subclasses. Not required — anything satisfying
    `AnomalyScorer` structurally works — but avoids repeating the
    boilerplate in every model module.
    """

    name: str = "base"
    requires_labels: bool = False
    # Conservative default: a model must claim row-locality explicitly, because
    # getting this wrong produces plausible numbers rather than an error.
    scores_row_locally: bool = False

    def fit(self, train: pl.LazyFrame) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def score(self, data: pl.LazyFrame) -> pl.Series:  # pragma: no cover - overridden
        raise NotImplementedError
