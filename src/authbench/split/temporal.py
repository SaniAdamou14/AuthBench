"""Strict temporal train/val/test split with a mechanical anti-leakage guard (US-107).

The split boundaries live in `conf/split/temporal.yaml`, never hard-coded.
`get_test_split` is the one function allowed to load the test partition; an
automated architecture test (`tests/unit/test_no_test_leakage.py`) fails the
build if any module under `authbench.models` or `authbench.features` imports
it — training code must never see the test set, by construction, not by
discipline alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


class LeakageError(RuntimeError):
    """Raised when a split's time bounds overlap or violate temporal ordering."""


@dataclass(frozen=True)
class TemporalSplitConfig:
    train_days: tuple[int, int]
    val_days: tuple[int, int]
    test_days: tuple[int, int]

    @classmethod
    def from_hydra(cls, cfg: object) -> TemporalSplitConfig:
        return cls(
            train_days=tuple(cfg.train_days),  # type: ignore[attr-defined]
            val_days=tuple(cfg.val_days),  # type: ignore[attr-defined]
            test_days=tuple(cfg.test_days),  # type: ignore[attr-defined]
        )


def get_train_split(data: pl.LazyFrame, config: TemporalSplitConfig) -> pl.LazyFrame:
    lo, hi = config.train_days
    return data.filter(pl.col("day").is_between(lo, hi))


def get_val_split(data: pl.LazyFrame, config: TemporalSplitConfig) -> pl.LazyFrame:
    lo, hi = config.val_days
    return data.filter(pl.col("day").is_between(lo, hi))


def get_test_split(data: pl.LazyFrame, config: TemporalSplitConfig) -> pl.LazyFrame:
    """The ONLY sanctioned entry point to the test partition.

    Forbidden inside `authbench.models.*` and `authbench.features.*` — see
    the module docstring and the architecture test that enforces it.
    """
    lo, hi = config.test_days
    return data.filter(pl.col("day").is_between(lo, hi))


def verify_temporal_order(train: pl.LazyFrame, val: pl.LazyFrame, test: pl.LazyFrame) -> None:
    """US-107: train.max(time) < val.min(time) < val.max(time) < test.min(time).

    Raises `LeakageError` rather than returning a bool — a leakage check that
    can be silently ignored is worse than none.
    """
    train_max = train.select(pl.col("time").max()).collect().item()
    val_min = val.select(pl.col("time").min()).collect().item()
    val_max = val.select(pl.col("time").max()).collect().item()
    test_min = test.select(pl.col("time").min()).collect().item()

    if train_max is None or val_min is None or val_max is None or test_min is None:
        raise LeakageError("One of train/val/test is empty — cannot verify temporal order.")

    if not (train_max < val_min):
        raise LeakageError(
            f"Train/val leakage: train max time {train_max} >= val min time {val_min}."
        )
    if not (val_max < test_min):
        raise LeakageError(f"Val/test leakage: val max time {val_max} >= test min time {test_min}.")


def random_ablation_split(
    data: pl.LazyFrame,
    *,
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Random (leaky) split — ablation only (section 6.5, RQ2). Never used for
    headline results. Callers must surface `conf/split/random_ablation.yaml`'s
    warning message whenever this is invoked.
    """
    n = data.select(pl.len()).collect().item()
    shuffled = data.with_columns(pl.arange(0, n).alias("_row")).with_columns(
        (pl.col("_row").hash(seed=seed) % n).alias("_shuffled_rank")
    )
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)

    test = shuffled.filter(pl.col("_shuffled_rank") < n_test).drop(["_row", "_shuffled_rank"])
    val = shuffled.filter(
        pl.col("_shuffled_rank").is_between(n_test, n_test + n_val, closed="left")
    ).drop(["_row", "_shuffled_rank"])
    train = shuffled.filter(pl.col("_shuffled_rank") >= n_test + n_val).drop(
        ["_row", "_shuffled_rank"]
    )
    return train, val, test
