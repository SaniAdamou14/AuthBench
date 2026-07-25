from __future__ import annotations

import polars as pl
import pytest

from authbench.split.temporal import (
    LeakageError,
    TemporalSplitConfig,
    get_test_split,
    get_train_split,
    get_val_split,
    verify_temporal_order,
)


def _events(days: list[int]) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "day": days,
            "time": [d * 86_400 + 100 for d in days],
        }
    )


CONFIG = TemporalSplitConfig(train_days=(0, 2), val_days=(3, 4), test_days=(5, 6))


def test_valid_split_passes_leakage_guard() -> None:
    events = _events([0, 1, 2, 3, 4, 5, 6])
    train = get_train_split(events, CONFIG)
    val = get_val_split(events, CONFIG)
    test = get_test_split(events, CONFIG)

    verify_temporal_order(train, val, test)  # must not raise

    assert train.select(pl.len()).collect().item() == 3
    assert val.select(pl.len()).collect().item() == 2
    assert test.select(pl.len()).collect().item() == 2


def test_overlapping_split_raises_leakage_error() -> None:
    # Deliberately construct a "val" set that includes a training-period timestamp.
    events = _events([0, 1, 2, 3, 4, 5, 6])
    train = get_train_split(events, CONFIG)
    leaky_val = events.filter(pl.col("day").is_between(2, 4))  # overlaps train's day 2
    test = get_test_split(events, CONFIG)

    with pytest.raises(LeakageError):
        verify_temporal_order(train, leaky_val, test)


def test_empty_partition_raises_leakage_error() -> None:
    events = _events([0, 1, 2])
    train = get_train_split(events, CONFIG)
    val = get_val_split(events, CONFIG)  # empty: no day in [3, 4]
    test = get_test_split(events, CONFIG)  # empty

    with pytest.raises(LeakageError):
        verify_temporal_order(train, val, test)
