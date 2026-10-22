"""Train-fitted vocabularies that map categorical identifiers to stable integer codes.

Polars' physical categorical codes depend on the order in which values were
first seen by the process, so two processes reading the same Parquet files in
different orders assign different codes. Any integer that crosses a process
boundary (the feature store, a fitted model) must come from here instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

PAD_ID = 0
OOV_ID = 1
NULL_ID = 2
FIRST_TOKEN_ID = 3


def fit_vocabulary(train: pl.LazyFrame, column: str) -> list[str]:
    """Sorted distinct non-null values of `column` in the training partition."""
    values = (
        train.select(pl.col(column).cast(pl.String))
        .drop_nulls()
        .unique()
        .sort(column)
        .collect()[column]
    )
    return values.to_list()


def n_tokens(vocabulary: Sequence[str]) -> int:
    """Size of the code space, reserved ids included."""
    return FIRST_TOKEN_ID + len(vocabulary)


def encode(column: str, vocabulary: Sequence[str]) -> pl.Expr:
    """Map `column` to its code: known values from `FIRST_TOKEN_ID`, unseen to `OOV_ID`, null to `NULL_ID`."""
    codes = list(range(FIRST_TOKEN_ID, FIRST_TOKEN_ID + len(vocabulary)))
    return (
        pl.when(pl.col(column).is_null())
        .then(pl.lit(NULL_ID, dtype=pl.Int32))
        .otherwise(
            pl.col(column)
            .cast(pl.String)
            .replace_strict(list(vocabulary), codes, default=OOV_ID, return_dtype=pl.Int32)
        )
        .alias(column)
    )


def write_vocabulary(vocabulary: Sequence[str], path: Path) -> None:
    pl.DataFrame({"value": list(vocabulary)}, schema={"value": pl.String}).write_parquet(path)


def read_vocabulary(path: Path) -> list[str]:
    return pl.read_parquet(path)["value"].to_list()
