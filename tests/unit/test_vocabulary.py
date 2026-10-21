"""Codes must not depend on the process that computes them (Polars categorical codes do)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from authbench.features.vocabulary import (
    FIRST_TOKEN_ID,
    NULL_ID,
    OOV_ID,
    encode,
    fit_vocabulary,
    n_tokens,
    read_vocabulary,
    write_vocabulary,
)


def _categorical(values: list[str | None]) -> pl.LazyFrame:
    return pl.LazyFrame({"host": values}, schema={"host": pl.Categorical})


def test_vocabulary_is_sorted_distinct_and_ignores_nulls() -> None:
    assert fit_vocabulary(_categorical(["C9", "C1", None, "C9", "C5"]), "host") == [
        "C1",
        "C5",
        "C9",
    ]


def test_known_unseen_and_null_values_get_distinct_reserved_codes() -> None:
    vocabulary = ["C1", "C5", "C9"]
    codes = (
        _categorical(["C5", "C7", None, "C1"])
        .select(encode("host", vocabulary))
        .collect()["host"]
        .to_list()
    )
    assert codes == [FIRST_TOKEN_ID + 1, OOV_ID, NULL_ID, FIRST_TOKEN_ID]
    assert n_tokens(vocabulary) == FIRST_TOKEN_ID + 3


def test_vocabulary_round_trips_through_parquet(tmp_path: Path) -> None:
    path = tmp_path / "vocab.parquet"
    write_vocabulary(["C1", "C5"], path)
    assert read_vocabulary(path) == ["C1", "C5"]


_READER = """
import json, sys
import polars as pl
from authbench.features.vocabulary import encode, fit_vocabulary
train, test, first = sys.argv[1], sys.argv[2], sys.argv[3]
order = [train, test] if first == "train" else [test, train]
frames = {path: pl.scan_parquet(path) for path in order}
for frame in frames.values():
    frame.select(pl.col("host").to_physical()).collect()
vocabulary = fit_vocabulary(frames[train], "host")
print(json.dumps(frames[test].select(encode("host", vocabulary)).collect()["host"].to_list()))
"""


def test_codes_are_identical_across_processes_that_read_in_opposite_orders(tmp_path: Path) -> None:
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    _categorical(["C9", "C1", "C5"]).collect().write_parquet(train)
    _categorical(["C5", "C7", "C9"]).collect().write_parquet(test)

    def codes(first: str) -> list[int]:
        out = subprocess.run(
            [sys.executable, "-c", _READER, str(train), str(test), first],
            capture_output=True,
            text=True,
            check=True,
        )
        return list(json.loads(out.stdout))

    assert codes("train") == codes("test") == [FIRST_TOKEN_ID + 1, OOV_ID, FIRST_TOKEN_ID + 2]
