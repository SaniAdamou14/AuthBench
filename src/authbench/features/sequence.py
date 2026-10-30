"""F6: the N most recent strictly-prior events of the same user, as fixed-length token sequences.

Every sequence is left-padded with `PAD_ID`, and no reserved token shares a
code with a real outcome: identifiers come from a train-fitted vocabulary
(`features.vocabulary`), success and failure are 1 and 2, and the first event
of a user has its own inter-arrival bucket.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from authbench.features.causal import sort_for_causal
from authbench.features.vocabulary import PAD_ID, encode

DEFAULT_CONTEXT_LENGTH = 32
N_DELTA_BUCKETS = 10
SUCCESS_TOKEN = 1
FAILURE_TOKEN = 2
FIRST_EVENT_BUCKET = 1

SEQUENCE_COLUMNS = (
    "sequence_dst",
    "sequence_auth_type",
    "sequence_success",
    "sequence_delta_bucket",
)


def _delta_bucket(delta_seconds: pl.Expr) -> pl.Expr:
    """Log-scale inter-arrival bucket in `[2, N_DELTA_BUCKETS + 1]`, `FIRST_EVENT_BUCKET` when there is no prior event."""
    scaled = (delta_seconds + 1.0).log().clip(0, 20) / 20 * (N_DELTA_BUCKETS - 1)
    return (
        pl.when(delta_seconds.is_null())
        .then(pl.lit(FIRST_EVENT_BUCKET, dtype=pl.Int32))
        .otherwise(scaled.round(0).cast(pl.Int32) + FIRST_EVENT_BUCKET + 1)
    )


def compute_f6(
    events: pl.LazyFrame,
    *,
    dst_vocabulary: Sequence[str],
    auth_type_vocabulary: Sequence[str],
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> pl.LazyFrame:
    """Add the four `SEQUENCE_COLUMNS` and `dst_code`, the current event's own destination token."""
    by_user = sort_for_causal(events, "src_user", "time")
    tokens = {
        "sequence_dst": "_dst",
        "sequence_auth_type": "_auth",
        "sequence_success": "_success",
        "sequence_delta_bucket": "_delta",
    }
    tokenized = by_user.with_columns(
        encode("dst_computer", dst_vocabulary).alias("_dst"),
        encode("auth_type", auth_type_vocabulary).alias("_auth"),
        pl.when(pl.col("success"))
        .then(pl.lit(SUCCESS_TOKEN, dtype=pl.Int32))
        .otherwise(pl.lit(FAILURE_TOKEN, dtype=pl.Int32))
        .alias("_success"),
        _delta_bucket(pl.col("time") - pl.col("time").shift(1).over("src_user")).alias("_delta"),
    )
    # Lag i is the i-th strictly-prior event: i starts at 1, so an event never sees itself.
    sequences = tokenized.with_columns(
        pl.concat_list(
            [
                pl.col(token).shift(lag).over("src_user").fill_null(PAD_ID)
                for lag in range(context_length, 0, -1)
            ]
        ).alias(name)
        for name, token in tokens.items()
    )
    return sequences.rename({"_dst": "dst_code"}).drop(["_auth", "_success", "_delta"])
