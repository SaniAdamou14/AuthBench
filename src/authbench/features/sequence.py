"""F6 — sequential encoding of the last N events per user, for sequence models (US-114).

Each event gets its own strictly-causal context: the N most recent *prior*
events of the same user, encoded as (dst_computer_id, auth_type_id, success,
discretized delta-t) tuples. Users with fewer than N prior events are
left-padded with a dedicated sentinel id (-1) — never with a real category
code, which would silently alias padding to an actual destination/auth type.
"""

from __future__ import annotations

import polars as pl

from authbench.features.causal import sort_for_causal

DEFAULT_CONTEXT_LENGTH = 32
PAD_ID = -1
N_DELTA_BUCKETS = 10


def _discretize_delta_t(delta_seconds_col: pl.Expr) -> pl.Expr:
    """Log-scale bucketing of inter-event delay, in `N_DELTA_BUCKETS` bins.

    `delta_seconds_col` is null for a user's first-ever event (no prior delta);
    that case gets its own bucket (0) rather than colliding with a real delay.
    """
    log_delta = (delta_seconds_col + 1.0).log()
    bucket = (log_delta.clip(0, 20) / 20 * (N_DELTA_BUCKETS - 1)).round(0).cast(pl.Int32) + 1
    return pl.when(delta_seconds_col.is_null()).then(0).otherwise(bucket)


def compute_f6(
    events: pl.LazyFrame, *, context_length: int = DEFAULT_CONTEXT_LENGTH
) -> pl.LazyFrame:
    """Add `sequence_dst`, `sequence_auth_type`, `sequence_success`,
    `sequence_delta_bucket` — each a fixed-length (`context_length`) list
    column of the N most recent strictly-prior events for the same user.
    """
    by_user = sort_for_causal(events, "src_user", "time")

    with_codes = by_user.with_columns(
        [
            pl.col("dst_computer").to_physical().alias("_dst_code"),
            pl.col("auth_type").to_physical().fill_null(-1).alias("_auth_code"),
            pl.col("success").cast(pl.Int32).alias("_success_int"),
            (pl.col("time") - pl.col("time").shift(1).over("src_user")).alias("_delta_t"),
        ]
    ).with_columns(_discretize_delta_t(pl.col("_delta_t")).alias("_delta_bucket"))

    # Shift(i) over the user gives the i-th strictly-prior event's value —
    # never the current row's own value (i starts at 1).
    dst_lags = [pl.col("_dst_code").shift(i).over("src_user") for i in range(1, context_length + 1)]
    auth_lags = [
        pl.col("_auth_code").shift(i).over("src_user") for i in range(1, context_length + 1)
    ]
    success_lags = [
        pl.col("_success_int").shift(i).over("src_user") for i in range(1, context_length + 1)
    ]
    delta_lags = [
        pl.col("_delta_bucket").shift(i).over("src_user") for i in range(1, context_length + 1)
    ]

    with_sequences = with_codes.with_columns(
        [
            pl.concat_list([lag.fill_null(PAD_ID) for lag in reversed(dst_lags)]).alias(
                "sequence_dst"
            ),
            pl.concat_list([lag.fill_null(PAD_ID) for lag in reversed(auth_lags)]).alias(
                "sequence_auth_type"
            ),
            pl.concat_list([lag.fill_null(0) for lag in reversed(success_lags)]).alias(
                "sequence_success"
            ),
            pl.concat_list([lag.fill_null(0) for lag in reversed(delta_lags)]).alias(
                "sequence_delta_bucket"
            ),
        ]
    )

    return with_sequences.drop(
        ["_dst_code", "_auth_code", "_success_int", "_delta_t", "_delta_bucket"]
    )
