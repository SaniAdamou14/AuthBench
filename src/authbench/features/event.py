"""F1 — event-level features: zero cost, available immediately, no history needed (US-109)."""

from __future__ import annotations

import polars as pl

F1_COLUMNS: list[str] = [
    "is_success",
    "auth_type_is_null",
    "logon_type_is_null",
    "src_user_is_machine",
    "src_dst_user_same",
    "src_dst_computer_same",
    "domain_crossing",
    "auth_type_freq",
    "logon_type_freq",
    "auth_orientation_freq",
]


def _frequency_encode(col: str) -> pl.Expr:
    """Frequency encoding — not one-hot: `auth_type`/`logon_type`/`auth_orientation`
    are high-cardinality categoricals where one-hot would explode dimensionality.
    """
    return (pl.col(col).count().over(col) / pl.len()).alias(f"{col}_freq")


def compute_f1(events: pl.LazyFrame) -> pl.LazyFrame:
    """Add the F1 event-level feature columns to a typed auth-event frame."""
    return events.with_columns(
        [
            pl.col("success").alias("is_success"),
            pl.col("auth_type_is_null"),
            pl.col("logon_type_is_null"),
            pl.col("src_user_is_machine"),
            (pl.col("src_user") == pl.col("dst_user")).alias("src_dst_user_same"),
            (pl.col("src_computer") == pl.col("dst_computer")).alias("src_dst_computer_same"),
            (pl.col("src_domain") != pl.col("dst_domain")).alias("domain_crossing"),
            _frequency_encode("auth_type"),
            _frequency_encode("logon_type"),
            _frequency_encode("auth_orientation"),
        ]
    )
