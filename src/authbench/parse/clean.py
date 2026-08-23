"""Cleaning and typing of raw LANL auth/redteam events (US-104).

Every decision here is explicit and counted, never silent:
- machine accounts (`$` suffix) are flagged, not dropped or coerced to humans.
- null `auth_type` / `logon_type` become dedicated boolean indicators — the
  nullity is itself a signal (see docstring in `schema.py`).
- dropped rows are counted by reason and surfaced in the data-quality report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from authbench.parse.schema import (
    LANL_NULL_TOKEN,
    SECONDS_PER_DAY,
    TYPED_AUTH_SCHEMA,
    TYPED_REDTEAM_SCHEMA,
)


@dataclass
class DropCounts:
    """Rows removed during cleaning, by reason — never a silent drop."""

    malformed_user_domain: int = 0
    null_time: int = 0
    other: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.malformed_user_domain + self.null_time + sum(self.other.values())


def _split_user_domain(col: str, user_out: str, domain_out: str) -> list[pl.Expr]:
    """Split a `user@domain` column into two, tolerating a missing `@domain`."""
    parts = pl.col(col).str.split_exact("@", 1)
    return [
        parts.struct.field("field_0").alias(user_out),
        parts.struct.field("field_1").alias(domain_out),
    ]


def _null_token_to_null(col: str) -> pl.Expr:
    """LANL's `?` sentinel becomes a real null, keeping the typed column clean."""
    return (
        pl.when(pl.col(col) == LANL_NULL_TOKEN)
        .then(None)
        .otherwise(pl.col(col))
        .cast(pl.Categorical)
        .alias(col)
    )


def _is_null_indicator(col: str) -> pl.Expr:
    """The `<col>_is_null` boolean — nullity is itself a signal (see `schema.py`).

    A bare `col == LANL_NULL_TOKEN` is *null*, not False, whenever the source
    field is already missing (an empty CSV field rather than a literal `?`).
    That null then flows straight into the F1 model matrix as a NaN. Both
    forms of "no value" have to answer True here.
    """
    return (
        (pl.col(col).is_null() | (pl.col(col) == LANL_NULL_TOKEN))
        .fill_null(True)
        .alias(f"{col}_is_null")
    )


def is_machine_account(user_col: str | pl.Expr) -> pl.Expr:
    """LANL machine accounts end with `$` — flagged explicitly, never dropped."""
    expr = pl.col(user_col) if isinstance(user_col, str) else user_col
    return expr.str.ends_with("$").fill_null(False)


def clean_auth(raw: pl.LazyFrame, *, id_offset: int = 0) -> tuple[pl.LazyFrame, DropCounts]:
    """Parse raw auth.txt columns into the typed schema.

    Rows whose `src_user_at_domain` or `dst_user_at_domain` do not split into
    exactly `user@domain` are dropped and counted (`malformed_user_domain`);
    everything else is a deterministic 1:1 transform.

    `id_offset` lets a streaming caller assign globally-unique, deterministic
    `event_id`s across batches (row index within the batch is otherwise always
    zero-based) without keeping any other cross-batch state — required for
    bit-identical reruns (NFR-02).
    """
    counts = DropCounts()

    n_before = raw.select(pl.len()).collect().item()

    with_split = raw.with_columns(
        [
            *_split_user_domain("src_user_at_domain", "src_user", "src_domain"),
            *_split_user_domain("dst_user_at_domain", "dst_user", "dst_domain"),
        ]
    )

    well_formed = with_split.filter(
        pl.col("src_user").is_not_null()
        & pl.col("src_domain").is_not_null()
        & pl.col("dst_user").is_not_null()
        & pl.col("dst_domain").is_not_null()
        & pl.col("time").is_not_null()
    )

    n_after = well_formed.select(pl.len()).collect().item()
    n_dropped = n_before - n_after
    # Attribute the drop to disjoint reasons. A row can be malformed *and*
    # have a null time; counting each reason independently and subtracting
    # would then double-count it and could drive `malformed_user_domain`
    # negative — so `null_time` is claimed first and `malformed_user_domain`
    # is whatever is left, which keeps `total` equal to `n_dropped` by
    # construction (US-102's row-count reconciliation depends on that).
    counts.null_time = min(
        raw.filter(pl.col("time").is_null()).select(pl.len()).collect().item(), n_dropped
    )
    counts.malformed_user_domain = n_dropped - counts.null_time

    typed = well_formed.select(
        [
            pl.col("time").cast(pl.Int32),
            (pl.col("time") // SECONDS_PER_DAY).cast(pl.Int16).alias("day"),
            pl.col("src_user").cast(pl.Categorical),
            pl.col("src_domain").cast(pl.Categorical),
            pl.col("dst_user").cast(pl.Categorical),
            pl.col("dst_domain").cast(pl.Categorical),
            pl.col("src_computer").cast(pl.Categorical),
            pl.col("dst_computer").cast(pl.Categorical),
            _null_token_to_null("auth_type"),
            _is_null_indicator("auth_type"),
            _null_token_to_null("logon_type"),
            _is_null_indicator("logon_type"),
            pl.col("auth_orientation").cast(pl.Categorical),
            (pl.col("success_failure") == "Success").alias("success"),
            is_machine_account("src_user").alias("src_user_is_machine"),
            is_machine_account("dst_user").alias("dst_user_is_machine"),
        ]
    ).with_row_index("event_id", offset=id_offset)

    typed = typed.select(list(TYPED_AUTH_SCHEMA.keys()))

    return typed, counts


def clean_redteam(raw: pl.LazyFrame) -> tuple[pl.LazyFrame, int]:
    """Parse raw redteam.txt into the typed schema.

    Returns the typed frame and the number of exact-duplicate rows removed.
    Measured on the real file: 749 raw rows, **34** exact duplicates, 715
    distinct. Widely-repeated secondary figures of 12 duplicates / 737 unique
    do not match the file, which is exactly why US-105 requires this count be
    logged rather than silently absorbed — a hard-coded 737 would have made a
    real discrepancy invisible.
    """
    parts = pl.col("user_at_domain").str.split_exact("@", 1)
    typed = raw.with_columns(
        [
            parts.struct.field("field_0").alias("user"),
            parts.struct.field("field_1").alias("domain"),
        ]
    ).select(
        [
            pl.col("time").cast(pl.Int32),
            (pl.col("time") // SECONDS_PER_DAY).cast(pl.Int16).alias("day"),
            pl.col("user").cast(pl.Categorical),
            pl.col("domain").cast(pl.Categorical),
            pl.col("src_computer").cast(pl.Categorical),
            pl.col("dst_computer").cast(pl.Categorical),
        ]
    )
    typed = typed.select(list(TYPED_REDTEAM_SCHEMA.keys()))

    n_before = typed.select(pl.len()).collect().item()
    deduplicated = typed.unique()
    n_after = deduplicated.select(pl.len()).collect().item()

    return deduplicated, n_before - n_after


def data_quality_report(typed: pl.LazyFrame, drop_counts: DropCounts) -> dict[str, object]:
    """Compute the per-column null rate, cardinality and duplicate counts (US-104).

    Runs as lazy aggregations so it scales to the full out-of-core dataset —
    the frame is never fully materialized in memory (NFR-01).
    """
    schema_cols = list(TYPED_AUTH_SCHEMA.keys())
    categorical_cols = [c for c in schema_cols if TYPED_AUTH_SCHEMA[c] == pl.Categorical]
    non_id_cols = [c for c in schema_cols if c != "event_id"]

    n_rows = typed.select(pl.len()).collect().item()

    null_counts = typed.select([pl.col(c).null_count().alias(c) for c in schema_cols]).collect()
    null_rates = {c: (null_counts[c][0] / n_rows if n_rows else 0.0) for c in schema_cols}

    cardinality_df = typed.select(
        [pl.col(c).n_unique().alias(c) for c in categorical_cols]
    ).collect()
    cardinality = {c: cardinality_df[c][0] for c in categorical_cols}

    n_distinct = (
        typed.select(pl.struct(non_id_cols).n_unique().alias("n_distinct")).collect().item()
    )
    n_duplicate_rows = n_rows - n_distinct

    return {
        "n_rows": n_rows,
        "null_rates": null_rates,
        "cardinality": cardinality,
        "n_duplicate_rows": n_duplicate_rows,
        "dropped": {
            "malformed_user_domain": drop_counts.malformed_user_domain,
            "null_time": drop_counts.null_time,
            "total": drop_counts.total,
        },
    }
