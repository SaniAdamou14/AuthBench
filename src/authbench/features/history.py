"""F2 — historical per-entity features over 1h / 24h / 7d expanding windows (US-110).

Computed for three entities: the source user, the source machine, and the
destination machine. Every window is strictly antecedent to the current
event via `features.causal` — the event itself never contributes to its own
aggregation, which is the single most common source of leakage in this
literature (spec section 4.1).
"""

from __future__ import annotations

import polars as pl

from authbench.features.causal import (
    causal_prior_events,
    causal_rolling_count,
    causal_rolling_sum,
    shannon_entropy_expr,
    sort_for_causal,
)

DEFAULT_WINDOWS_HOURS: list[int] = [1, 24, 168]
DEFAULT_ENTITIES: list[str] = ["src_user", "src_computer", "dst_computer"]


def _window_suffix(hours: int) -> str:
    if hours % 24 == 0 and hours >= 24:
        return f"{hours // 24}d"
    return f"{hours}h"


def compute_f2_for_entity_window(
    events: pl.LazyFrame,
    *,
    entity_col: str,
    window_hours: int,
) -> pl.LazyFrame:
    """Return `event_id` plus the F2 columns for one (entity, window) pair."""
    window_seconds = window_hours * 3600
    suffix = _window_suffix(window_hours)
    prefix = f"{entity_col}_{suffix}"

    sorted_events = sort_for_causal(events, entity_col, "time")

    n_events = causal_rolling_count(
        sorted_events, entity_col=entity_col, time_col="time", window_seconds=window_seconds
    )
    n_failures = causal_rolling_sum(
        sorted_events,
        entity_col=entity_col,
        time_col="time",
        value_col="_is_failure",
        window_seconds=window_seconds,
    )

    with_counts = sorted_events.with_columns(
        [
            n_events.alias(f"{prefix}_n_events"),
            n_failures.alias(f"{prefix}_n_failures"),
        ]
    ).with_columns(
        (
            pl.col(f"{prefix}_n_failures")
            / pl.when(pl.col(f"{prefix}_n_events") > 0)
            .then(pl.col(f"{prefix}_n_events"))
            .otherwise(1)
        ).alias(f"{prefix}_failure_ratio")
    )

    diversity = _compute_diversity(
        events, entity_col=entity_col, window_seconds=window_seconds, prefix=prefix
    )

    # `diversity` only has rows for events with at least one strictly-prior
    # event in the window — an entity's first-ever event in the period has
    # none, and is therefore absent (not merely null) from `diversity`. The
    # left join below reintroduces those events with nulls, so every count
    # must be filled *after* the join, not only inside `_compute_diversity`.
    return (
        with_counts.select(
            ["event_id", f"{prefix}_n_events", f"{prefix}_n_failures", f"{prefix}_failure_ratio"]
        )
        .join(diversity, on="event_id", how="left")
        .with_columns(
            [
                pl.col(f"{prefix}_n_distinct_dst").fill_null(0),
                pl.col(f"{prefix}_dst_entropy").fill_null(0.0),
                pl.col(f"{prefix}_n_distinct_auth_type").fill_null(0),
            ]
        )
    )


def _diversity_partner_column(entity_col: str) -> str:
    """The natural "what did this entity reach?" counterpart column.

    For a user or a source machine, that's the destination reached. For a
    destination machine itself, counting "distinct destinations" would be
    degenerate (it's always the entity itself) — the meaningful counterpart
    is the diversity of *users* reaching it instead.
    """
    return "src_user" if entity_col == "dst_computer" else "dst_computer"


def _compute_diversity(
    events: pl.LazyFrame, *, entity_col: str, window_seconds: int, prefix: str
) -> pl.LazyFrame:
    """Distinct destinations reached, entropy of the destination distribution,
    and distinct auth types seen — all over strictly-prior events (US-110).
    """
    partner_col = _diversity_partner_column(entity_col)
    joined = causal_prior_events(
        events,
        entity_col=entity_col,
        time_col="time",
        id_col="event_id",
        carry_cols=[partner_col, "auth_type"],
        window_seconds=window_seconds,
        namespace=prefix,
    )

    dst_counts = (
        joined.group_by(["_current_id", partner_col])
        .agg(pl.len().alias("_n"))
        .group_by("_current_id")
        .agg(
            [
                pl.col(partner_col).n_unique().alias(f"{prefix}_n_distinct_dst"),
                shannon_entropy_expr("_n").alias(f"{prefix}_dst_entropy"),
            ]
        )
    )

    auth_type_counts = joined.group_by("_current_id").agg(
        pl.col("auth_type").n_unique().alias(f"{prefix}_n_distinct_auth_type")
    )

    result = dst_counts.join(auth_type_counts, on="_current_id", how="full", coalesce=True)
    return result.rename({"_current_id": "event_id"}).with_columns(
        [
            pl.col(f"{prefix}_n_distinct_dst").fill_null(0),
            pl.col(f"{prefix}_dst_entropy").fill_null(0.0),
            pl.col(f"{prefix}_n_distinct_auth_type").fill_null(0),
        ]
    )


def compute_f2(
    events: pl.LazyFrame,
    *,
    entities: list[str] | None = None,
    windows_hours: list[int] | None = None,
) -> pl.LazyFrame:
    """Compute the full F2 feature set (all entities x all windows) and join
    everything back onto the original events by `event_id`.
    """
    entities = entities or DEFAULT_ENTITIES
    windows_hours = windows_hours or DEFAULT_WINDOWS_HOURS

    base = events.with_columns((~pl.col("success")).cast(pl.Int32).alias("_is_failure"))

    result = events
    for entity_col in entities:
        for window_hours in windows_hours:
            feat = compute_f2_for_entity_window(
                base, entity_col=entity_col, window_hours=window_hours
            )
            result = result.join(feat, on="event_id", how="left")

    return result
