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
    causal_distinct_count,
    causal_prior_events,
    causal_rolling_count,
    causal_rolling_sum,
    shannon_entropy_expr,
    sort_for_causal,
)

DEFAULT_WINDOWS_HOURS: list[int] = [1, 24, 168]
DEFAULT_ENTITIES: list[str] = ["src_user", "src_computer", "dst_computer"]

# The (entity, window) pairs whose diversity columns anything downstream reads.
#
# Exactly one: M1's rule R2 compares `src_user_1h_n_distinct_dst` against the
# user's own 99.9th percentile. The other eight pairs were each running a
# `causal_prior_events` self-join — one row per (event, prior event of the same
# entity in the window) — and producing three columns that no model, rule or
# report has ever consumed. On the demo sample that is invisible; over a 7-day
# window on LANL, where a busy machine sees millions of events, it is the
# difference between a stage that completes and one that does not.
DEFAULT_DIVERSITY_PAIRS: list[tuple[str, int]] = [("src_user", 1)]

# The entities and windows anything downstream actually reads.
#
# `FEATURE_STORE_COLUMNS` consumes four F2 columns in total: the source user's
# 1h and 24h event counts and failure ratios, that user's 1h distinct-host
# count (M1's R2), and the destination machine's 1h failure count (M1's R3).
# Computing the default 3 entities x 3 windows produces 54 columns to serve
# those four, and the 168-hour window is the most expensive of the nine: a
# rolling aggregation over a week of a user's history, per event, discarded
# immediately. Restricting the grid is not a loss of information — nothing
# read the rest.
DEFAULT_F2_ENTITIES: list[str] = ["src_user", "dst_computer"]
DEFAULT_F2_WINDOWS_HOURS: list[int] = [1, 24]


def _window_suffix(hours: int) -> str:
    if hours % 24 == 0 and hours >= 24:
        return f"{hours // 24}d"
    return f"{hours}h"


def compute_f2_for_entity_window(
    events: pl.LazyFrame,
    *,
    entity_col: str,
    window_hours: int,
    with_diversity: bool = True,
    with_distribution_stats: bool = False,
) -> pl.LazyFrame:
    """Return `event_id` plus the F2 columns for one (entity, window) pair.

    `with_diversity=False` drops the three set/distribution columns
    (`_n_distinct_dst`, `_dst_entropy`, `_n_distinct_auth_type`) and, with
    them, the `causal_prior_events` self-join that produces them. The counts
    and ratios are rolling aggregations and cost O(n log n); the self-join
    costs O(events per entity per window) *per event*, which at LANL's
    ~1,500 events per user per day is the difference between a stage that
    finishes and one that does not. Callers that do not consume the diversity
    columns should not pay for them.
    """
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

    counts = with_counts.select(
        ["event_id", f"{prefix}_n_events", f"{prefix}_n_failures", f"{prefix}_failure_ratio"]
    )
    if not with_diversity:
        return counts

    # The distinct-partner count comes from the interval sweep, not the
    # self-join: same number, O(n log n) instead of O(events per entity per
    # window) per event. `tests/unit/test_distinct_count.py` holds the two
    # against each other.
    distinct = causal_distinct_count(
        events,
        entity_col=entity_col,
        partner_col=_diversity_partner_column(entity_col),
        time_col="time",
        id_col="event_id",
        window_seconds=window_seconds,
        out_col=f"{prefix}_n_distinct_dst",
    )
    result = counts.join(distinct, on="event_id", how="left").with_columns(
        pl.col(f"{prefix}_n_distinct_dst").fill_null(0)
    )
    if not with_distribution_stats:
        return result

    # Entropy and distinct-auth-type still need the pairwise expansion; nothing
    # downstream consumes them, so they are off by default and this branch
    # exists for analysis on samples small enough to afford it.
    stats = _compute_distribution_stats(
        events, entity_col=entity_col, window_seconds=window_seconds, prefix=prefix
    )
    return result.join(stats, on="event_id", how="left").with_columns(
        [
            pl.col(f"{prefix}_dst_entropy").fill_null(0.0),
            pl.col(f"{prefix}_n_distinct_auth_type").fill_null(0),
        ]
    )


def _diversity_partner_column(entity_col: str) -> str:
    """The natural "what did this entity reach?" counterpart column.

    For a user or a source machine, that's the destination reached. For a
    destination machine itself, counting "distinct destinations" would be
    degenerate (it's always the entity itself) — the meaningful counterpart
    is the diversity of *users* reaching it instead.
    """
    return "src_user" if entity_col == "dst_computer" else "dst_computer"


def _compute_distribution_stats(
    events: pl.LazyFrame, *, entity_col: str, window_seconds: int, prefix: str
) -> pl.LazyFrame:
    """Entropy of the partner distribution and distinct auth types seen, over
    strictly-prior events (US-110).

    These genuinely need one row per (event, prior event) pair — an entropy is
    not a count. Nothing downstream reads them, so they are opt-in: at LANL
    scale this expansion is 34.6 billion rows for a single day.
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
        .agg(shannon_entropy_expr("_n").alias(f"{prefix}_dst_entropy"))
    )

    auth_type_counts = joined.group_by("_current_id").agg(
        pl.col("auth_type").n_unique().alias(f"{prefix}_n_distinct_auth_type")
    )

    result = dst_counts.join(auth_type_counts, on="_current_id", how="full", coalesce=True)
    return result.rename({"_current_id": "event_id"}).with_columns(
        [
            pl.col(f"{prefix}_dst_entropy").fill_null(0.0),
            pl.col(f"{prefix}_n_distinct_auth_type").fill_null(0),
        ]
    )


def compute_f2(
    events: pl.LazyFrame,
    *,
    entities: list[str] | None = None,
    windows_hours: list[int] | None = None,
    diversity_for: list[tuple[str, int]] | None = None,
) -> pl.LazyFrame:
    """Compute the full F2 feature set (all entities x all windows) and join
    everything back onto the original events by `event_id`.

    `diversity_for` lists the (entity, window_hours) pairs whose set-valued
    columns are actually wanted. `None` means all of them, which is what the
    causality tests exercise and what a small sample can afford. The full
    pipeline passes `DEFAULT_DIVERSITY_PAIRS` instead, because exactly one of
    the nine pairs is consumed downstream and the other eight were each paying
    for a self-join whose output nothing read.
    """
    entities = entities or DEFAULT_ENTITIES
    windows_hours = windows_hours or DEFAULT_WINDOWS_HOURS
    wanted = None if diversity_for is None else {tuple(pair) for pair in diversity_for}

    base = events.with_columns((~pl.col("success")).cast(pl.Int32).alias("_is_failure"))

    result = events
    for entity_col in entities:
        for window_hours in windows_hours:
            feat = compute_f2_for_entity_window(
                base,
                entity_col=entity_col,
                window_hours=window_hours,
                with_diversity=wanted is None or (entity_col, window_hours) in wanted,
            )
            result = result.join(feat, on="event_id", how="left")

    return result
