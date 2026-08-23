"""The one shared causal-window primitive (US-108).

Every windowed feature in F2/F3/F5 goes through this module. No feature
module is allowed to write its own ad hoc rolling aggregation — that
restriction is enforced by code review and by `tests/unit/test_causality.py`,
which builds a synthetic event whose feature value would change if the
future were visible, for every feature that claims to be causal.

`closed="left"` on Polars' `rolling_*_by` means the window is
`[t - window, t)`: it includes everything from `window` seconds ago up to,
but never including, the current event's own timestamp. That is what makes
these primitives causal by construction rather than by convention.
"""

from __future__ import annotations

import polars as pl


def sort_for_causal(df: pl.LazyFrame, entity_col: str, time_col: str) -> pl.LazyFrame:
    """Polars' `rolling_*_by` requires ordering by `time_col` within each
    `entity_col` group. Every causal primitive below assumes its input has
    already been sorted this way.
    """
    return df.sort([entity_col, time_col])


def causal_rolling_count(
    df: pl.LazyFrame, *, entity_col: str, time_col: str, window_seconds: int
) -> pl.Expr:
    """Number of prior events of the same entity strictly within the window."""
    return (
        pl.repeat(1, pl.len(), dtype=pl.Int32)
        .rolling_sum_by(time_col, window_size=f"{window_seconds}i", closed="left")
        .over(entity_col)
        .fill_null(0)
    )


def causal_rolling_sum(
    df: pl.LazyFrame,
    *,
    entity_col: str,
    time_col: str,
    value_col: str,
    window_seconds: int,
) -> pl.Expr:
    """Sum of `value_col` over prior events of the same entity within the window."""
    return (
        pl.col(value_col)
        .cast(pl.Int32)
        .rolling_sum_by(time_col, window_size=f"{window_seconds}i", closed="left")
        .over(entity_col)
        .fill_null(0)
    )


def causal_prior_events(
    events: pl.LazyFrame,
    *,
    entity_col: str,
    time_col: str,
    id_col: str,
    carry_cols: list[str],
    window_seconds: int,
    namespace: str = "",
) -> pl.LazyFrame:
    """Self-join giving one row per (current event, strictly-prior event of the
    same entity within the window) pair.

    This is the primitive behind features that a running sum can't express —
    distinct-destination counts, entropy of a destination distribution, and
    similar set/distribution statistics (F2 diversity, F3 novelty).

    `namespace` must be unique per call site when several `causal_prior_events`
    calls feed into the same overall lazy query (e.g. one per entity/window
    pair in F2): Polars' query planner resolves the whole graph together, and
    reusing identical temporary column names across sibling subqueries has
    been observed to trip its duplicate-projection check even though each
    subquery is independently well-formed.

    Cost scales with events-per-entity-per-window, which is fine at demo/LANL
    scale for reasonably-sized windows; a full 1B-row production run should
    substitute a DuckDB windowed query with approximate (HyperLogLog) distinct
    counts for the 7-day window instead of this exact self-join.
    """
    t_current, t_prior = f"_{namespace}_t_current", f"_{namespace}_t_prior"
    current_id, prior_id = f"_{namespace}_current_id", f"_{namespace}_prior_id"

    current = events.select([entity_col, time_col, id_col]).rename(
        {time_col: t_current, id_col: current_id}
    )
    prior = events.select([entity_col, time_col, id_col, *carry_cols]).rename(
        {time_col: t_prior, id_col: prior_id}
    )

    joined = current.join(prior, on=entity_col, how="left").filter(
        (pl.col(t_prior) < pl.col(t_current))
        & (pl.col(t_prior) >= pl.col(t_current) - window_seconds)
    )
    return joined.rename({current_id: "_current_id", prior_id: "_prior_id"})


def causal_distinct_count(
    events: pl.LazyFrame,
    *,
    entity_col: str,
    partner_col: str,
    time_col: str,
    id_col: str,
    window_seconds: int,
    out_col: str,
) -> pl.LazyFrame:
    """Distinct `partner_col` values seen by `entity_col` over the strictly
    prior `window_seconds` — exactly, in O(n log n), without a self-join.

    `causal_prior_events` answers this by materializing one row per (event,
    prior event of the same entity in the window) pair. Measured on LANL day 0
    with a one-hour window on `src_user`: 34.6 **billion** pairs for 15.7
    million events, because a single busy machine account can log 33,489 times
    in one hour and contributes the square of that on its own. Extrapolated to
    an eight-day training split it is some 277 billion pairs, several terabytes
    of intermediate data — the stage does not run, at any memory budget.

    The reformulation. An event `e` is counted in the window ending at query
    time `T` exactly when it is the *first* occurrence of its partner inside
    that window, which is true for a contiguous range of `T` and no other:

        t_e in [T - W, T)  and  prev_same_pair(e) < T - W
          <=>  T in ( max(t_e, prev_e + W),  t_e + W ]

    Every distinct partner present in the window has exactly one such event, so
    the distinct count *is* the number of these intervals covering `T`. Counting
    intervals that cover a point is a sweep: +1 where each opens, -1 where it
    closes, sorted, cumulatively summed. Exact — not an approximation, not a
    sketch — and it never builds a pair.
    """
    base = events.select([id_col, entity_col, partner_col, time_col])

    with_prev = base.sort([entity_col, partner_col, time_col]).with_columns(
        pl.col(time_col).shift(1).over([entity_col, partner_col]).alias("_prev_pair_time")
    )
    bounds = with_prev.select(
        [
            entity_col,
            # Opens at the later of "the event happened" and "the previous
            # sighting of this partner fell out of the window". A first-ever
            # pair has no previous sighting, so it opens when the event happens.
            pl.max_horizontal(
                pl.col(time_col),
                pl.col("_prev_pair_time") + window_seconds,
            )
            .fill_null(pl.col(time_col))
            .alias("_open"),
            (pl.col(time_col) + window_seconds).alias("_close"),
        ]
    )

    # `_kind` orders ties: queries (0) are read *before* any delta at the same
    # timestamp, which is what makes the counts strict (`< T`, never `<= T`)
    # and therefore keeps the window half-open exactly as `closed="left"` does
    # for the rolling primitives above.
    opens = bounds.select(
        [pl.col(entity_col), pl.col("_open").alias("_t"), pl.lit(1, dtype=pl.Int32).alias("_delta")]
    )
    closes = bounds.select(
        [
            pl.col(entity_col),
            pl.col("_close").alias("_t"),
            pl.lit(-1, dtype=pl.Int32).alias("_delta"),
        ]
    )
    queries = base.select(
        [
            pl.col(entity_col),
            pl.col(time_col).alias("_t"),
            pl.lit(0, dtype=pl.Int32).alias("_delta"),
            pl.col(id_col),
        ]
    )

    deltas = pl.concat([opens, closes], how="vertical").with_columns(
        pl.lit(1, dtype=pl.Int8).alias("_kind"),
        pl.lit(None, dtype=base.collect_schema()[id_col]).alias(id_col),
    )
    sweep = (
        pl.concat(
            [queries.with_columns(pl.lit(0, dtype=pl.Int8).alias("_kind")), deltas], how="diagonal"
        )
        .sort([entity_col, "_t", "_kind"])
        .with_columns(pl.col("_delta").cum_sum().over(entity_col).alias(out_col))
    )

    return sweep.filter(pl.col("_kind") == 0).select([id_col, pl.col(out_col).cast(pl.Int32)])


def causal_first_occurrence(
    events: pl.LazyFrame, *, group_cols: list[str], time_col: str, id_col: str
) -> pl.LazyFrame:
    """Return `id_col` plus `is_new` (first-ever occurrence of this `group_cols`
    combination) and `occurrence_index` (count of strictly-prior occurrences,
    0 for the first). The primitive behind F3's pair novelty and any rule that
    needs "has this user ever done X before?" (e.g. M1's R6).
    """
    key = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in group_cols], separator="::").alias(
        "_group_key"
    )
    with_key = events.select([id_col, time_col, *group_cols]).with_columns(key)
    sorted_ = with_key.sort(["_group_key", time_col])
    with_idx = sorted_.with_columns(
        [
            pl.int_range(0, pl.len()).over("_group_key").alias("_occurrence_index"),
            pl.col(time_col).shift(1).over("_group_key").alias("_prev_time"),
        ]
    )
    return with_idx.select(
        [
            id_col,
            (pl.col("_occurrence_index") == 0).alias("is_new"),
            pl.col("_occurrence_index").alias("occurrence_index"),
            (pl.col(time_col) - pl.col("_prev_time")).alias("seconds_since_prev"),
        ]
    )


def shannon_entropy_expr(count_col: str) -> pl.Expr:
    """Shannon entropy (nats) of a distribution given per-category counts in `count_col`."""
    total = pl.col(count_col).sum()
    p = pl.col(count_col) / total
    return (-(p * p.log()).sum()).fill_nan(0.0)
