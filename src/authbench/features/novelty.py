"""F3 — novelty and rarity of the (user, destination) pair (US-111).

Empirically the family carrying most of the lateral-movement signal (spec
section 4.1). `pair_is_new` marks the *first-ever* occurrence of a
(src_user, dst_computer) pair — and, by construction, is exactly the event
that both "a new host was reached by this user" and "a new user reached this
host" refer to, so `user_new_host_count_24h` and `host_new_user_count_24h`
are both derived as a causal rolling sum of the same flag, per entity.
"""

from __future__ import annotations

import polars as pl

from authbench.features.causal import causal_rolling_sum, sort_for_causal

NEW_PAIR_SENTINEL_DAYS = float("inf")


def compute_f3(events: pl.LazyFrame) -> pl.LazyFrame:
    """Add `pair_is_new`, `days_since_pair_last_seen`, `pair_global_rarity`,
    `user_new_host_count_24h`, `host_new_user_count_24h`.

    All quantities are causal: they depend only on events strictly before the
    current one (US-108).
    """
    with_pair = events.with_columns(
        pl.concat_str(
            [pl.col("src_user").cast(pl.Utf8), pl.col("dst_computer").cast(pl.Utf8)], separator="::"
        ).alias("_pair_key")
    )

    by_pair = with_pair.sort(["_pair_key", "time"]).with_columns(
        [
            pl.int_range(0, pl.len()).over("_pair_key").alias("_pair_occurrence_index"),
            pl.col("time").shift(1).over("_pair_key").alias("_pair_prev_time"),
        ]
    )

    by_time = by_pair.sort("time").with_columns(
        pl.int_range(0, pl.len()).alias("_global_occurrence_index")
    )

    with_novelty = by_time.with_columns(
        [
            (pl.col("_pair_occurrence_index") == 0).alias("pair_is_new"),
            pl.when(pl.col("_pair_occurrence_index") == 0)
            .then(pl.lit(NEW_PAIR_SENTINEL_DAYS))
            .otherwise((pl.col("time") - pl.col("_pair_prev_time")) / 86_400.0)
            .alias("days_since_pair_last_seen"),
            (
                -(
                    (pl.col("_pair_occurrence_index") + 1)
                    / (pl.col("_global_occurrence_index") + 1)
                ).log()
            ).alias("pair_global_rarity"),
        ]
    )

    novelty_base = with_novelty.select(
        [
            "event_id",
            "pair_is_new",
            "days_since_pair_last_seen",
            "pair_global_rarity",
        ]
    ).with_columns(pl.col("pair_is_new").cast(pl.Int32).alias("_pair_is_new_int"))

    # Rolling sums each require their own [entity, time] sort (US-108) — computed
    # on dedicated small frames and joined back onto the original row order by
    # event_id, exactly as the F2 primitives do.
    joined_flag = events.select(["event_id", "src_user", "dst_computer", "time"]).join(
        novelty_base.select(["event_id", "_pair_is_new_int"]), on="event_id"
    )

    by_user = sort_for_causal(joined_flag, "src_user", "time")
    user_new_host = by_user.select(
        [
            "event_id",
            causal_rolling_sum(
                by_user,
                entity_col="src_user",
                time_col="time",
                value_col="_pair_is_new_int",
                window_seconds=24 * 3600,
            ).alias("user_new_host_count_24h"),
        ]
    )

    by_host = sort_for_causal(joined_flag, "dst_computer", "time")
    host_new_user = by_host.select(
        [
            "event_id",
            causal_rolling_sum(
                by_host,
                entity_col="dst_computer",
                time_col="time",
                value_col="_pair_is_new_int",
                window_seconds=24 * 3600,
            ).alias("host_new_user_count_24h"),
        ]
    )

    result = (
        events.join(novelty_base.drop("_pair_is_new_int"), on="event_id", how="left")
        .join(user_new_host, on="event_id", how="left")
        .join(host_new_user, on="event_id", how="left")
    )

    return result
