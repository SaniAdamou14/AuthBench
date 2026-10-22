"""F3: novelty and rarity of the user-destination, user-source and source-destination relations (US-111).

`*_is_new` marks the first occurrence of a relation within the window a
partition reads. `*_rarity` is -log((k + 1) / (n + 1)), with k the earlier
occurrences of the relation and n the events that condition it (all events
for the user-destination pair, the user's events for the user-source pair, the
source's events for the source-destination edge) counted up to the start of the
event's day. Fixing the denominator for the day means two equally novel events
of one day score equally: their order is left to the stated tie-break instead of
to the time at which they happened.

The source side carries the signal most exploited on LANL: in the evaluated
test partition, 97.5% of the red-team events share one source computer.
"""

from __future__ import annotations

import polars as pl

from authbench.features.causal import causal_rolling_sum, sort_for_causal

NEW_PAIR_SENTINEL_DAYS = float("inf")

NOVELTY_COLUMNS = (
    "pair_is_new",
    "days_since_pair_last_seen",
    "pair_global_rarity",
    "user_src_is_new",
    "user_src_rarity",
    "src_dst_is_new",
    "src_dst_rarity",
)


def _key(*columns: str) -> pl.Expr:
    return pl.concat_str([pl.col(c).cast(pl.Utf8) for c in columns], separator="::")


def _occurrences(events: pl.LazyFrame, columns: tuple[str, ...], prefix: str) -> pl.LazyFrame:
    """Earlier occurrences of the relation and the time of the last one, per event."""
    return (
        events.select("event_id", "time", _key(*columns).alias("_key"))
        .sort(["_key", "time", "event_id"])
        .select(
            "event_id",
            pl.int_range(0, pl.len()).over("_key").alias(f"_{prefix}_k"),
            pl.col("time").shift(1).over("_key").alias(f"_{prefix}_prev_time"),
        )
    )


def _before_day(events: pl.LazyFrame, columns: tuple[str, ...], alias: str) -> pl.LazyFrame:
    """Events of the conditioning entity on days before each event's day (all events if no columns)."""
    keyed = events.select(
        "event_id", "day", (_key(*columns) if columns else pl.lit("")).alias("_key")
    )
    per_day = (
        keyed.group_by("_key", "day")
        .agg(pl.len().alias("_n"))
        .sort("_key", "day")
        .with_columns((pl.col("_n").cum_sum().over("_key") - pl.col("_n")).alias(alias))
        .drop("_n")
    )
    return keyed.join(per_day, on=["_key", "day"], how="left").select("event_id", alias)


def _rarity(k: str, n: str) -> pl.Expr:
    return -((pl.col(k) + 1) / (pl.col(n) + 1)).log()


def compute_f3(events: pl.LazyFrame) -> pl.LazyFrame:
    """Add `NOVELTY_COLUMNS`, `user_new_host_count_24h` and `host_new_user_count_24h`.

    Every quantity depends only on events strictly before the current one in
    (time, event_id) order, or on days strictly before the current day (US-108).
    """
    # Seven branches read `events`; without the cache each re-executes the whole upstream plan.
    events = events.cache()
    relations = {
        "pair": ("src_user", "dst_computer"),
        "user_src": ("src_user", "src_computer"),
        "src_dst": ("src_computer", "dst_computer"),
    }
    conditioning = {"pair": (), "user_src": ("src_user",), "src_dst": ("src_computer",)}

    novelty = events.select("event_id", "time")
    for prefix, columns in relations.items():
        novelty = novelty.join(
            _occurrences(events, columns, prefix), on="event_id", how="left"
        ).join(_before_day(events, conditioning[prefix], f"_{prefix}_n"), on="event_id", how="left")

    novelty = novelty.select(
        "event_id",
        (pl.col("_pair_k") == 0).alias("pair_is_new"),
        pl.when(pl.col("_pair_k") == 0)
        .then(pl.lit(NEW_PAIR_SENTINEL_DAYS))
        .otherwise((pl.col("time") - pl.col("_pair_prev_time")) / 86_400.0)
        .alias("days_since_pair_last_seen"),
        _rarity("_pair_k", "_pair_n").alias("pair_global_rarity"),
        (pl.col("_user_src_k") == 0).alias("user_src_is_new"),
        _rarity("_user_src_k", "_user_src_n").alias("user_src_rarity"),
        (pl.col("_src_dst_k") == 0).alias("src_dst_is_new"),
        _rarity("_src_dst_k", "_src_dst_n").alias("src_dst_rarity"),
    )

    # Rolling sums each need their own [entity, time] sort (US-108), joined back by event_id.
    flagged = events.select("event_id", "src_user", "dst_computer", "time").join(
        novelty.select("event_id", pl.col("pair_is_new").cast(pl.Int32).alias("_new")),
        on="event_id",
    )
    by_user = sort_for_causal(flagged, "src_user", "time")
    user_new_host = by_user.select(
        "event_id",
        causal_rolling_sum(
            by_user, entity_col="src_user", time_col="time", value_col="_new", window_seconds=86_400
        ).alias("user_new_host_count_24h"),
    )
    by_host = sort_for_causal(flagged, "dst_computer", "time")
    host_new_user = by_host.select(
        "event_id",
        causal_rolling_sum(
            by_host,
            entity_col="dst_computer",
            time_col="time",
            value_col="_new",
            window_seconds=86_400,
        ).alias("host_new_user_count_24h"),
    )

    return (
        events.join(novelty, on="event_id", how="left")
        .join(user_new_host, on="event_id", how="left")
        .join(host_new_user, on="event_id", how="left")
    )
