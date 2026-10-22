"""Day-by-day featurisation with carried state, equal to featurising the whole window at once.

Memory is bounded by one day and its predecessor instead of the whole window,
which is what lets a 58-day run fit on a modest machine. Three kinds of feature
need three treatments:

- row-local (F1, the hour encoding): computed on the day alone;
- windows of at most 24 hours (F2, the 24-hour novelty counts): computed on the
  previous day followed by the current one, keeping the current one;
- cumulative from the window start (F3 relations, the F4 profile): computed from
  counts and sums carried across days. The carried circular sums enter the
  sequential cumulative sum as its first term, so they add in the same order as
  in the whole-window computation and the results are equal bit for bit.

`tests/unit/test_incremental.py` asserts that equality against `featurize`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from authbench.features.causal import causal_rolling_sum, sort_for_causal
from authbench.features.event import FrequencyEncoding, compute_f1
from authbench.features.history import (
    DEFAULT_DIVERSITY_PAIRS,
    DEFAULT_F2_ENTITIES,
    DEFAULT_F2_WINDOWS_HOURS,
    compute_f2,
)
from authbench.features.novelty import NEW_PAIR_SENTINEL_DAYS, _key, _rarity
from authbench.features.temporal import (
    PROFILE_HELPERS,
    NightWindow,
    profile_columns,
    with_hour_angles,
)

# prefix -> (relation columns, conditioning columns or None when the relation has no rarity)
RELATIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...] | None]] = {
    "pair": (("src_user", "dst_computer"), ()),
    "user_src": (("src_user", "src_computer"), ("src_user",)),
    "src_dst": (("src_computer", "dst_computer"), ("src_computer",)),
    "user_auth": (("src_user", "auth_type"), None),
}
CONTEXT_COLUMNS = ["event_id", "time", "day", "src_user", "dst_computer", "pair_is_new"]


@dataclass
class FeatureState:
    """What one day hands to the next."""

    counts: dict[str, pl.DataFrame] = field(default_factory=dict)
    conditioning: dict[str, pl.DataFrame] = field(default_factory=dict)
    profile: pl.DataFrame | None = None
    previous_day: pl.DataFrame | None = None


def _merge_counts(old: pl.DataFrame | None, new: pl.DataFrame, value: str) -> pl.DataFrame:
    if old is None:
        return new
    return pl.concat([old, new]).group_by("_key").agg(pl.col(value).sum())


def _novelty(day: pl.DataFrame, state: FeatureState) -> tuple[pl.DataFrame, FeatureState]:
    out = day.select("event_id", "time")
    counts, conditioning = dict(state.counts), dict(state.conditioning)
    for prefix, (columns, cond) in RELATIONS.items():
        keyed = (
            day.select("event_id", "time", _key(*columns).alias("_key"))
            .sort("_key", "time", "event_id")
            .with_columns(
                pl.int_range(0, pl.len()).over("_key").alias("_i"),
                pl.col("time").shift(1).over("_key").alias("_prev"),
            )
        )
        carried = counts.get(prefix)
        if carried is not None:
            keyed = keyed.join(carried, on="_key", how="left")
        else:
            keyed = keyed.with_columns(
                pl.lit(None, pl.Int64).alias("_c"), pl.lit(None, pl.Int64).alias("_last")
            )
        keyed = keyed.select(
            "event_id",
            (pl.col("_i") + pl.col("_c").fill_null(0)).alias(f"_{prefix}_k"),
            pl.when(pl.col("_i") == 0)
            .then(pl.col("_last"))
            .otherwise(pl.col("_prev"))
            .alias(f"_{prefix}_prev_time"),
        )
        out = out.join(keyed, on="event_id", how="left")
        day_counts = (
            day.select(_key(*columns).alias("_key"), "time")
            .group_by("_key")
            .agg(
                pl.len().cast(pl.Int64).alias("_c"),
                pl.col("time").max().cast(pl.Int64).alias("_last"),
            )
        )
        counts[prefix] = (
            day_counts
            if carried is None
            else pl.concat([carried, day_counts])
            .group_by("_key")
            .agg(pl.col("_c").sum(), pl.col("_last").max())
        )
        if cond is None:
            continue
        cond_key = (_key(*cond) if cond else pl.lit("")).alias("_key")
        before = conditioning.get(prefix)
        n = day.select("event_id", cond_key)
        n = (
            n.join(before, on="_key", how="left")
            if before is not None
            else n.with_columns(pl.lit(None, pl.Int64).alias("_n"))
        )
        out = out.join(
            n.select("event_id", pl.col("_n").fill_null(0).alias(f"_{prefix}_n")),
            on="event_id",
            how="left",
        )
        # With event_id alongside, a constant key is broadcast to every row instead of one.
        day_n = (
            day.select("event_id", cond_key)
            .group_by("_key")
            .agg(pl.len().cast(pl.Int64).alias("_n"))
        )
        conditioning[prefix] = _merge_counts(before, day_n, "_n")

    features = out.select(
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
        (pl.col("_user_auth_k") == 0).alias("user_auth_is_new"),
    )
    return features, FeatureState(counts, conditioning, state.profile, state.previous_day)


def _new_pair_counts(
    day: pl.DataFrame, novelty: pl.DataFrame, previous: pl.DataFrame | None
) -> pl.DataFrame:
    """The 24-hour rolling counts of new pairs per user and per host, over the previous day and this one."""
    current = day.select("event_id", "time", "day", "src_user", "dst_computer").join(
        novelty.select("event_id", "pair_is_new"), on="event_id"
    )
    frames = [current] if previous is None else [previous.select(CONTEXT_COLUMNS), current]
    flagged = (
        pl.concat(frames).with_columns(pl.col("pair_is_new").cast(pl.Int32).alias("_new")).lazy()
    )
    by_user = sort_for_causal(flagged, "src_user", "time")
    by_host = sort_for_causal(flagged, "dst_computer", "time")
    user = by_user.select(
        "event_id",
        causal_rolling_sum(
            by_user, entity_col="src_user", time_col="time", value_col="_new", window_seconds=86_400
        ).alias("user_new_host_count_24h"),
    )
    host = by_host.select(
        "event_id",
        causal_rolling_sum(
            by_host,
            entity_col="dst_computer",
            time_col="time",
            value_col="_new",
            window_seconds=86_400,
        ).alias("host_new_user_count_24h"),
    )
    ids = current.select("event_id").lazy()
    return ids.join(user, on="event_id", how="left").join(host, on="event_id", how="left").collect()


def _profile(
    day: pl.DataFrame, night_window: NightWindow, carried: pl.DataFrame | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """F4 on one day, the user's carried sums entering each cumulative sum as its first term."""
    angled = with_hour_angles(day.lazy(), night_window).collect()
    rows = angled.select(
        "event_id",
        "time",
        "src_user",
        "src_computer",
        "dst_computer",
        "_angle",
        "hour_sin",
        "hour_cos",
        pl.lit(False).alias("_carried"),
    )
    if carried is not None:
        seeds = carried.select(
            pl.lit(None, rows.schema["event_id"]).alias("event_id"),
            pl.col("last_time").cast(rows.schema["time"]).alias("time"),
            pl.col("src_user").cast(rows.schema["src_user"]),
            pl.lit(None, rows.schema["src_computer"]).alias("src_computer"),
            pl.col("last_dst").cast(rows.schema["dst_computer"]).alias("dst_computer"),
            pl.lit(None, pl.Float64).alias("_angle"),
            pl.col("cum_sin").alias("hour_sin"),
            pl.col("cum_cos").alias("hour_cos"),
            pl.lit(True).alias("_carried"),
        ).join(rows.select("src_user").unique(), on="src_user", how="semi")
        rows = pl.concat([seeds, rows])
    ordered = (
        rows.sort("src_user", pl.col("_carried").not_(), "time", "event_id")
        .with_columns(
            pl.col("hour_sin").cum_sum().over("src_user").alias("_cum_sin"),
            pl.col("hour_cos").cum_sum().over("src_user").alias("_cum_cos"),
        )
        .with_columns(
            pl.col("_cum_sin").shift(1).over("src_user").alias("_cum_sin_prior"),
            pl.col("_cum_cos").shift(1).over("src_user").alias("_cum_cos_prior"),
            pl.col("time").shift(1).over("src_user").alias("_prev_time_same_user"),
            pl.col("dst_computer").shift(1).over("src_user").alias("_prev_dst_same_user"),
        )
    )
    real = ordered.filter(~pl.col("_carried"))
    features = real.with_columns(profile_columns()).select(
        "event_id",
        "hour_deviation_from_profile",
        "hours_since_prev_event_same_user",
        "prev_dst_is_src",
    )
    last = ordered.group_by("src_user", maintain_order=True).agg(
        pl.col("_cum_sin").last().alias("cum_sin"),
        pl.col("_cum_cos").last().alias("cum_cos"),
        pl.col("time").last().alias("last_time"),
        pl.col("dst_computer").last().alias("last_dst"),
    )
    if carried is not None:
        untouched = carried.join(last.select("src_user"), on="src_user", how="anti")
        last = pl.concat([untouched, last.cast(untouched.schema)])
    hours = angled.select("event_id", "hour_sin", "hour_cos", "is_night_window")
    return hours.join(features, on="event_id", how="left"), last


def featurize_day(
    day: pl.DataFrame,
    state: FeatureState,
    night_window: NightWindow,
    frequency_encoding: FrequencyEncoding,
) -> tuple[pl.DataFrame, FeatureState]:
    """Features of one day's labelled events, and the state the next day needs."""
    f1 = compute_f1(day.lazy(), frequency_encoding).collect()
    previous = state.previous_day
    window = day if previous is None else pl.concat([previous.select(day.columns), day])
    f2_full = (
        compute_f2(
            window.lazy(),
            entities=DEFAULT_F2_ENTITIES,
            windows_hours=DEFAULT_F2_WINDOWS_HOURS,
            diversity_for=DEFAULT_DIVERSITY_PAIRS,
        )
        .join(day.select("event_id").lazy(), on="event_id", how="semi")
        .collect()
    )
    f2 = f2_full.select("event_id", *[c for c in f2_full.columns if c not in day.columns])
    novelty, state = _novelty(day, state)
    rolling = _new_pair_counts(day, novelty, previous)
    temporal, profile = _profile(day, night_window, state.profile)

    features = (
        f1.join(f2, on="event_id", how="left")
        .join(novelty, on="event_id", how="left")
        .join(rolling, on="event_id", how="left")
        .join(temporal, on="event_id", how="left")
    )
    context = day.join(novelty.select("event_id", "pair_is_new"), on="event_id", how="left")
    return features, FeatureState(state.counts, state.conditioning, profile, context)


def featurize_window(
    days: list[pl.DataFrame],
    night_window: NightWindow,
    frequency_encoding: FrequencyEncoding,
) -> pl.DataFrame:
    """Featurise consecutive days in order from an empty state, concatenated: the reference for tests."""
    state, parts = FeatureState(), []
    for day in days:
        features, state = featurize_day(day, state, night_window, frequency_encoding)
        parts.append(features)
    return pl.concat(parts)


__all__ = ["PROFILE_HELPERS", "FeatureState", "featurize_day", "featurize_window"]
