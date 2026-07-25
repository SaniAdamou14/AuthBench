"""F4 — temporal features (US-112).

LANL time has no timezone or calendar date: it is a raw second counter
starting at epoch 1. "Hour of day" only exists modulo 86 400, and the
day/night boundary is not assumed — it is calibrated empirically from the
activity trough (spec section 2.1), and that calibration must be fit on the
training split only, never on the full dataset, or it silently leaks the
val/test distribution into a "fixed" feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from authbench.features.causal import sort_for_causal
from authbench.parse.schema import SECONDS_PER_DAY

TWO_PI = 2 * math.pi


@dataclass(frozen=True)
class NightWindow:
    """A calibrated low-activity window, in seconds-of-day, possibly wrapping midnight."""

    start_second: int
    end_second: int

    def contains_expr(self, second_of_day_col: str) -> pl.Expr:
        s, e = self.start_second, self.end_second
        col = pl.col(second_of_day_col)
        if s <= e:
            return col.is_between(s, e, closed="left")
        return (col >= s) | (col < e)


def calibrate_night_window(
    train_events: pl.LazyFrame, *, width_hours: int = 6, n_buckets: int = 96
) -> NightWindow:
    """Find the contiguous `width_hours` window of lowest event volume,
    calibrated on `train_events` only (US-112 / spec 2.1). Never call this on
    val or test data — that would be exactly the kind of "assumed, not
    calibrated" leakage the spec calls out explicitly.
    """
    bucket_seconds = SECONDS_PER_DAY // n_buckets
    counts = (
        train_events.with_columns(
            ((pl.col("time") % SECONDS_PER_DAY) // bucket_seconds).alias("_bucket")
        )
        .group_by("_bucket")
        .agg(pl.len().alias("_n"))
        .collect()
        .sort("_bucket")
    )

    bucket_n = [0] * n_buckets
    for row in counts.iter_rows(named=True):
        bucket_n[int(row["_bucket"])] = row["_n"]

    width_buckets = max(1, round(width_hours * 3600 / bucket_seconds))
    best_start, best_sum = 0, None
    for start in range(n_buckets):
        window_sum = sum(bucket_n[(start + i) % n_buckets] for i in range(width_buckets))
        if best_sum is None or window_sum < best_sum:
            best_sum, best_start = window_sum, start

    start_second = best_start * bucket_seconds
    end_second = (start_second + width_buckets * bucket_seconds) % SECONDS_PER_DAY
    return NightWindow(start_second=start_second, end_second=end_second)


def compute_f4(events: pl.LazyFrame, night_window: NightWindow) -> pl.LazyFrame:
    """Add cyclical hour encoding, per-user circular-mean time-of-day deviation
    (causal, expanding), inter-event delay, and the calibrated night flag.
    """
    with_angle = (
        events.with_columns(
            [
                (pl.col("time") % SECONDS_PER_DAY).alias("_second_of_day"),
            ]
        )
        .with_columns(
            [
                ((pl.col("_second_of_day") / SECONDS_PER_DAY) * TWO_PI).alias("_angle"),
            ]
        )
        .with_columns(
            [
                pl.col("_angle").sin().alias("hour_sin"),
                pl.col("_angle").cos().alias("hour_cos"),
                night_window.contains_expr("_second_of_day").alias("is_night_window"),
            ]
        )
    )

    by_user = sort_for_causal(with_angle, "src_user", "time")
    with_profile = by_user.with_columns(
        [
            pl.col("hour_sin").cum_sum().shift(1).over("src_user").alias("_cum_sin_prior"),
            pl.col("hour_cos").cum_sum().shift(1).over("src_user").alias("_cum_cos_prior"),
            pl.col("time").shift(1).over("src_user").alias("_prev_time_same_user"),
        ]
    )

    with_deviation = with_profile.with_columns(
        [
            pl.arctan2("_cum_sin_prior", "_cum_cos_prior").alias("_profile_angle"),
        ]
    ).with_columns(
        ((pl.col("_angle") - pl.col("_profile_angle") + math.pi) % TWO_PI - math.pi)
        .abs()
        .fill_null(0.0)
        .alias("hour_deviation_from_profile"),
        ((pl.col("time") - pl.col("_prev_time_same_user")) / 3600.0)
        .fill_null(float("inf"))
        .alias("hours_since_prev_event_same_user"),
    )

    return with_deviation.drop(
        [
            "_second_of_day",
            "_angle",
            "_cum_sin_prior",
            "_cum_cos_prior",
            "_prev_time_same_user",
            "_profile_angle",
        ]
    )
