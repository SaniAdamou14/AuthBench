"""Exact-quadruplet labeling and campaign grouping (US-105, US-106).

The join key is (`time`, `src_user`, `src_domain`, `src_computer`,
`dst_computer`) — matching the redteam quadruplet (`time`, `user`, `domain`,
`src_computer`, `dst_computer`) against the *source* identity of the auth
event, never against time alone. A looser join inflates the positive count
and makes results incomparable to the published literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class LabelingReport:
    n_redteam_events: int
    n_matched_in_auth: int
    positive_rate: float
    n_campaigns: int


def label_auth_events(auth: pl.LazyFrame, redteam: pl.LazyFrame) -> pl.LazyFrame:
    """Add a boolean `is_malicious` column to `auth` via an exact quadruplet join.

    `auth` must carry the typed schema from `parse.clean.clean_auth`;
    `redteam` the typed schema from `parse.clean.clean_redteam`.
    """
    redteam_flag = redteam.select(
        [
            pl.col("time"),
            pl.col("user").alias("src_user"),
            pl.col("domain").alias("src_domain"),
            pl.col("src_computer"),
            pl.col("dst_computer"),
            pl.lit(True).alias("is_malicious"),
        ]
    )

    labeled = auth.join(
        redteam_flag,
        on=["time", "src_user", "src_domain", "src_computer", "dst_computer"],
        how="left",
    ).with_columns(pl.col("is_malicious").fill_null(False))

    return labeled


def labeling_report(
    labeled: pl.LazyFrame, redteam: pl.LazyFrame, n_campaigns: int
) -> LabelingReport:
    n_redteam = redteam.select(pl.len()).collect().item()
    matched = labeled.filter(pl.col("is_malicious")).select(pl.len()).collect().item()
    n_total = labeled.select(pl.len()).collect().item()

    return LabelingReport(
        n_redteam_events=n_redteam,
        n_matched_in_auth=matched,
        positive_rate=(matched / n_total if n_total else 0.0),
        n_campaigns=n_campaigns,
    )


def group_into_campaigns(redteam: pl.LazyFrame, gap_hours: float = 24.0) -> pl.LazyFrame:
    """Group red-team events into campaigns per `user@domain`.

    A new campaign starts whenever the gap since that user's previous
    red-team event exceeds `gap_hours`. Returns `redteam` with two added
    columns: `campaign_id` (a dense integer, unique across all users) and
    `user_at_domain` (for joining back).

    This grouping has no canonical version in the literature (section 4.2) —
    the gap threshold is a first-class, published, sensitivity-tested
    parameter, never a hidden constant.
    """
    gap_seconds = gap_hours * 3600.0

    with_key = redteam.with_columns(
        (pl.col("user").cast(pl.Utf8) + "@" + pl.col("domain").cast(pl.Utf8)).alias(
            "user_at_domain"
        )
    ).sort(["user_at_domain", "time"])

    with_gap = with_key.with_columns(
        (pl.col("time") - pl.col("time").shift(1).over("user_at_domain")).alias("gap_since_prev")
    )

    with_break = with_gap.with_columns(
        (pl.col("gap_since_prev").is_null() | (pl.col("gap_since_prev") > gap_seconds)).alias(
            "is_new_campaign"
        )
    )

    with_local_id = with_break.with_columns(
        pl.col("is_new_campaign").cum_sum().over("user_at_domain").alias("local_campaign_id")
    )

    # Make campaign ids globally unique by combining the per-user local id with
    # a dense rank of the user themselves.
    with_user_rank = with_local_id.with_columns(
        pl.col("user_at_domain").rank(method="dense").alias("user_rank")
    )

    result = with_user_rank.with_columns(
        (pl.col("user_rank").cast(pl.Utf8) + "_" + pl.col("local_campaign_id").cast(pl.Utf8))
        .rank(method="dense")
        .cast(pl.Int64)
        .alias("campaign_id")
    ).drop(["gap_since_prev", "is_new_campaign", "local_campaign_id", "user_rank"])

    return result


def attach_campaign_id(labeled_auth: pl.LazyFrame, campaigns: pl.LazyFrame) -> pl.LazyFrame:
    """Join `campaign_id` onto malicious auth events via the same exact quadruplet.

    Benign events get a null `campaign_id`. This is what US-106's evaluation
    (recall *per campaign*, not per event) and the event→campaign CSV artifact
    are built on.
    """
    campaign_keys = campaigns.select(
        [
            pl.col("time"),
            pl.col("user").alias("src_user"),
            pl.col("domain").alias("src_domain"),
            pl.col("src_computer"),
            pl.col("dst_computer"),
            pl.col("campaign_id"),
        ]
    )
    return labeled_auth.join(
        campaign_keys,
        on=["time", "src_user", "src_domain", "src_computer", "dst_computer"],
        how="left",
    )


def campaign_summary(campaigns: pl.LazyFrame) -> pl.DataFrame:
    """Per-campaign duration and event count — published as a reusable artifact (US-106)."""
    return (
        campaigns.group_by("campaign_id")
        .agg(
            [
                pl.col("user_at_domain").first().alias("user_at_domain"),
                pl.len().alias("n_events"),
                pl.col("time").min().alias("start_time"),
                pl.col("time").max().alias("end_time"),
            ]
        )
        .with_columns((pl.col("end_time") - pl.col("start_time")).alias("duration_seconds"))
        .sort("campaign_id")
        .collect()
    )
