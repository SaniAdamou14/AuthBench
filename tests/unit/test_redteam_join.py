from __future__ import annotations

import polars as pl

from authbench.label.redteam_join import (
    attach_campaign_id,
    group_into_campaigns,
    label_auth_events,
)

AUTH_SCHEMA = {
    "event_id": pl.UInt64,
    "time": pl.Int32,
    "day": pl.Int16,
    "src_user": pl.Categorical,
    "src_domain": pl.Categorical,
    "dst_user": pl.Categorical,
    "dst_domain": pl.Categorical,
    "src_computer": pl.Categorical,
    "dst_computer": pl.Categorical,
}

REDTEAM_SCHEMA = {
    "time": pl.Int32,
    "day": pl.Int16,
    "user": pl.Categorical,
    "domain": pl.Categorical,
    "src_computer": pl.Categorical,
    "dst_computer": pl.Categorical,
}


def test_label_auth_events_exact_quadruplet_match() -> None:
    auth = pl.LazyFrame(
        [
            {
                "event_id": 0,
                "time": 100,
                "day": 0,
                "src_user": "alice",
                "src_domain": "DOM",
                "dst_user": "alice",
                "dst_domain": "DOM",
                "src_computer": "A",
                "dst_computer": "B",
            },
            {
                "event_id": 1,
                "time": 100,
                "day": 0,
                "src_user": "alice",
                "src_domain": "DOM",
                "dst_user": "alice",
                "dst_domain": "DOM",
                "src_computer": "A",
                "dst_computer": "C",
            },
        ],
        schema=AUTH_SCHEMA,
    )
    redteam = pl.LazyFrame(
        [
            {
                "time": 100,
                "day": 0,
                "user": "alice",
                "domain": "DOM",
                "src_computer": "A",
                "dst_computer": "B",
            }
        ],
        schema=REDTEAM_SCHEMA,
    )

    labeled = label_auth_events(auth, redteam).collect()
    malicious = dict(
        zip(labeled["event_id"].to_list(), labeled["is_malicious"].to_list(), strict=True)
    )

    # Only the exact quadruplet match (event 0, dst_computer=B) is malicious;
    # event 1 differs only by dst_computer and must NOT inflate the positive count.
    assert malicious[0] is True
    assert malicious[1] is False


def test_group_into_campaigns_splits_on_gap() -> None:
    redteam = pl.LazyFrame(
        [
            {
                "time": 0,
                "day": 0,
                "user": "alice",
                "domain": "DOM",
                "src_computer": "A",
                "dst_computer": "B",
            },
            {
                "time": 3600,
                "day": 0,
                "user": "alice",
                "domain": "DOM",
                "src_computer": "B",
                "dst_computer": "C",
            },
            # Same user, but 30 hours later -> a new campaign under a 24h gap threshold.
            {
                "time": 3600 + 30 * 3600,
                "day": 1,
                "user": "alice",
                "domain": "DOM",
                "src_computer": "C",
                "dst_computer": "D",
            },
        ],
        schema=REDTEAM_SCHEMA,
    )

    campaigns = group_into_campaigns(redteam, gap_hours=24.0).collect().sort("time")
    campaign_ids = campaigns["campaign_id"].to_list()

    assert campaign_ids[0] == campaign_ids[1]  # within the gap threshold
    assert campaign_ids[2] != campaign_ids[1]  # beyond the gap threshold


def test_attach_campaign_id_leaves_benign_events_null() -> None:
    auth = pl.LazyFrame(
        [
            {
                "event_id": 0,
                "time": 0,
                "day": 0,
                "src_user": "alice",
                "src_domain": "DOM",
                "dst_user": "alice",
                "dst_domain": "DOM",
                "src_computer": "A",
                "dst_computer": "B",
            },
            {
                "event_id": 1,
                "time": 500,
                "day": 0,
                "src_user": "bob",
                "src_domain": "DOM",
                "dst_user": "bob",
                "dst_domain": "DOM",
                "src_computer": "X",
                "dst_computer": "Y",
            },
        ],
        schema=AUTH_SCHEMA,
    )
    redteam = pl.LazyFrame(
        [
            {
                "time": 0,
                "day": 0,
                "user": "alice",
                "domain": "DOM",
                "src_computer": "A",
                "dst_computer": "B",
            }
        ],
        schema=REDTEAM_SCHEMA,
    )
    campaigns = group_into_campaigns(redteam, gap_hours=24.0)
    result = attach_campaign_id(auth, campaigns).collect()

    by_id = dict(zip(result["event_id"].to_list(), result["campaign_id"].to_list(), strict=True))
    assert by_id[0] is not None
    assert by_id[1] is None
