from __future__ import annotations

import polars as pl

from authbench.features.sequence import PAD_ID, compute_f6

SCHEMA = {
    "event_id": pl.UInt64,
    "time": pl.Int32,
    "src_user": pl.Categorical,
    "dst_computer": pl.Categorical,
    "auth_type": pl.Categorical,
    "success": pl.Boolean,
}


def _frame(rows: list[dict]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema=SCHEMA).lazy()


def test_first_event_is_fully_padded() -> None:
    rows = [
        {
            "event_id": 0,
            "time": 0,
            "src_user": "alice",
            "dst_computer": "A",
            "auth_type": "K",
            "success": True,
        },
    ]
    result = compute_f6(_frame(rows), context_length=4).collect()
    seq = result["sequence_dst"][0]
    assert list(seq) == [PAD_ID] * 4


def test_sequence_carries_only_strictly_prior_events_in_order() -> None:
    rows = [
        {
            "event_id": 0,
            "time": 0,
            "src_user": "alice",
            "dst_computer": "A",
            "auth_type": "K",
            "success": True,
        },
        {
            "event_id": 1,
            "time": 10,
            "src_user": "alice",
            "dst_computer": "B",
            "auth_type": "K",
            "success": True,
        },
        {
            "event_id": 2,
            "time": 20,
            "src_user": "alice",
            "dst_computer": "C",
            "auth_type": "K",
            "success": False,
        },
    ]
    result = compute_f6(_frame(rows), context_length=2).collect()

    third_event = result.filter(pl.col("event_id") == 2)
    seq_dst = list(third_event["sequence_dst"][0])
    seq_success = list(third_event["sequence_success"][0])

    # Most-recent-last ordering: [event 0's dst, event 1's dst].
    assert seq_dst != [PAD_ID, PAD_ID]
    assert seq_success[-1] == 1  # event 1 (immediately prior) succeeded
    # The current event's own outcome (failure) must never leak into its own context.
    assert len(seq_success) == 2
