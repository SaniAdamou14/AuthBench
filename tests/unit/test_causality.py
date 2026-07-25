"""US-108: for every claimed-causal feature (F2, F3, F4), build a synthetic
event whose feature value would change if a future event were visible, and
assert the computed value matches the causal (past-only) expectation.
"""

from __future__ import annotations

import polars as pl

from authbench.features.history import compute_f2
from authbench.features.novelty import compute_f3
from authbench.features.temporal import NightWindow, compute_f4

BASE_COLUMNS = {
    "event_id": pl.UInt64,
    "time": pl.Int32,
    "day": pl.Int16,
    "src_user": pl.Categorical,
    "src_computer": pl.Categorical,
    "dst_computer": pl.Categorical,
    "dst_user": pl.Categorical,
    "auth_type": pl.Categorical,
    "success": pl.Boolean,
}


def _frame(rows: list[dict]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema=BASE_COLUMNS).lazy()


def test_f2_history_count_excludes_current_and_future_events() -> None:
    # Same user, 3 events one hour apart: at the 2nd event exactly one prior
    # event exists within the 1h window; a 3rd (future) event must not change it.
    rows = [
        {
            "event_id": 0,
            "time": 0,
            "day": 0,
            "src_user": "alice",
            "src_computer": "A",
            "dst_computer": "X",
            "dst_user": "alice",
            "auth_type": "K",
            "success": True,
        },
        {
            "event_id": 1,
            "time": 60,
            "day": 0,
            "src_user": "alice",
            "src_computer": "A",
            "dst_computer": "Y",
            "dst_user": "alice",
            "auth_type": "K",
            "success": True,
        },
    ]
    without_future = compute_f2(_frame(rows), entities=["src_user"], windows_hours=[1]).collect()
    second_event_count = without_future.filter(pl.col("event_id") == 1)["src_user_1h_n_events"][0]
    assert second_event_count == 1  # only event 0 is a strictly-prior event

    rows_with_future = [
        *rows,
        {
            "event_id": 2,
            "time": 120,
            "day": 0,
            "src_user": "alice",
            "src_computer": "A",
            "dst_computer": "Z",
            "dst_user": "alice",
            "auth_type": "K",
            "success": True,
        },
    ]
    with_future = compute_f2(
        _frame(rows_with_future), entities=["src_user"], windows_hours=[1]
    ).collect()
    second_event_count_with_future = with_future.filter(pl.col("event_id") == 1)[
        "src_user_1h_n_events"
    ][0]

    assert second_event_count_with_future == second_event_count == 1


def test_f3_pair_is_new_unaffected_by_future_repeat_of_the_pair() -> None:
    rows = [
        {
            "event_id": 0,
            "time": 0,
            "day": 0,
            "src_user": "bob",
            "src_computer": "A",
            "dst_computer": "X",
            "dst_user": "bob",
            "auth_type": "K",
            "success": True,
        },
    ]
    only_first = compute_f3(_frame(rows)).collect()
    assert only_first["pair_is_new"][0] is True

    rows_with_future_repeat = [
        *rows,
        {
            "event_id": 1,
            "time": 3600,
            "day": 0,
            "src_user": "bob",
            "src_computer": "A",
            "dst_computer": "X",
            "dst_user": "bob",
            "auth_type": "K",
            "success": True,
        },
    ]
    with_future = compute_f3(_frame(rows_with_future_repeat)).collect()
    first_event_row = with_future.filter(pl.col("event_id") == 0)
    second_event_row = with_future.filter(pl.col("event_id") == 1)

    # The first occurrence is still "new" even though the pair repeats later...
    assert first_event_row["pair_is_new"][0] is True
    # ...and the second occurrence is correctly "not new", proving the flag is
    # tracking real history, not just always True.
    assert second_event_row["pair_is_new"][0] is False


def test_f4_first_event_has_infinite_delay_regardless_of_future_events() -> None:
    rows = [
        {
            "event_id": 0,
            "time": 100,
            "day": 0,
            "src_user": "carol",
            "src_computer": "A",
            "dst_computer": "X",
            "dst_user": "carol",
            "auth_type": "K",
            "success": True,
        },
        {
            "event_id": 1,
            "time": 200,
            "day": 0,
            "src_user": "carol",
            "src_computer": "A",
            "dst_computer": "Y",
            "dst_user": "carol",
            "auth_type": "K",
            "success": True,
        },
    ]
    night_window = NightWindow(start_second=0, end_second=21_600)
    result = compute_f4(_frame(rows), night_window).collect()

    first_event = result.filter(pl.col("event_id") == 0)
    second_event = result.filter(pl.col("event_id") == 1)

    assert first_event["hours_since_prev_event_same_user"][0] == float("inf")
    assert second_event["hours_since_prev_event_same_user"][0] == (200 - 100) / 3600.0
