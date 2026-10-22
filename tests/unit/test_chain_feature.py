from __future__ import annotations

import polars as pl

from authbench.features.temporal import NightWindow, compute_f4


def _f4(rows: list[tuple[int, str, str, str]]) -> pl.DataFrame:
    frame = pl.LazyFrame(
        [
            {"event_id": i, "time": t, "src_user": u, "src_computer": s, "dst_computer": d}
            for i, (t, u, s, d) in enumerate(rows)
        ],
        schema={
            "event_id": pl.Int64,
            "time": pl.Int64,
            "src_user": pl.Categorical,
            "src_computer": pl.Categorical,
            "dst_computer": pl.Categorical,
        },
    )
    return compute_f4(frame, NightWindow(0, 3600)).collect().sort("event_id")


def test_a_hop_starts_where_the_same_users_previous_event_ended() -> None:
    result = _f4(
        [
            (0, "alice", "A", "B"),
            (60, "bob", "Z", "C"),
            (120, "alice", "B", "C"),
            (180, "alice", "Q", "D"),
        ]
    )
    assert result["prev_dst_is_src"].to_list() == [False, False, True, False]
    assert result["hours_since_prev_event_same_user"][2] == 120 / 3600
