"""F3 relations: user-destination, user-source and source-destination novelty and day-anchored rarity."""

from __future__ import annotations

import math

import polars as pl
import pytest

from authbench.features.novelty import compute_f3

SCHEMA = {
    "event_id": pl.Int64,
    "time": pl.Int64,
    "day": pl.Int64,
    "src_user": pl.Categorical,
    "src_computer": pl.Categorical,
    "dst_computer": pl.Categorical,
    "auth_type": pl.Categorical,
}


def _f3(rows: list[tuple[int, str | None, str, str, str]]) -> pl.DataFrame:
    frame = pl.LazyFrame(
        [
            {
                "event_id": i,
                "time": t,
                "day": t // 86_400,
                "src_user": u,
                "src_computer": s,
                "dst_computer": d,
                "auth_type": a,
            }
            for i, (t, a, u, s, d) in enumerate(rows)
        ],
        schema=SCHEMA,
    )
    return compute_f3(frame).collect().sort("event_id")


def test_source_novelty_flags_a_known_user_on_a_new_source() -> None:
    result = _f3(
        [
            (0, "K", "alice", "C1", "C9"),
            (10, "K", "alice", "C1", "C8"),
            (20, "K", "alice", "C2", "C9"),
        ]
    )
    assert result["user_src_is_new"].to_list() == [True, False, True]
    assert result["src_dst_is_new"].to_list() == [True, True, True]
    assert result["pair_is_new"].to_list() == [True, True, False]


def test_equally_novel_events_of_one_day_have_equal_rarity() -> None:
    day = 86_400
    result = _f3(
        [
            (0, "K", "u0", "C0", "C0"),
            (1, "K", "u1", "C1", "C1"),
            (day + 10, "K", "alice", "C5", "C6"),
            (day + 50_000, "K", "bob", "C7", "C8"),
        ]
    )
    later_day = result.filter(pl.col("day") == 1)
    assert later_day["pair_global_rarity"].n_unique() == 1
    assert later_day["pair_global_rarity"][0] == pytest.approx(-math.log(1 / 3))


def test_user_source_rarity_is_conditioned_on_the_users_earlier_days() -> None:
    day = 86_400
    result = _f3(
        [
            (0, "K", "alice", "C1", "C9"),
            (5, "K", "alice", "C1", "C9"),
            (day, "K", "alice", "C1", "C9"),
            (day + 5, "K", "alice", "C3", "C9"),
        ]
    )
    rarity = result["user_src_rarity"].to_list()
    assert rarity[2] == pytest.approx(-math.log(3 / 3))
    assert rarity[3] == pytest.approx(-math.log(1 / 3))
    assert rarity[0] == pytest.approx(0.0)


def test_novelty_ignores_future_events() -> None:
    past = [(0, "K", "alice", "C1", "C9"), (10, "K", "alice", "C2", "C9")]
    future = [(20, "K", "alice", "C2", "C9"), (30, "K", "bob", "C1", "C9")]
    short, long = _f3(past), _f3(past + future).head(len(past))
    for column in (
        "pair_is_new",
        "user_src_is_new",
        "src_dst_is_new",
        "user_src_rarity",
        "src_dst_rarity",
    ):
        assert short[column].to_list() == long[column].to_list(), column


def test_a_null_auth_type_is_new_per_user_not_once_for_everyone() -> None:
    result = _f3(
        [
            (0, None, "alice", "C1", "C9"),
            (10, None, "bob", "C1", "C9"),
            (20, None, "alice", "C1", "C9"),
            (30, "K", "alice", "C1", "C9"),
        ]
    )
    assert result["user_auth_is_new"].to_list() == [True, True, False, True]
