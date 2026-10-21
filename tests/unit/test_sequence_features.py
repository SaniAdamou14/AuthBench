from __future__ import annotations

import polars as pl

from authbench.features.sequence import (
    FAILURE_TOKEN,
    FIRST_EVENT_BUCKET,
    SEQUENCE_COLUMNS,
    SUCCESS_TOKEN,
    compute_f6,
)
from authbench.features.vocabulary import FIRST_TOKEN_ID, NULL_ID, OOV_ID, PAD_ID

SCHEMA = {
    "event_id": pl.UInt64,
    "time": pl.Int32,
    "src_user": pl.Categorical,
    "dst_computer": pl.Categorical,
    "auth_type": pl.Categorical,
    "success": pl.Boolean,
}
DST_VOCABULARY = ["A", "B", "C"]
AUTH_VOCABULARY = ["K", "N"]


def _f6(rows: list[tuple[int, int, str, str, str | None, bool]], context: int) -> pl.DataFrame:
    frame = pl.LazyFrame([dict(zip(SCHEMA, row, strict=True)) for row in rows], schema=SCHEMA)
    return (
        compute_f6(
            frame,
            dst_vocabulary=DST_VOCABULARY,
            auth_type_vocabulary=AUTH_VOCABULARY,
            context_length=context,
        )
        .collect()
        .sort("event_id")
    )


def _code(value: str) -> int:
    return FIRST_TOKEN_ID + DST_VOCABULARY.index(value)


def test_first_event_is_fully_padded() -> None:
    result = _f6([(0, 0, "alice", "A", "K", True)], context=4)
    for column in SEQUENCE_COLUMNS:
        assert result[column][0].to_list() == [PAD_ID] * 4, column


def test_sequence_holds_only_strictly_prior_events_oldest_first() -> None:
    result = _f6(
        [
            (0, 0, "alice", "A", "K", True),
            (1, 10, "alice", "B", "N", True),
            (2, 20, "alice", "C", "K", False),
        ],
        context=2,
    )
    third = result.row(2, named=True)
    assert third["sequence_dst"] == [_code("A"), _code("B")]
    assert third["sequence_success"] == [SUCCESS_TOKEN, SUCCESS_TOKEN]
    assert third["dst_code"] == _code("C")


def test_padding_never_aliases_a_failure_or_a_first_event() -> None:
    result = _f6([(0, 0, "alice", "A", "K", False), (1, 5, "alice", "B", "K", True)], context=2)
    second = result.row(1, named=True)
    assert second["sequence_success"] == [PAD_ID, FAILURE_TOKEN]
    assert second["sequence_delta_bucket"] == [PAD_ID, FIRST_EVENT_BUCKET]


def test_unseen_and_null_categories_get_reserved_codes() -> None:
    result = _f6([(0, 0, "alice", "Z", None, True), (1, 5, "alice", "A", "K", True)], context=1)
    second = result.row(1, named=True)
    assert result.row(0, named=True)["dst_code"] == OOV_ID
    assert second["sequence_dst"] == [OOV_ID]
    assert second["sequence_auth_type"] == [NULL_ID]


def test_other_users_never_enter_a_context() -> None:
    result = _f6([(0, 0, "bob", "A", "K", True), (1, 5, "alice", "B", "K", True)], context=1)
    assert result.row(1, named=True)["sequence_dst"] == [PAD_ID]
