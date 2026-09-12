"""The pipeline's two unstated assumptions about Polars `Categorical`.

`parse.clean` casts every high-cardinality identifier to `Categorical`, and two
things downstream depend on how Polars then compares them:

- `label.redteam_join` joins `auth` against `redteam` on `src_user`,
  `src_domain`, `src_computer` and `dst_computer` — columns cast in *different*
  frames, scanned from different files, so their physical encodings are
  unrelated;
- `features.event.compute_f1` derives `src_dst_user_same`,
  `src_dst_computer_same` and `domain_crossing` by comparing two categorical
  columns against each other.

Both are correct on Polars 1.43: comparison and joins resolve by string value,
not by physical code. Neither is guaranteed by anything this project controls.
`pyproject.toml` pins `polars>=1.9,<2.0`, and if a release inside that range
ever compared by code instead, the red-team join would silently match the wrong
events and F1 would silently invert three boolean features — a benchmark that
still produced entirely plausible numbers.

There is no `pl.enable_string_cache()` anywhere in the codebase, and after
these tests that is a recorded fact rather than an oversight.
"""

from __future__ import annotations

import polars as pl


def test_categoricals_from_separate_frames_join_by_value_not_by_code() -> None:
    """`label.redteam_join`'s join key, reduced to its essentials.

    The two frames are built so their physical encodings cannot coincide: the
    same three strings appear in a different order in each.
    """
    left = pl.DataFrame({"k": ["zeta", "alpha", "mid"], "v": [1, 2, 3]}).with_columns(
        pl.col("k").cast(pl.Categorical)
    )
    right = pl.DataFrame({"k": ["alpha", "mid", "zeta"], "flag": [True, True, True]}).with_columns(
        pl.col("k").cast(pl.Categorical)
    )

    assert left["k"].to_physical().to_list() != right["k"].to_physical().to_list(), (
        "the fixture must have mismatched encodings or it tests nothing"
    )

    joined = left.join(right, on="k", how="left").sort("v")

    assert joined["flag"].to_list() == [True, True, True]


def test_a_categorical_join_that_should_miss_does_miss() -> None:
    """The other half: matching by value must not become matching by anything
    looser. A key absent from the right frame stays null however the codes line
    up.
    """
    left = pl.DataFrame({"k": ["alpha", "absent"], "v": [1, 2]}).with_columns(
        pl.col("k").cast(pl.Categorical)
    )
    right = pl.DataFrame({"k": ["alpha"], "flag": [True]}).with_columns(
        pl.col("k").cast(pl.Categorical)
    )

    joined = left.join(right, on="k", how="left").sort("v")

    assert joined["flag"].to_list() == [True, None]


def test_two_categorical_columns_compare_by_value_within_one_frame() -> None:
    """`compute_f1`'s `src_dst_user_same` / `src_dst_computer_same` /
    `domain_crossing`, reduced to their essentials.

    Both columns are cast in the same `select`, but each gets its own encoding
    from its own order of first appearance — so `bob == bob` and `dave == dave`
    have to be true while the underlying codes differ.
    """
    frame = pl.DataFrame(
        {
            "src_user": ["alice", "bob", "carol", "dave"],
            "dst_user": ["zoe", "bob", "alice", "dave"],
        }
    ).select(pl.col("src_user").cast(pl.Categorical), pl.col("dst_user").cast(pl.Categorical))

    assert frame["src_user"].to_physical().to_list() != frame["dst_user"].to_physical().to_list(), (
        "the fixture must have mismatched encodings or it tests nothing"
    )

    same = frame.with_columns((pl.col("src_user") == pl.col("dst_user")).alias("same"))["same"]

    assert same.to_list() == [False, True, False, True]


def test_categorical_inequality_is_the_negation_of_equality() -> None:
    """`domain_crossing` is built with `!=` rather than `==`, so the negation
    is asserted rather than assumed to follow.
    """
    frame = pl.DataFrame(
        {"src_domain": ["DOM1", "DOM2", "DOM1"], "dst_domain": ["DOM1", "DOM1", "DOM3"]}
    ).select(pl.col("src_domain").cast(pl.Categorical), pl.col("dst_domain").cast(pl.Categorical))

    crossing = frame.with_columns(
        (pl.col("src_domain") != pl.col("dst_domain")).alias("domain_crossing"),
        (pl.col("src_domain") == pl.col("dst_domain")).alias("same_domain"),
    )

    assert crossing["domain_crossing"].to_list() == [False, True, True]
    assert crossing["same_domain"].to_list() == [
        not x for x in crossing["domain_crossing"].to_list()
    ]
