"""M1 must return one score per event, in the caller's row order.

Every call site attaches `score()`'s output back onto its input frame *by
position* (`frame.with_columns(pl.Series("_score", scores))`). M1 computes R6
and R7 on frames re-sorted by (group key, time) and joins them back, and
Polars gives no row-order guarantee on a join unless asked. A reordering here
would misattribute every score to a different event while leaving a results
table that looks entirely reasonable — so it is worth a test that fails
loudly rather than a comment that hopes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from authbench.models.rules import RULES, RulesScorer


def _features(n: int = 200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    users = [f"U{i}" for i in rng.integers(0, 7, n)]
    time = np.sort(rng.integers(0, 200_000, n))
    return pl.DataFrame(
        {
            "event_id": np.arange(n),
            "time": time,
            "day": time // 86_400,
            "src_user": users,
            "auth_type": [f"A{i}" for i in rng.integers(0, 3, n)],
            "src_computer": [f"C{i}" for i in rng.integers(0, 9, n)],
            "dst_computer": [f"C{i}" for i in rng.integers(0, 9, n)],
            "pair_is_new": rng.integers(0, 2, n).astype(bool),
            "src_user_1h_n_distinct_dst": rng.integers(0, 6, n),
            "dst_computer_1h_n_failures": rng.integers(0, 4, n),
            "is_success": rng.integers(0, 2, n).astype(bool),
            "hour_deviation_from_profile": rng.random(n),
            "src_user_is_machine": rng.integers(0, 2, n).astype(bool),
            "pair_global_rarity": rng.random(n),
        }
    )


def test_rule_scores_preserve_the_input_row_order() -> None:
    # Deliberately not sorted by src_user or by event_id: the internal R6/R7
    # frames are, and that is exactly what the join has to undo.
    features = _features().sample(fraction=1.0, shuffle=True, seed=3)

    scorer = RulesScorer()
    scorer.fit(features.lazy())
    scored = scorer.rule_scores(features.lazy())

    assert scored["event_id"].to_list() == features["event_id"].to_list()


def test_score_returns_one_value_per_event_aligned_with_the_input() -> None:
    features = _features().sample(fraction=1.0, shuffle=True, seed=11)

    scorer = RulesScorer()
    scorer.fit(features.lazy())
    scores = scorer.score(features.lazy())

    assert len(scores) == features.height
    assert scores.null_count() == 0

    # Scoring the same events in a different order must give each event the
    # same score — the property that positional attachment silently depends on.
    permuted = features.sample(fraction=1.0, shuffle=True, seed=99)
    permuted_scores = scorer.score(permuted.lazy())

    by_event = dict(zip(features["event_id"].to_list(), scores.to_list(), strict=True))
    for event_id, score in zip(
        permuted["event_id"].to_list(), permuted_scores.to_list(), strict=True
    ):
        assert score == by_event[event_id]


def test_weights_sum_to_one_and_cover_every_rule() -> None:
    scorer = RulesScorer()
    assert set(scorer.weights) == {rule.id for rule in RULES}
    assert sum(scorer.weights.values()) == pytest.approx(1.0)


def test_scores_of_earlier_events_do_not_depend_on_later_events() -> None:
    features = _features(n=300, seed=5)
    scorer = RulesScorer()
    scorer.fit(features.lazy())
    whole = scorer.score(features.lazy()).to_numpy()
    prefix = scorer.score(features.head(120).lazy()).to_numpy()
    np.testing.assert_array_equal(prefix, whole[:120])


def test_rule_scores_require_a_fitted_reference() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        RulesScorer().rule_scores(_features().lazy())
