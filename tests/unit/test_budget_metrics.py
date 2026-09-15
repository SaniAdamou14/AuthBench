"""The alert-budget metrics — the project's central register, and until now the
one with no unit test of its own.

The cases here are the ones that were silently wrong: a tie at the budget
threshold decided by whatever order the frame happened to arrive in, and a
column of zeros that could not distinguish a model which missed by one rank
from one which missed by nine million.
"""

from __future__ import annotations

import polars as pl
import pytest

from authbench.evaluate.budget import (
    budget_for_first_detection,
    campaign_detection_budgets,
    campaign_recall_at_budget,
    campaign_recall_bracket,
    compute_budget_curve,
    event_recall_at_budget,
    tie_exposure,
)


def _tied_day(n_benign: int = 5) -> pl.DataFrame:
    """One day, every event on the same score, the malicious one last by time."""
    n = n_benign + 1
    return pl.DataFrame(
        {
            "event_id": list(range(n)),
            "day": [0] * n,
            "time": list(range(n)),
            "score": [1.0] * n,
            "is_malicious": [False] * n_benign + [True],
            "campaign_id": [None] * n_benign + [1],
        },
        schema_overrides={"campaign_id": pl.Int64},
    )


def test_a_tie_is_broken_by_arrival_order_not_by_frame_order() -> None:
    """The bug this module was carrying.

    With every score equal, the old implementation ranked by row position, so
    the answer depended on how the caller happened to have sorted the frame —
    and after `compute_f4` that sort is `(src_user, time)`, i.e. alphabetical
    by user. Shuffling the rows must not move the metric.
    """
    frame = _tied_day()
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=7)
    reversed_ = frame.reverse()

    for other in (shuffled, reversed_):
        assert campaign_recall_at_budget(other, "score", 3) == campaign_recall_at_budget(
            frame, "score", 3
        )
        assert event_recall_at_budget(other, "score", 3) == event_recall_at_budget(
            frame, "score", 3
        )


def test_the_queue_order_is_arrival_order() -> None:
    """Six events tied, the malicious one last to arrive. A budget of 3 takes
    the first three by time, so it misses; a budget of 6 takes everything.
    """
    frame = _tied_day()

    assert campaign_recall_at_budget(frame, "score", 3) == 0.0
    assert campaign_recall_at_budget(frame, "score", 6) == 1.0


def test_the_bracket_reports_what_the_tie_could_have_done() -> None:
    """The reported 0.0 above is one of two possible answers, and the bracket
    has to say so: break the same tie the other way and recall is 1.0.
    """
    low, high = campaign_recall_bracket(_tied_day(), "score", 3)

    assert low == 0.0
    assert high == 1.0


def test_a_continuous_score_has_no_bracket_to_report() -> None:
    """A model whose scores are all distinct leaves the tie-break nothing to
    decide, and the bracket must collapse onto the point estimate rather than
    manufacturing an interval.
    """
    frame = _tied_day().with_columns(
        pl.Series("score", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    )
    low, high = campaign_recall_bracket(frame, "score", 3)

    assert low == high == campaign_recall_at_budget(frame, "score", 3)


def test_the_best_case_tie_break_spends_one_slot_per_campaign() -> None:
    """Two slots, and a tie group holding three events of campaign 1 and one of
    campaign 2. Taking two events of campaign 1 detects one campaign; taking
    one of each detects two, which is what the upper end of the bracket must
    find.
    """
    frame = pl.DataFrame(
        {
            "event_id": [0, 1, 2, 3, 4, 5],
            "day": [0] * 6,
            "time": [0, 1, 2, 3, 4, 5],
            "score": [1.0] * 6,
            "is_malicious": [True, True, True, True, False, False],
            "campaign_id": [1, 1, 1, 2, None, None],
        },
        schema_overrides={"campaign_id": pl.Int64},
    )

    low, high = campaign_recall_bracket(frame, "score", 2)
    assert low == 0.0  # both benign events go first
    assert high == 1.0  # one event of each campaign


def test_tie_exposure_counts_the_slots_and_the_contenders() -> None:
    frame = _tied_day(n_benign=99)  # 100 events, all tied

    exposure = tie_exposure(frame, "score", 10)

    assert exposure.budget == 10
    assert exposure.n_contested_slots == 10
    assert exposure.n_contenders == 100
    assert exposure.n_contested_days == 1
    assert exposure.is_contested is True


def test_a_tie_that_fits_inside_the_budget_is_not_contested() -> None:
    """Six tied events and a budget of ten: every one of them is alerted
    whatever the ordering, so there is nothing for a tie-break to decide.
    """
    exposure = tie_exposure(_tied_day(), "score", 10)

    assert exposure.n_contenders == 0
    assert exposure.is_contested is False


def test_an_untied_score_reports_no_exposure() -> None:
    frame = _tied_day().with_columns(pl.Series("score", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))

    assert tie_exposure(frame, "score", 3).is_contested is False


def _two_days() -> pl.DataFrame:
    """Two days; the malicious events sit at rank 3 on day 0 and rank 2 on day 1."""
    return pl.DataFrame(
        {
            "event_id": [0, 1, 2, 3, 4, 5, 6, 7],
            "day": [0, 0, 0, 0, 1, 1, 1, 1],
            "time": [0, 1, 2, 3, 86_400, 86_401, 86_402, 86_403],
            "score": [0.9, 0.8, 0.7, 0.1, 0.9, 0.8, 0.2, 0.1],
            "is_malicious": [False, False, True, False, False, True, False, False],
            "campaign_id": [None, None, 1, None, None, 2, None, None],
        },
        schema_overrides={"campaign_id": pl.Int64},
    )


def test_budget_for_first_detection_is_the_best_rank_any_campaign_reaches() -> None:
    """Campaign 2 sits at rank 2 of its day, campaign 1 at rank 3 of its own.
    The smallest budget that catches anything is therefore 2.
    """
    assert budget_for_first_detection(_two_days(), "score") == 2


def test_budget_for_first_detection_agrees_with_sweeping_every_budget() -> None:
    """The point of computing it in one pass is that it must equal what a sweep
    would find. Asserted rather than assumed.
    """
    frame = _two_days()
    minimum = budget_for_first_detection(frame, "score")
    assert minimum is not None

    assert campaign_recall_at_budget(frame, "score", minimum) > 0.0
    assert campaign_recall_at_budget(frame, "score", minimum - 1) == 0.0


def test_campaign_detection_budgets_gives_one_row_per_campaign() -> None:
    budgets = campaign_detection_budgets(_two_days(), "score")

    assert budgets.height == 2
    assert dict(zip(*budgets.to_dict(as_series=False).values(), strict=True)) == {2: 2, 1: 3}


def test_budget_for_first_detection_is_none_without_campaigns() -> None:
    frame = _two_days().with_columns(pl.lit(None, dtype=pl.Int64).alias("campaign_id"))

    assert budget_for_first_detection(frame, "score") is None


def test_a_campaign_ranked_last_still_gets_a_finite_budget() -> None:
    """`None` must mean "nothing to detect", never "never detected".

    A rank is bounded by the size of its own day, so a model that ranks the
    attack dead last still has a budget at which it would be caught — an
    absurd one, which is exactly the information a saturated recall row
    destroys. Reporting `None` here would collapse "missed by everything" back
    into the same bucket as "no campaign in the split".
    """
    frame = _two_days().with_columns(
        # The malicious events now carry the two lowest scores in their days.
        pl.Series("score", [0.9, 0.8, 0.01, 0.7, 0.9, 0.02, 0.8, 0.7])
    )

    assert budget_for_first_detection(frame, "score") == 4
    assert campaign_recall_at_budget(frame, "score", 3) == 0.0


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_recall_never_leaves_its_own_bracket(budget: int) -> None:
    """The reported point estimate is one achievable tie-break among others, so
    it has to lie inside the bracket at every budget — otherwise the bracket is
    not bracketing the thing it is printed next to.
    """
    frame = _tied_day()
    low, high = campaign_recall_bracket(frame, "score", budget)
    point = campaign_recall_at_budget(frame, "score", budget)

    assert low <= point <= high


def test_the_curve_carries_the_bracket_the_exposure_and_the_first_budget() -> None:
    curve = compute_budget_curve(_two_days(), "score", "M_test", budgets=[1, 2, 4])

    assert curve.budgets == [1, 2, 4]
    assert len(curve.campaign_recall_min) == len(curve.campaign_recall_max) == 3
    assert len(curve.tie_exposure) == 3
    assert curve.budget_for_first_detection == 2
    assert curve.campaign_recall == [0.0, 0.5, 1.0]
