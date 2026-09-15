"""`evaluate_model` assembles every reported metric in one place.

Both `authbench demo` and the DVC `train_eval` stage go through it, so the
two cannot drift into reporting different things — and the operational and
literature-comparable registers stay separated in the emitted JSON, which is
the whole reason the split exists.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from authbench.evaluate.summary import ModelEvaluation, evaluate_model

BUDGETS = [1, 2]


def _scored(scores: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_id": [0, 1, 2, 3, 4, 5],
            "day": [0, 0, 0, 1, 1, 1],
            "time": [100, 200, 300, 86_500, 86_600, 86_700],
            "is_malicious": [True, False, False, True, False, False],
            "campaign_id": [1, None, None, 2, None, None],
            "_score__m": scores,
        }
    )


def _evaluate(scores: list[float], **kwargs: object) -> ModelEvaluation:
    return evaluate_model(
        _scored(scores),
        "_score__m",
        "m",
        budgets=BUDGETS,
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_perfect_ranking_detects_every_campaign_at_budget_one() -> None:
    # The malicious event is the top-scored one on each of the two days.
    evaluation = _evaluate([0.9, 0.1, 0.2, 0.9, 0.1, 0.2])

    assert evaluation.curve.campaign_recall[0] == 1.0
    ttd = evaluation.time_to_detection[0]
    assert ttd.budget == 1
    assert ttd.n_detected == 2
    assert ttd.n_never_detected == 0
    assert ttd.median_delay_seconds == 0.0


def test_undetected_campaigns_are_counted_never_folded_into_the_delays() -> None:
    """US-127: a campaign that is never alerted must not silently leave the
    denominator, and must not contribute a delay of zero either.
    """
    evaluation = _evaluate([0.1, 0.9, 0.8, 0.1, 0.9, 0.8])
    ttd = evaluation.time_to_detection[0]

    assert ttd.n_total_campaigns == 2
    assert ttd.n_detected == 0
    assert ttd.n_never_detected == 2
    assert ttd.median_delay_seconds is None
    assert ttd.mean_delay_seconds is None


def test_metrics_stay_in_separate_registers_in_the_emitted_record() -> None:
    record = _evaluate([0.9, 0.1, 0.2, 0.9, 0.1, 0.2], fpr_targets=[0.5]).to_dict()

    assert set(record["operational"]) == {  # type: ignore[arg-type]
        "budgets",
        "event_recall",
        "campaign_recall",
        # What the recall row alone cannot say: how far the tie-break could
        # move each point, what it was competing over, and the budget at which
        # this model first detects anything.
        "campaign_recall_tie_bracket",
        "tie_exposure",
        "budget_for_first_detection",
        "time_to_detection",
    }
    assert set(record["literature_comparable"]) == {  # type: ignore[arg-type]
        "roc_auc",
        "precision_at_k_global",
        "recall_at_fixed_fpr",
    }
    assert record["model"] == "m"


def test_roc_auc_is_omitted_rather_than_crashing_when_a_class_is_absent() -> None:
    """sklearn raises on a single-class target. A degenerate split is a bug
    worth surfacing upstream, not a stack trace from inside a metric.
    """
    frame = _scored([0.9, 0.1, 0.2, 0.9, 0.1, 0.2]).with_columns(
        pl.lit(False).alias("is_malicious"), pl.lit(None, dtype=pl.Int64).alias("campaign_id")
    )
    evaluation = evaluate_model(frame, "_score__m", "m", budgets=BUDGETS, fpr_targets=[0.5])

    assert evaluation.roc_auc is None
    assert evaluation.recall_at_fixed_fpr == {}
    assert np.isnan(evaluation.auc_pr) or evaluation.auc_pr == 0.0


def test_roc_auc_can_be_switched_off_entirely() -> None:
    evaluation = _evaluate([0.9, 0.1, 0.2, 0.9, 0.1, 0.2], report_roc_auc=False)
    assert evaluation.roc_auc is None


def test_score_resolution_separates_a_fine_score_from_a_degenerate_one() -> None:
    """The number that answers the tie-artifact objection to the ROC-AUC claim.

    A reviewer can reply that ROC-AUC credits a tied pair 0.5, so a coarse
    score is pulled toward 0.5 by arithmetic rather than by anything about the
    operating point. These two fields make that checkable per model instead of
    arguable.
    """
    fine = _evaluate([0.9, 0.1, 0.2, 0.8, 0.3, 0.4]).score_resolution
    assert fine is not None
    assert fine.n_events == 6
    assert fine.n_distinct_scores == 6
    assert fine.largest_tie_fraction == 1 / 6

    # A binary score, the shape M0b always-fail has: two verdicts for the whole
    # split, and every rank metric over it is mostly reporting tie conventions.
    binary = _evaluate([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).score_resolution
    assert binary is not None
    assert binary.n_distinct_scores == 2
    assert binary.largest_tie_fraction == 4 / 6


def test_score_resolution_of_a_constant_score_is_maximally_degenerate() -> None:
    """One value for every event: the model has a single verdict, and
    `largest_tie_fraction` must say 1.0 rather than quietly look ordinary.
    """
    resolution = _evaluate([0.5] * 6).score_resolution

    assert resolution is not None
    assert resolution.n_distinct_scores == 1
    assert resolution.largest_tie_fraction == 1.0


def test_score_resolution_is_emitted_outside_both_registers() -> None:
    """It qualifies a metric in each register, so it belongs to neither."""
    record = _evaluate([0.9, 0.1, 0.2, 0.8, 0.3, 0.4]).to_dict()

    assert set(record["score_resolution"]) == {  # type: ignore[arg-type]
        "n_events",
        "n_distinct_scores",
        "largest_tie_fraction",
    }
    assert "score_resolution" not in record["operational"]  # type: ignore[operator]
    assert "score_resolution" not in record["literature_comparable"]  # type: ignore[operator]
