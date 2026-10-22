from __future__ import annotations

import numpy as np
import pytest

from authbench.evaluate.stats_tests import (
    PairwiseComparison,
    holm_adjusted_p_values,
    holm_bonferroni,
    render_comparison_sentence,
    render_p_value,
)


def test_holm_bonferroni_rejects_only_small_p_values() -> None:
    p_values = [0.001, 0.6, 0.02]
    reject = holm_bonferroni(p_values, alpha=0.05)
    assert reject[0] is True
    assert reject[1] is False


@pytest.mark.parametrize(
    "p_values",
    [
        [0.001, 0.6, 0.02],
        [0.5, 0.5, 0.5],
        # Twelve pairs tied on the bootstrap's resolution floor and one just
        # above it — the exact shape of the published LANL family, where the
        # adjusted value is what tells a reader how little margin there is.
        [*([0.001] * 12), 0.004998, 0.021989, *([0.4] * 7)],
        [1.0],
    ],
)
def test_adjusted_p_values_agree_with_the_step_down_rule_at_every_alpha(
    p_values: list[float],
) -> None:
    """The two must not be able to disagree.

    `holm_bonferroni` walks the step-down rule; `holm_adjusted_p_values`
    reports the alpha at which each comparison would start to be rejected.
    They are two readings of one procedure, and the adjusted value is only
    worth publishing if `adjusted <= alpha` gives the same verdict as the rule
    itself — at every alpha, not just at 0.05.

    `alpha == 1` is excluded and the reason is the clipping, not the
    procedure: an adjusted p-value may not exceed 1, so a family the step-down
    rule rejects nothing of still ends up with adjusted values of exactly 1.0,
    which `<= 1.0` then admits. No significance level is 1.
    """
    adjusted = holm_adjusted_p_values(p_values)

    for alpha in (0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.999):
        assert [a <= alpha for a in adjusted] == holm_bonferroni(p_values, alpha=alpha), (
            f"disagreement at alpha={alpha}"
        )


def test_adjusted_p_values_are_monotone_never_shrink_and_stay_bounded() -> None:
    p_values = [0.04, 0.001, 0.3, 0.02]
    adjusted = holm_adjusted_p_values(p_values)

    assert all(a <= 1.0 for a in adjusted)
    # Correcting for multiplicity can only make a p-value larger.
    assert all(a >= p for a, p in zip(adjusted, p_values, strict=True))
    # Monotone in the original p-value ordering.
    by_rank = [adjusted[i] for i in np.argsort(p_values)]
    assert by_rank == sorted(by_rank)


def test_adjusted_p_values_of_an_empty_family() -> None:
    assert holm_adjusted_p_values([]) == []


def test_a_p_value_at_the_resolution_floor_is_printed_as_a_bound() -> None:
    """The floor is a bound, and the sentence has to say so.

    A bootstrap in which no resample crossed zero has shown that p lies
    *below* `2/(R+1)`. Writing `=` there reports the number of resamples as if
    it were a measurement of the models — which is the single claim this
    project exists to refuse.
    """
    floored = PairwiseComparison(
        model_a="A",
        model_b="B",
        metric_name="auc_pr",
        diff=0.05,
        p_value_raw=0.001,
        significant=True,
        p_value_holm_adjusted=0.021,
        at_resolution_floor=True,
    )
    measured = PairwiseComparison(
        model_a="A",
        model_b="B",
        metric_name="auc_pr",
        diff=0.05,
        p_value_raw=0.0049,
        significant=True,
        p_value_holm_adjusted=0.0441,
        at_resolution_floor=False,
    )

    assert render_p_value(floored) == "Holm-adjusted p <= 0.0210"
    assert render_p_value(measured) == "Holm-adjusted p = 0.0441"
    assert "Holm-adjusted p <= 0.0210" in render_comparison_sentence(floored)


def test_the_printed_p_value_is_the_adjusted_one_not_the_raw_one() -> None:
    """Regression guard for the mislabelling this replaced: the JSON key said
    `p_value_corrected` and the sentence said "corrected p", while both
    carried the uncorrected number.
    """
    comparison = PairwiseComparison(
        model_a="A",
        model_b="B",
        metric_name="auc_pr",
        diff=0.05,
        p_value_raw=0.002,
        significant=True,
        p_value_holm_adjusted=0.042,
    )
    record = comparison.to_dict()

    assert record["p_value_raw"] == 0.002
    assert record["p_value_holm_adjusted"] == 0.042
    assert "p_value_corrected" not in record
    assert "0.0420" in str(record["sentence"])
    assert "0.0020" not in str(record["sentence"])


def test_render_comparison_sentence_never_says_outperforms_when_not_significant() -> None:
    comparison = PairwiseComparison(
        model_a="A",
        model_b="B",
        metric_name="auc_pr",
        diff=0.05,
        p_value_raw=0.4,
        significant=False,
    )
    sentence = render_comparison_sentence(comparison)
    assert "outperforms" not in sentence
    assert "not significantly different" in sentence


def test_render_comparison_sentence_says_outperforms_when_significant() -> None:
    comparison = PairwiseComparison(
        model_a="A",
        model_b="B",
        metric_name="auc_pr",
        diff=0.05,
        p_value_raw=0.001,
        significant=True,
    )
    sentence = render_comparison_sentence(comparison)
    assert "outperforms" in sentence
    assert "A outperforms B" in sentence
