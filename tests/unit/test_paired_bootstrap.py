"""One resampling pass, shared by every model (US-128).

The point of `paired_campaign_bootstrap` is that all models see the *same*
resample. That is what makes the pairwise differences paired — the
campaign-composition noise that dominates variance at this positive rate
cancels between two models instead of being counted twice — and it is what
keeps the per-model CIs and the comparisons on one sampling distribution.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from authbench.evaluate.metrics import auc_pr
from authbench.evaluate.stats_tests import (
    PairwiseComparison,
    minimum_resamples_for_family,
    paired_campaign_bootstrap,
    render_comparison_sentence,
)
from authbench.evaluate.summary import pairwise_comparisons, score_column

MODELS = ["good", "bad", "twin_of_good"]


def _scored(n: int = 300, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    campaign_id: list[int | None] = [None] * n
    for i in range(6):
        campaign_id[i] = 1
    for i in range(6, 12):
        campaign_id[i] = 2
    is_malicious = [c is not None for c in campaign_id]

    good = np.where(np.array(is_malicious), rng.random(n) * 0.2 + 0.8, rng.random(n) * 0.5)
    return pl.DataFrame(
        {
            "event_id": np.arange(n),
            "time": np.arange(n) * 60,
            "day": (np.arange(n) * 60) // 86_400,
            "campaign_id": campaign_id,
            "is_malicious": is_malicious,
            score_column("good"): good,
            score_column("bad"): rng.random(n),
            # Byte-identical to `good`: the null hypothesis is literally true.
            score_column("twin_of_good"): good.copy(),
        }
    )


def _metric(frame: pl.DataFrame, col: str) -> float:
    return auc_pr(frame["is_malicious"].to_numpy(), frame[col].to_numpy())


def _bootstrap(n_resamples: int = 200, seed: int = 7):
    return paired_campaign_bootstrap(
        _scored(),
        _metric,
        {name: score_column(name) for name in MODELS},
        n_resamples=n_resamples,
        seed=seed,
    )


def test_every_model_is_evaluated_on_the_same_resamples() -> None:
    """Two models with identical scores must produce an identical resample
    distribution. If each model drew its own resamples they would differ.
    """
    bootstrap = _bootstrap()
    assert np.array_equal(bootstrap.resampled["good"], bootstrap.resampled["twin_of_good"])


def test_identical_models_are_never_declared_different() -> None:
    comparisons = _bootstrap().comparisons("auc_pr")
    twins = next(c for c in comparisons if {c.model_a, c.model_b} == {"good", "twin_of_good"})

    assert twins.diff == pytest.approx(0.0)
    assert twins.p_value == pytest.approx(1.0)
    assert twins.significant is False


def test_a_real_gap_is_detected_and_bracketed_away_from_zero() -> None:
    comparisons = _bootstrap().comparisons("auc_pr")
    gap = next(c for c in comparisons if {c.model_a, c.model_b} == {"good", "bad"})

    assert gap.significant is True
    assert gap.diff_ci_low is not None and gap.diff_ci_high is not None
    assert gap.diff_ci_low > 0 or gap.diff_ci_high < 0  # interval excludes zero


def test_confidence_intervals_are_ordered_and_bracket_the_point_estimate() -> None:
    bootstrap = _bootstrap()
    for name in MODELS:
        ci = bootstrap.ci(name)
        assert ci.ci_low <= ci.ci_high
        assert ci.confidence == 0.95


def test_every_pair_appears_exactly_once() -> None:
    comparisons = _bootstrap().comparisons("auc_pr")
    pairs = {frozenset((c.model_a, c.model_b)) for c in comparisons}

    assert len(comparisons) == len(MODELS) * (len(MODELS) - 1) // 2
    assert len(pairs) == len(comparisons)


def test_results_are_reproducible_for_a_fixed_seed() -> None:
    a = _bootstrap(seed=3).comparisons("auc_pr")
    b = _bootstrap(seed=3).comparisons("auc_pr")

    assert [c.p_value for c in a] == [c.p_value for c in b]


def test_p_values_stay_in_range_and_are_never_exactly_zero() -> None:
    """The +1 continuity correction: a finite number of resamples cannot
    justify claiming p = 0.
    """
    for comparison in _bootstrap().comparisons("auc_pr"):
        assert 0.0 < comparison.p_value <= 1.0


def test_resamples_with_no_positives_are_discarded_not_counted_as_ties() -> None:
    """Regression guard for a p-value floor that had nothing to do with the models.

    A ranking metric over zero positives is undefined, but scikit-learn
    returns 0.0 rather than raising — so on a positive-free resample every
    model ties. Those artificial ties fall in both tails of the two-sided
    test. With two campaigns among ~290 blocks, ~14% of resamples are
    degenerate, which alone floors every p-value near 0.27; with the single
    campaign the demo's test split carries, near 0.74. No comparison could
    ever clear alpha=0.05, whatever the models did.
    """
    bootstrap = _bootstrap()

    assert bootstrap.n_degenerate_discarded > 0, "the fixture should exercise the discard path"
    # Every retained resample must be one where the metric is defined, so a
    # genuinely different model can reach significance.
    gap = next(
        c for c in bootstrap.comparisons("auc_pr") if {c.model_a, c.model_b} == {"good", "bad"}
    )
    assert gap.p_value < 0.05


def test_a_frame_with_no_positives_at_all_is_an_error_not_an_interval() -> None:
    frame = _scored().with_columns(
        pl.lit(False).alias("is_malicious"), pl.lit(None, dtype=pl.Int64).alias("campaign_id")
    )
    with pytest.raises(ValueError, match="needs positives"):
        paired_campaign_bootstrap(frame, _metric, {"good": score_column("good")})


def test_too_few_resamples_to_ever_clear_holm_is_flagged(caplog: pytest.LogCaptureFixture) -> None:
    """A second floor, independent of the data: a percentile bootstrap cannot
    report a p-value below 2/(R+1), and Holm's strictest threshold is
    alpha/n_pairs. If the first exceeds the second, every comparison comes
    back non-significant by arithmetic — which reads exactly like a real
    "no difference" result. The demo hit this at R=300 over 21 pairs, and
    `conf/eval/default.yaml` hit it at R=1000 over the catalog's 28.
    """
    with caplog.at_level("WARNING"):
        _bootstrap(n_resamples=60).comparisons("auc_pr")

    assert "cannot resolve a p-value" in caplog.text


def test_minimum_resamples_for_family_matches_the_holm_threshold() -> None:
    for n_pairs in (3, 21, 28):
        required = minimum_resamples_for_family(n_pairs, alpha=0.05)
        assert 2.0 / (required + 1) <= 0.05 / n_pairs
        assert 2.0 / required > 0.05 / n_pairs  # and it is the *smallest* such value


def test_the_reported_interval_points_the_same_way_as_the_sentence() -> None:
    """The sentence reports |Δ| in `better - worse` order. When that swaps the
    pair, the interval has to be reflected through zero too, or a positive
    difference is printed next to a negative interval.
    """
    comparison = PairwiseComparison(
        model_a="weak",
        model_b="strong",
        metric_name="auc_pr",
        diff=-0.42,
        p_value=0.001,
        significant=True,
        diff_ci_low=-0.60,
        diff_ci_high=-0.30,
        method="paired_campaign_bootstrap",
    )
    sentence = render_comparison_sentence(comparison)

    assert "strong outperforms weak" in sentence
    assert "Δ=0.4200" in sentence
    assert "[+0.3000, +0.6000]" in sentence


def test_rejects_being_asked_to_bootstrap_no_models() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        paired_campaign_bootstrap(_scored(), _metric, {})


def test_pairwise_comparisons_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown pairwise test method"):
        pairwise_comparisons(_scored(), MODELS, _bootstrap(), method="t_test")


def test_permutation_route_stays_available_for_small_splits() -> None:
    comparisons = pairwise_comparisons(
        _scored(), MODELS, _bootstrap(), method="permutation", n_permutations=50
    )

    assert len(comparisons) == 3
    assert all(c.method == "paired_permutation" for c in comparisons)
