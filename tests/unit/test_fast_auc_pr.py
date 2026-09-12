"""The fast AUC-PR path must be a specialization, never an approximation.

`paired_campaign_bootstrap(fast_auc_pr=True)` replaces "gather a full-size
frame and call scikit-learn" with "walk a pre-sorted rank order under a weight
vector". That is the change that brings a real-dataset bootstrap from days down
to hours, and it is worth exactly nothing if the numbers move.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from authbench.evaluate.metrics import auc_pr
from authbench.evaluate.stats_tests import RankedScores, paired_campaign_bootstrap


def _frame(n_campaigns: int = 6, events_per_campaign: int = 4, n_benign: int = 400) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    campaign_id: list[int | None] = []
    for c in range(1, n_campaigns + 1):
        campaign_id += [c] * events_per_campaign
    campaign_id += [None] * n_benign
    n = len(campaign_id)
    is_malicious = [c is not None for c in campaign_id]
    return pl.DataFrame(
        {
            "campaign_id": campaign_id,
            "is_malicious": is_malicious,
            "good": [0.8 if m else v for m, v in zip(is_malicious, rng.random(n), strict=True)],
            # Coarse rounding on purpose: ties are where a rank-order average
            # precision is easiest to get subtly wrong.
            "tied": np.round(rng.random(n), 1).tolist(),
        }
    )


@pytest.mark.parametrize("column", ["good", "tied"])
def test_ranked_scores_matches_sklearn_weighted_average_precision(column: str) -> None:
    frame = _frame()
    y = frame["is_malicious"].to_numpy()
    scores = frame[column].to_numpy()
    ranked = RankedScores.from_arrays(y, scores)
    rng = np.random.default_rng(0)

    for _ in range(25):
        counts = rng.integers(0, 4, size=frame.height)
        if not (counts * y).sum():
            continue
        assert ranked.average_precision(counts) == pytest.approx(
            average_precision_score(y, scores, sample_weight=counts), abs=1e-12
        )


def test_a_resample_with_no_positive_reports_nan_not_zero() -> None:
    """scikit-learn returns 0.0 here, which ties every model together and puts
    a floor under every p-value. The bootstrap needs to *discard* the draw, so
    the fast path has to distinguish "undefined" from "worst possible"."""
    frame = _frame()
    y = frame["is_malicious"].to_numpy()
    ranked = RankedScores.from_arrays(y, frame["good"].to_numpy())

    counts = np.where(y, 0, 1).astype(np.int32)

    assert np.isnan(ranked.average_precision(counts))


@pytest.mark.parametrize("column", ["good", "tied"])
def test_the_fast_bootstrap_returns_the_same_numbers_as_the_generic_one(column: str) -> None:
    frame = _frame()
    metric = lambda f, col: auc_pr(f["is_malicious"].to_numpy(), f[col].to_numpy())  # noqa: E731
    kwargs = {"n_resamples": 300, "seed": 5}

    slow = paired_campaign_bootstrap(frame, metric, {"m": column}, **kwargs)
    fast = paired_campaign_bootstrap(frame, metric, {"m": column}, fast_auc_pr=True, **kwargs)

    assert fast.point_estimates == pytest.approx(slow.point_estimates)
    assert fast.n_degenerate_discarded == slow.n_degenerate_discarded, (
        "both paths must discard exactly the same degenerate draws"
    )
    np.testing.assert_allclose(fast.resampled["m"], slow.resampled["m"], rtol=0, atol=1e-12)

    slow_ci, fast_ci = slow.ci("m"), fast.ci("m")
    assert (fast_ci.ci_low, fast_ci.ci_high) == pytest.approx((slow_ci.ci_low, slow_ci.ci_high))


def test_the_two_paths_agree_on_significance_across_a_model_family() -> None:
    frame = _frame()
    columns = {"good": "good", "tied": "tied"}
    metric = lambda f, col: auc_pr(f["is_malicious"].to_numpy(), f[col].to_numpy())  # noqa: E731

    slow = paired_campaign_bootstrap(frame, metric, columns, n_resamples=300, seed=5)
    fast = paired_campaign_bootstrap(
        frame, metric, columns, n_resamples=300, seed=5, fast_auc_pr=True
    )

    slow_out = slow.comparisons("auc_pr")
    fast_out = fast.comparisons("auc_pr")

    assert len(slow_out) == len(fast_out)
    for a, b in zip(slow_out, fast_out, strict=True):
        # Verdicts must match exactly: they are what the report prints.
        assert (a.model_a, a.model_b, a.significant) == (b.model_a, b.model_b, b.significant)
        assert a.p_value_raw == pytest.approx(b.p_value_raw, abs=1e-15)
        # The two paths sum the same terms in a different order, so they agree
        # to floating-point precision rather than bit-for-bit. Anything looser
        # than this would be a different estimator, not a faster one.
        assert a.diff == pytest.approx(b.diff, abs=1e-12)
        assert a.diff_ci_low == pytest.approx(b.diff_ci_low, abs=1e-12)
        assert a.diff_ci_high == pytest.approx(b.diff_ci_high, abs=1e-12)


def test_the_degenerate_count_is_a_property_of_the_draw_not_of_the_family() -> None:
    """One shared draw sequence means one degenerate count.

    Whether a resample is discarded depends only on the counts it produced —
    truncation only ever drops rows ranked below the last positive, so every
    model sees the same number of positives on every draw. Adding models to the
    family therefore cannot change the count. The reporting loop used to keep
    whichever model's count arrived last, which was right by accident; this
    pins it as an invariant instead.
    """
    frame = _frame()
    metric = lambda f, col: auc_pr(f["is_malicious"].to_numpy(), f[col].to_numpy())  # noqa: E731
    kwargs = {"n_resamples": 250, "seed": 11, "fast_auc_pr": True}

    alone = paired_campaign_bootstrap(frame, metric, {"good": "good"}, **kwargs)
    family = paired_campaign_bootstrap(frame, metric, {"good": "good", "tied": "tied"}, **kwargs)

    assert alone.n_degenerate_discarded == family.n_degenerate_discarded


@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_parallelism_changes_the_wall_clock_and_nothing_else(n_jobs: int) -> None:
    """The pairing must survive distribution.

    Every model re-seeds from the same `seed` and replays the identical draw
    sequence, and acceptance depends only on the counts — never on a model's
    scores. So which worker runs which model, and in what order they finish,
    cannot reach the numbers. Asserted rather than argued: the first real LANL
    run spent 4h49 in this loop on one core, and the fix is worthless if it
    perturbs a single resample.
    """
    frame = _frame()
    columns = {"good": "good", "tied": "tied"}
    metric = lambda f, col: auc_pr(f["is_malicious"].to_numpy(), f[col].to_numpy())  # noqa: E731

    kwargs = {"n_resamples": 200, "seed": 7, "fast_auc_pr": True}
    serial = paired_campaign_bootstrap(frame, metric, columns, n_jobs=1, **kwargs)
    parallel = paired_campaign_bootstrap(frame, metric, columns, n_jobs=n_jobs, **kwargs)

    for name in columns:
        np.testing.assert_array_equal(
            serial.resampled[name],
            parallel.resampled[name],
            err_msg=f"{name} resamples differ between n_jobs=1 and n_jobs={n_jobs}",
        )
    assert serial.n_degenerate_discarded == parallel.n_degenerate_discarded

    serial_out = [c.to_dict() for c in serial.comparisons("auc_pr")]
    parallel_out = [c.to_dict() for c in parallel.comparisons("auc_pr")]
    assert serial_out == parallel_out
