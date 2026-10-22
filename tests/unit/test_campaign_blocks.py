"""The campaign-block bootstrap: its draws, its benign bins, and refusing to call a one-campaign split significant."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from authbench.evaluate.statistics import statistics_from_frame
from authbench.evaluate.stats_tests import (
    MIN_CAMPAIGNS_FOR_SIGNIFICANCE,
    PairedBootstrap,
    _shared_draws,
    benign_bins,
    paired_campaign_bootstrap,
)


def _frame(n_campaigns: int, events_per_campaign: int = 4, n_benign: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    campaign_id: list[int | None] = []
    for c in range(1, n_campaigns + 1):
        campaign_id += [c] * events_per_campaign
    campaign_id += [None] * n_benign
    n = len(campaign_id)
    is_malicious = [c is not None for c in campaign_id]
    return pl.DataFrame(
        {
            "event_id": np.arange(n),
            "time": np.arange(n) * 10,
            "day": np.zeros(n, dtype=np.int64),
            "campaign_id": campaign_id,
            "is_malicious": is_malicious,
            # A weak-but-real signal, so the two models are genuinely ordered.
            "good": [0.9 if m else v for m, v in zip(is_malicious, rng.random(n), strict=True)],
            "bad": rng.random(n).tolist(),
        }
    )


def _bootstrap(frame: pl.DataFrame, n_resamples: int = 400) -> PairedBootstrap:
    stats = statistics_from_frame(frame, {"good": "good", "bad": "bad"}, budgets=[10], top_k=10)
    return paired_campaign_bootstrap(stats, n_resamples=n_resamples, seed=1)


def test_the_same_seed_reproduces_the_same_draws() -> None:
    first = _shared_draws(5, 300, 200, seed=3, max_attempts=4000)
    second = _shared_draws(5, 300, 200, seed=3, max_attempts=4000)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)


def test_every_kept_draw_has_a_campaign_and_the_blocks_add_up() -> None:
    multiplicity, benign, n_degenerate = _shared_draws(1, 300, 500, seed=0, max_attempts=10_000)
    assert (multiplicity.sum(axis=1) > 0).all()
    assert np.array_equal(multiplicity.sum(axis=1) + benign, np.full(500, 301))
    # One campaign among 301 blocks: a draw misses it with probability (300/301)^301, about e^-1.
    share = n_degenerate / (n_degenerate + 500)
    assert 0.30 < share < 0.44


def test_the_campaign_share_of_draws_matches_the_blocks_share() -> None:
    multiplicity, _, _ = _shared_draws(10, 90, 4000, seed=2, max_attempts=80_000)
    assert abs(multiplicity.sum(axis=1).mean() - 10.0) < 0.2


def test_benign_bins_partition_the_benign_events() -> None:
    stats = statistics_from_frame(_frame(3), {"bad": "bad"}, budgets=[10], top_k=10)["bad"]
    bins = benign_bins(stats)
    assert bins.sum() == stats.n_benign
    assert (bins >= 0).all()
    assert bins.size == 2 * stats.thresholds.size + 1


def test_a_single_campaign_is_never_reported_as_significant() -> None:
    """One campaign gives a block bootstrap nothing to vary, so
    every p-value lands on the 2/(R+1) resolution floor and reads as a
    significant win. The floor is the bootstrap's, not the models'."""
    bootstrap = _bootstrap(_frame(n_campaigns=1))

    assert bootstrap.n_campaign_blocks == 1
    comparisons = bootstrap.comparisons("auc_pr", alpha=0.05)
    assert comparisons, "there is still a comparison to report"
    assert not any(c.significant for c in comparisons)
    assert all("not significantly different" in c.to_dict()["sentence"] for c in comparisons)
    # The point estimate is untouched: only the verdict is withheld.
    assert comparisons[0].diff != 0.0


def test_enough_campaigns_can_still_reach_significance() -> None:
    bootstrap = _bootstrap(_frame(n_campaigns=12, n_benign=200))

    assert bootstrap.n_campaign_blocks >= MIN_CAMPAIGNS_FOR_SIGNIFICANCE
    comparisons = bootstrap.comparisons("auc_pr", alpha=0.05)
    assert any(c.significant for c in comparisons), (
        "the guard must not suppress a genuinely resolvable difference"
    )


def test_the_binned_jackknife_equals_leaving_out_every_unit_by_hand() -> None:
    from sklearn.metrics import average_precision_score

    from authbench.evaluate.stats_tests import jackknife

    rng = np.random.default_rng(5)
    frame = _frame(n_campaigns=4, n_benign=120).with_columns(
        pl.Series("bad", rng.integers(0, 6, 136).astype(float))
    )
    stats = statistics_from_frame(frame, {"bad": "bad"}, budgets=[10], top_k=10)["bad"]
    campaigns = stats.positives["campaign_id"].to_numpy()
    block = np.unique(campaigns, return_inverse=True)[1]
    values, weights = jackknife(stats, block, 4)

    y, s = frame["is_malicious"].to_numpy(), frame["bad"].to_numpy()
    brute = [
        average_precision_score(y[keep], s[keep])
        for keep in [frame["campaign_id"].to_numpy() != c for c in np.unique(campaigns)]
    ] + [average_precision_score(np.delete(y, i), np.delete(s, i)) for i in np.flatnonzero(~y)]
    expanded = np.repeat(values, weights.astype(int))
    np.testing.assert_allclose(np.sort(expanded), np.sort(brute), rtol=1e-12)


def test_bca_matches_scipy_on_the_same_bootstrap_distribution() -> None:
    from scipy.stats import bootstrap

    from authbench.evaluate.stats_tests import bca_interval

    data = np.random.default_rng(1).exponential(size=40)
    result = bootstrap((data,), np.mean, n_resamples=5000, method="BCa", random_state=3)
    leave_one_out = (data.sum() - data) / (data.size - 1)
    ours = bca_interval(
        result.bootstrap_distribution, float(data.mean()), leave_one_out, np.ones(data.size), 0.95
    )
    assert ours.method == "bca"
    assert ours.ci_low == pytest.approx(result.confidence_interval.low, rel=1e-9)
    assert ours.ci_high == pytest.approx(result.confidence_interval.high, rel=1e-9)
    assert ours.ci_low_mc[0] <= ours.ci_low <= ours.ci_low_mc[1]


def test_bca_falls_back_to_percentile_when_the_correction_is_undefined() -> None:
    from authbench.evaluate.stats_tests import bca_interval

    resampled = np.linspace(1.0, 2.0, 101)
    result = bca_interval(resampled, 0.5, np.array([1.0, 2.0]), np.ones(2), 0.95)
    assert result.method == "percentile"
    assert result.ci_low == pytest.approx(np.quantile(resampled, 0.025))


def _users_frame(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = 400
    malicious = rng.random(n) < 0.05
    return pl.DataFrame(
        {
            "event_id": np.arange(n),
            "time": np.sort(rng.integers(0, 3 * 86_400, n)),
            "is_malicious": malicious,
            "campaign_id": [int(i % 3) + 1 if m else None for i, m in enumerate(malicious)],
            "src_user": [f"U{u}" for u in rng.integers(0, 25, n)],
            "score": rng.integers(0, 7, n).astype(float) + malicious * rng.random(n),
        }
    ).with_columns((pl.col("time") // 86_400).alias("day"))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_user_cluster_resample_is_scikit_learns_weighted_average_precision(seed: int) -> None:
    from sklearn.metrics import average_precision_score

    from authbench.evaluate.stats_tests import cluster_average_precision

    frame = _users_frame(seed)
    stats = statistics_from_frame(
        frame, {"m": "score"}, budgets=[1], top_k=1, cluster_column="src_user"
    )["m"]
    rng = np.random.default_rng(seed)
    multiplicity = rng.integers(0, 3, (4, stats.n_clusters))
    multiplicity[:, stats.positive_cluster[0]] += 1
    assert stats.benign_cluster_bins is not None
    assert int(stats.benign_cluster_bins[2].sum()) == stats.n_benign
    assert stats.n_clusters == frame["src_user"].n_unique()
    ours = cluster_average_precision(stats, multiplicity)
    index = _cluster_index_of(frame)
    for r in range(multiplicity.shape[0]):
        weights = multiplicity[r, index].astype(float)
        expected = average_precision_score(
            frame["is_malicious"], frame["score"], sample_weight=weights
        )
        assert ours[r] == pytest.approx(expected, rel=1e-12)


def _cluster_index_of(frame: pl.DataFrame) -> np.ndarray:
    """Recover the accumulator's user numbering: positives first, then days in order."""
    from authbench.evaluate.statistics import SplitAccumulator

    accumulator = SplitAccumulator(
        frame.filter(pl.col("is_malicious")).select(
            "event_id", "campaign_id", "day", "time", "src_user", "score"
        ),
        {"m": "score"},
        budgets=[1],
        top_k=1,
        cluster_column="src_user",
    )
    index = np.empty(frame.height, dtype=np.int64)
    for day in frame.partition_by("day", maintain_order=True):
        rows = day["event_id"].to_numpy()
        index[rows] = accumulator._clusters(day["src_user"])
    return index


def test_the_user_cluster_bootstrap_is_paired_and_reproducible() -> None:
    from authbench.evaluate.stats_tests import cluster_bootstrap

    frame = _users_frame(0).with_columns(pl.col("score").alias("twin"))
    stats = statistics_from_frame(
        frame, {"m": "score", "twin": "twin"}, budgets=[1], top_k=1, cluster_column="src_user"
    )
    first = cluster_bootstrap(stats, n_resamples=300, seed=4, chunk=64)
    second = cluster_bootstrap(stats, n_resamples=300, seed=4, chunk=64)
    assert np.array_equal(first.resampled["m"], first.resampled["twin"])
    assert np.array_equal(first.resampled["m"], second.resampled["m"])
    assert first.resampled["m"].size == 300
    assert first.ci("m").method == "percentile"
