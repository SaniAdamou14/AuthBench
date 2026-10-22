"""The campaign-block bootstrap: its draws, its benign bins, and refusing to call a one-campaign split significant."""

from __future__ import annotations

import numpy as np
import polars as pl

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
    """One campaign gives a campaign-stratified bootstrap nothing to vary, so
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
