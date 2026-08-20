"""The campaign-block bootstrap: same scheme, at a size that can actually run,
and refusing to call a one-campaign split significant.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from authbench.evaluate.stats_tests import (
    MIN_CAMPAIGNS_FOR_SIGNIFICANCE,
    CampaignBlocks,
    build_campaign_blocks,
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
            "campaign_id": campaign_id,
            "is_malicious": is_malicious,
            # A weak-but-real signal, so the two models are genuinely ordered.
            "good": [0.9 if m else v for m, v in zip(is_malicious, rng.random(n), strict=True)],
            "bad": rng.random(n).tolist(),
        }
    )


def test_the_compact_and_explicit_block_forms_describe_the_same_scheme() -> None:
    frame = _frame(n_campaigns=3)

    explicit = build_campaign_blocks(frame)
    compact = CampaignBlocks.from_frame(frame)

    assert len(explicit) == compact.n_blocks
    assert compact.n_campaign_blocks == 3
    assert sorted(len(b) for b in explicit if len(b) > 1) == [4, 4, 4]
    # Every row index appears exactly once across the blocks, in either form.
    assert sorted(np.concatenate(explicit).tolist()) == list(range(frame.height))
    assert sorted(
        np.concatenate([*compact.campaign_blocks, compact.singleton_idx]).tolist()
    ) == list(range(frame.height))


def test_a_draw_returns_one_index_per_block_and_reports_its_campaign_count() -> None:
    blocks = CampaignBlocks.from_frame(_frame(n_campaigns=3))
    rng = np.random.default_rng(7)

    for _ in range(50):
        idx, n_campaign_draws = blocks.draw(rng)
        # n_blocks draws: campaigns contribute 4 rows each, singletons 1.
        assert idx.size == n_campaign_draws * 4 + (blocks.n_blocks - n_campaign_draws)
        assert 0 <= n_campaign_draws <= blocks.n_blocks


def test_the_campaign_share_of_draws_matches_the_blocks_share() -> None:
    """The compact draw is only legitimate if it is the same distribution as
    drawing uniformly from the combined block list."""
    blocks = CampaignBlocks.from_frame(_frame(n_campaigns=10, n_benign=90))
    rng = np.random.default_rng(3)

    counts = np.array([blocks.draw(rng)[1] for _ in range(400)])

    expected = blocks.n_blocks * blocks.n_campaign_blocks / blocks.n_blocks
    assert abs(counts.mean() - expected) < 3 * counts.std(ddof=1) / np.sqrt(counts.size) + 0.5


def _bootstrap(frame: pl.DataFrame, n_resamples: int = 400):
    return paired_campaign_bootstrap(
        frame,
        lambda f, col: float(np.corrcoef(f["is_malicious"].to_numpy(), f[col].to_numpy())[0, 1]),
        {"good": "good", "bad": "bad"},
        n_resamples=n_resamples,
        seed=1,
    )


def test_a_single_campaign_is_never_reported_as_significant() -> None:
    """One campaign gives a campaign-stratified bootstrap nothing to vary, so
    every p-value lands on the 2/(R+1) resolution floor and reads as a
    significant win. The floor is the bootstrap's, not the models'."""
    bootstrap = _bootstrap(_frame(n_campaigns=1))

    assert bootstrap.n_campaign_blocks == 1
    comparisons = bootstrap.comparisons("corr", alpha=0.05)
    assert comparisons, "there is still a comparison to report"
    assert not any(c.significant for c in comparisons)
    assert all("not significantly different" in c.to_dict()["sentence"] for c in comparisons)
    # The point estimate is untouched: only the verdict is withheld.
    assert comparisons[0].diff != 0.0


def test_enough_campaigns_can_still_reach_significance() -> None:
    bootstrap = _bootstrap(_frame(n_campaigns=12, n_benign=200))

    assert bootstrap.n_campaign_blocks >= MIN_CAMPAIGNS_FOR_SIGNIFICANCE
    comparisons = bootstrap.comparisons("corr", alpha=0.05)
    assert any(c.significant for c in comparisons), (
        "the guard must not suppress a genuinely resolvable difference"
    )
