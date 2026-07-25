from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from authbench.evaluate.stats_tests import (
    PairwiseComparison,
    bootstrap_ci,
    build_campaign_blocks,
    holm_bonferroni,
    permutation_test,
    render_comparison_sentence,
)


def _frame_with_campaigns() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    campaign_id = [None] * n
    # Two campaigns of 5 events each, the rest benign singletons.
    for i in range(5):
        campaign_id[i] = 1
    for i in range(5, 10):
        campaign_id[i] = 2
    is_malicious = [c is not None for c in campaign_id]
    return pl.DataFrame(
        {
            "campaign_id": campaign_id,
            "is_malicious": is_malicious,
            "score": rng.random(n).tolist(),
        }
    )


def test_build_campaign_blocks_groups_campaign_events_together() -> None:
    frame = _frame_with_campaigns()
    blocks = build_campaign_blocks(frame)

    # 2 campaign blocks + one singleton block per benign event.
    n_benign = frame.filter(pl.col("campaign_id").is_null()).height
    assert len(blocks) == 2 + n_benign
    campaign_block_sizes = sorted(len(b) for b in blocks if len(b) > 1)
    assert campaign_block_sizes == [5, 5]


def test_bootstrap_ci_contains_point_estimate_and_is_ordered() -> None:
    frame = _frame_with_campaigns()

    def metric(f: pl.DataFrame) -> float:
        return float(f["is_malicious"].sum()) / f.height

    result = bootstrap_ci(frame, metric, n_resamples=200, seed=1)
    assert result.ci_low <= result.ci_high
    assert 0.0 <= result.point_estimate <= 1.0


def test_permutation_test_p_value_in_valid_range() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(100) < 0.2).astype(int)
    scores_a = rng.random(100)
    scores_b = rng.random(100)

    def metric(y_true: np.ndarray, scores: np.ndarray) -> float:
        return float(np.mean(scores[y_true == 1]))

    p = permutation_test(y, scores_a, scores_b, metric, n_permutations=200, seed=0)
    assert 0.0 <= p <= 1.0


def test_permutation_test_identical_scores_give_p_value_one() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(50) < 0.3).astype(int)
    scores = rng.random(50)

    def metric(y_true: np.ndarray, s: np.ndarray) -> float:
        return float(np.mean(s))

    p = permutation_test(y, scores, scores.copy(), metric, n_permutations=100, seed=0)
    assert p == pytest.approx(1.0)


def test_holm_bonferroni_rejects_only_small_p_values() -> None:
    p_values = [0.001, 0.6, 0.02]
    reject = holm_bonferroni(p_values, alpha=0.05)
    assert reject[0] is True
    assert reject[1] is False


def test_render_comparison_sentence_never_says_outperforms_when_not_significant() -> None:
    comparison = PairwiseComparison(
        model_a="A", model_b="B", metric_name="auc_pr", diff=0.05, p_value=0.4, significant=False
    )
    sentence = render_comparison_sentence(comparison)
    assert "outperforms" not in sentence
    assert "not significantly different" in sentence


def test_render_comparison_sentence_says_outperforms_when_significant() -> None:
    comparison = PairwiseComparison(
        model_a="A", model_b="B", metric_name="auc_pr", diff=0.05, p_value=0.001, significant=True
    )
    sentence = render_comparison_sentence(comparison)
    assert "outperforms" in sentence
    assert "A outperforms B" in sentence
