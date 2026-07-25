"""Statistical uncertainty: campaign-stratified bootstrap CIs and paired
permutation tests with Holm-Bonferroni correction (US-128).

With 737 positives concentrated in a handful of campaigns, treating
individual events as independent draws grossly underestimates variance.
Every bootstrap resample here is drawn at the **campaign** level: each
campaign's events move together as one block, and every benign event is its
own singleton block.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl


def build_campaign_blocks(frame: pl.DataFrame) -> list[np.ndarray]:
    """Row-index blocks: one block per campaign (all its events together),
    plus one singleton block per benign (non-campaign) event.
    """
    with_idx = frame.with_row_index("_idx")

    campaign_blocks = [
        np.array(idx_list)
        for idx_list in with_idx.filter(pl.col("campaign_id").is_not_null())
        .group_by("campaign_id")
        .agg(pl.col("_idx"))["_idx"]
        .to_list()
    ]
    singleton_idx = with_idx.filter(pl.col("campaign_id").is_null())["_idx"].to_numpy()
    singleton_blocks = [np.array([i]) for i in singleton_idx]

    return campaign_blocks + singleton_blocks


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence: float


def bootstrap_ci(
    frame: pl.DataFrame,
    metric_fn: Callable[[pl.DataFrame], float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Campaign-stratified bootstrap confidence interval for `metric_fn(frame)`."""
    blocks = build_campaign_blocks(frame)
    n_blocks = len(blocks)
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(frame)

    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([blocks[j] for j in chosen])
        resampled = frame[idx]
        stats[i] = metric_fn(resampled)

    alpha = 1 - confidence
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        point_estimate=point_estimate, ci_low=float(lo), ci_high=float(hi), confidence=confidence
    )


def permutation_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> float:
    """Paired permutation test on `metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b)`.

    At each permutation, each event's two model scores are swapped
    independently with probability 0.5 — the null hypothesis being that the
    two models are exchangeable per event.
    """
    observed_diff = metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        swap = rng.integers(0, 2, size=n).astype(bool)
        perm_a = np.where(swap, scores_b, scores_a)
        perm_b = np.where(swap, scores_a, scores_b)
        diffs[i] = metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b)

    return float((np.sum(np.abs(diffs) >= abs(observed_diff)) + 1) / (n_permutations + 1))


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns, per original index,
    whether that comparison is significant at the family-wise `alpha`.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


@dataclass
class PairwiseComparison:
    model_a: str
    model_b: str
    metric_name: str
    diff: float
    p_value: float
    significant: bool


def compare_models(
    model_names: list[str],
    y_true: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str,
    *,
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[PairwiseComparison]:
    """All pairwise comparisons among `model_names`, Holm-Bonferroni corrected
    across the whole family of comparisons (US-128).
    """
    pairs = [(a, b) for i, a in enumerate(model_names) for b in model_names[i + 1 :]]
    raw_p_values = [
        permutation_test(
            y_true,
            scores_by_model[a],
            scores_by_model[b],
            metric_fn,
            n_permutations=n_permutations,
            seed=seed,
        )
        for a, b in pairs
    ]
    significances = holm_bonferroni(raw_p_values, alpha=alpha)

    return [
        PairwiseComparison(
            model_a=a,
            model_b=b,
            metric_name=metric_name,
            diff=metric_fn(y_true, scores_by_model[a]) - metric_fn(y_true, scores_by_model[b]),
            p_value=p,
            significant=sig,
        )
        for (a, b), p, sig in zip(pairs, raw_p_values, significances, strict=True)
    ]


def render_comparison_sentence(comparison: PairwiseComparison) -> str:
    """US-128: the generator must refuse the word "outperforms" whenever the
    gap is not significant at the corrected threshold. Structurally enforced
    here — "outperforms" only appears on the `significant` branch.
    """
    better, worse = (
        (comparison.model_a, comparison.model_b)
        if comparison.diff >= 0
        else (comparison.model_b, comparison.model_a)
    )
    if comparison.significant:
        return (
            f"{better} outperforms {worse} on {comparison.metric_name} "
            f"(Δ={abs(comparison.diff):.4f}, corrected p={comparison.p_value:.4f})."
        )
    return (
        f"{comparison.model_a} and {comparison.model_b} are not significantly different "
        f"on {comparison.metric_name} (Δ={comparison.diff:.4f}, corrected p={comparison.p_value:.4f})."
    )
