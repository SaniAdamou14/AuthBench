"""Statistical uncertainty: campaign-stratified bootstrap CIs and paired
permutation tests with Holm-Bonferroni correction (US-128).

With 737 positives concentrated in a handful of campaigns, treating
individual events as independent draws grossly underestimates variance.
Every bootstrap resample here is drawn at the **campaign** level: each
campaign's events move together as one block, and every benign event is its
own singleton block.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def minimum_resamples_for_family(n_comparisons: int, alpha: float = 0.05) -> int:
    """Resamples needed before a family of `n_comparisons` can produce *any*
    significant result under Holm-Bonferroni.

    A percentile bootstrap cannot report a p-value below `2 / (R + 1)`; Holm's
    strictest threshold is `alpha / n_comparisons`. If the first is above the
    second, every comparison is non-significant by arithmetic, whatever the
    models did — a floor that looks exactly like a genuine "no difference"
    finding. Solving `2 / (R + 1) <= alpha / m` gives this bound.
    """
    return int(np.ceil(2 * n_comparisons / alpha)) - 1


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


@dataclass
class PairedBootstrap:
    """One campaign-stratified resampling pass, shared by every model.

    Drawing a separate set of resamples per model would make the per-model
    CIs and the pairwise comparisons answer subtly different questions, and
    it would cost `n_models` times as much. Here every model is scored on the
    *same* resample, which is what makes the differences paired: the
    campaign-composition noise that dominates variance at a ~1e-7 positive
    rate cancels between two models on a given resample instead of being
    counted twice.
    """

    model_names: list[str]
    point_estimates: dict[str, float]
    resampled: dict[str, np.ndarray]
    confidence: float
    n_resamples: int
    n_degenerate_discarded: int = 0

    @property
    def minimum_resolvable_p_value(self) -> float:
        """The smallest p-value this many resamples can express, two-sided."""
        return 2.0 / (self.n_resamples + 1)

    def ci(self, model_name: str) -> BootstrapResult:
        alpha = 1 - self.confidence
        lo, hi = np.quantile(self.resampled[model_name], [alpha / 2, 1 - alpha / 2])
        return BootstrapResult(
            point_estimate=self.point_estimates[model_name],
            ci_low=float(lo),
            ci_high=float(hi),
            confidence=self.confidence,
        )

    def comparisons(self, metric_name: str, *, alpha: float = 0.05) -> list[PairwiseComparison]:
        """Every pairwise comparison, Holm-Bonferroni corrected across the
        whole family (US-128).

        The p-value is the two-sided percentile-bootstrap p-value for
        H0: Δ = 0 — twice the smaller tail mass of the resampled difference,
        with the usual +1 continuity correction so a p-value is never
        reported as exactly zero on a finite number of resamples.

        Ties (Δ* exactly 0) count toward both tails, which is the
        conservative convention. That is only sound because degenerate
        resamples never reach here — see `paired_campaign_bootstrap`.
        """
        pairs = [(a, b) for i, a in enumerate(self.model_names) for b in self.model_names[i + 1 :]]
        ci_alpha = 1 - self.confidence

        required = minimum_resamples_for_family(len(pairs), alpha)
        if self.n_resamples < required:
            logger.warning(
                "%d resamples cannot resolve a p-value below %.5f, but Holm-Bonferroni over "
                "%d pairs demands %.5f. Every comparison will come back non-significant by "
                "arithmetic alone — indistinguishable from a real null result. Raise "
                "eval.bootstrap.n_resamples to at least %d.",
                self.n_resamples,
                self.minimum_resolvable_p_value,
                len(pairs),
                alpha / len(pairs),
                required,
            )

        raw_p_values, diffs, intervals = [], [], []
        for a, b in pairs:
            delta = self.resampled[a] - self.resampled[b]
            n_le = int(np.count_nonzero(delta <= 0))
            n_ge = int(np.count_nonzero(delta >= 0))
            tail = min(n_le, n_ge)
            raw_p_values.append(min(1.0, 2.0 * (tail + 1) / (self.n_resamples + 1)))

            diffs.append(self.point_estimates[a] - self.point_estimates[b])
            lo, hi = np.quantile(delta, [ci_alpha / 2, 1 - ci_alpha / 2])
            intervals.append((float(lo), float(hi)))

        significances = holm_bonferroni(raw_p_values, alpha=alpha)

        return [
            PairwiseComparison(
                model_a=a,
                model_b=b,
                metric_name=metric_name,
                diff=diff,
                p_value=p,
                significant=sig,
                diff_ci_low=ci[0],
                diff_ci_high=ci[1],
                method="paired_campaign_bootstrap",
            )
            for (a, b), diff, p, sig, ci in zip(
                pairs, diffs, raw_p_values, significances, intervals, strict=True
            )
        ]


def paired_campaign_bootstrap(
    frame: pl.DataFrame,
    metric_fn: Callable[[pl.DataFrame, str], float],
    score_columns: dict[str, str],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    max_attempts_factor: int = 20,
) -> PairedBootstrap:
    """Resample `frame` at the campaign-block level `n_resamples` times and
    evaluate every model in `score_columns` on each resample.

    `score_columns` maps model name -> the column holding that model's scores;
    all of them must already live on `frame`, which is what makes a single
    resampling pass serve every model. `metric_fn` takes the resampled frame
    and a score column name.

    Resamples that draw **no positive at all** are discarded and redrawn.
    Every metric here is a ranking metric, and a ranking metric over zero
    positives is undefined — but scikit-learn returns 0.0 for it rather than
    raising, so every model would tie on such a resample. Those artificial
    ties land in both tails of the two-sided test and put a floor under the
    p-value that has nothing to do with the models: with one campaign in the
    test split roughly 37% of resamples are degenerate (1 - 1/n_blocks)^n_blocks
    → e⁻¹), flooring every p-value near 0.74 and making significance
    unreachable by construction. Conditioning on a non-degenerate resample is
    the standard fix; `n_degenerate_discarded` records how often it applied,
    because a high count is itself a finding about the split.
    """
    if not score_columns:
        raise ValueError("paired_campaign_bootstrap needs at least one model to evaluate.")
    if frame.filter(pl.col("is_malicious")).height == 0:
        raise ValueError(
            "paired_campaign_bootstrap needs positives: the frame has none, so every "
            "ranking metric on it is undefined. This is a split bug, not a result."
        )

    blocks = build_campaign_blocks(frame)
    n_blocks = len(blocks)
    rng = np.random.default_rng(seed)

    model_names = list(score_columns)
    point_estimates = {name: metric_fn(frame, score_columns[name]) for name in model_names}
    resampled = {name: np.empty(n_resamples) for name in model_names}

    accepted, attempts, max_attempts = 0, 0, n_resamples * max_attempts_factor
    while accepted < n_resamples and attempts < max_attempts:
        attempts += 1
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([blocks[j] for j in chosen])
        resample = frame[idx]
        if resample.filter(pl.col("is_malicious")).height == 0:
            continue
        for name in model_names:
            resampled[name][accepted] = metric_fn(resample, score_columns[name])
        accepted += 1

    if accepted < n_resamples:
        raise ValueError(
            f"Only {accepted}/{n_resamples} non-degenerate resamples in {attempts} attempts. "
            "The test split has too few campaigns for a campaign-stratified bootstrap to "
            "say anything — report the point estimates without intervals rather than "
            "intervals nobody should trust."
        )

    return PairedBootstrap(
        model_names=model_names,
        point_estimates=point_estimates,
        resampled=resampled,
        confidence=confidence,
        n_resamples=n_resamples,
        n_degenerate_discarded=attempts - accepted,
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
    # Populated by `PairedBootstrap.comparisons`; the permutation route has no
    # sampling distribution for the difference and leaves them None.
    diff_ci_low: float | None = None
    diff_ci_high: float | None = None
    method: str = "permutation"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric_name,
            "diff": self.diff,
            "diff_ci_low": self.diff_ci_low,
            "diff_ci_high": self.diff_ci_high,
            "p_value_corrected": self.p_value,
            "significant": self.significant,
            "method": self.method,
            "sentence": render_comparison_sentence(self),
        }


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
            method="paired_permutation",
        )
        for (a, b), p, sig in zip(pairs, raw_p_values, significances, strict=True)
    ]


def render_comparison_sentence(comparison: PairwiseComparison) -> str:
    """US-128: the generator must refuse the word "outperforms" whenever the
    gap is not significant at the corrected threshold. Structurally enforced
    here — "outperforms" only appears on the `significant` branch.
    """
    swapped = comparison.diff < 0
    better, worse = (
        (comparison.model_b, comparison.model_a)
        if swapped
        else (comparison.model_a, comparison.model_b)
    )

    interval = ""
    if comparison.diff_ci_low is not None and comparison.diff_ci_high is not None:
        # The sentence reports |Δ| in `better - worse` order, so when that
        # swaps the pair the interval has to be reflected through zero as
        # well — otherwise a positive difference gets printed next to a
        # negative interval, which reads like a contradiction.
        low, high = (
            (-comparison.diff_ci_high, -comparison.diff_ci_low)
            if swapped
            else (comparison.diff_ci_low, comparison.diff_ci_high)
        )
        interval = f", 95% CI [{low:+.4f}, {high:+.4f}]"
    if comparison.significant:
        return (
            f"{better} outperforms {worse} on {comparison.metric_name} "
            f"(Δ={abs(comparison.diff):.4f}{interval}, corrected p={comparison.p_value:.4f})."
        )
    return (
        f"{comparison.model_a} and {comparison.model_b} are not significantly different "
        f"on {comparison.metric_name} (Δ={comparison.diff:.4f}{interval}, "
        f"corrected p={comparison.p_value:.4f})."
    )
