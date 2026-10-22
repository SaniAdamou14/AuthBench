"""Statistical uncertainty: a paired campaign-block bootstrap with Holm-Bonferroni correction (US-128).

With 715 positives concentrated in a handful of campaigns, treating
individual events as independent draws grossly underestimates variance.
Every bootstrap resample here is drawn at the **campaign** level: each
campaign's events move together as one block, and every benign event is its
own singleton block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from authbench.evaluate.statistics import (
    ModelStatistics,
    average_precision,
    positive_weight_by_group,
    weighted_average_precision,
)

logger = logging.getLogger(__name__)

# The sample size of a campaign-stratified bootstrap is the number of
# campaigns, not the number of events. With a single campaign every resample
# is a re-weighting of the same attack, so no pairwise difference can change
# sign and every p-value lands on the resolution floor — significance is not
# estimable, and `PairedBootstrap.comparisons` says so instead of reporting it.
MIN_CAMPAIGNS_FOR_SIGNIFICANCE = 2


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


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence: float


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
    # How many campaigns the resampled split actually contained. This is the
    # sample size of a campaign-stratified bootstrap — not the event count —
    # and `comparisons` refuses to call anything significant below
    # `MIN_CAMPAIGNS_FOR_SIGNIFICANCE`.
    n_campaign_blocks: int = 0

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

        raw_p_values, diffs, intervals, at_floor = [], [], [], []
        for a, b in pairs:
            delta = self.resampled[a] - self.resampled[b]
            n_le = int(np.count_nonzero(delta <= 0))
            n_ge = int(np.count_nonzero(delta >= 0))
            tail = min(n_le, n_ge)
            raw_p_values.append(min(1.0, 2.0 * (tail + 1) / (self.n_resamples + 1)))
            # `tail == 0` is the entire resolution story, recorded rather than
            # left to be inferred from the printed number: not one resample out
            # of `n_resamples` put the difference on the other side of zero, so
            # the p-value above is the floor `2/(R+1)` and nothing was measured
            # beneath it. The comparison then supports "p is at most this", not
            # "p is this" — and on both published runs every single significant
            # pair is one of these.
            at_floor.append(tail == 0)

            diffs.append(self.point_estimates[a] - self.point_estimates[b])
            lo, hi = np.quantile(delta, [ci_alpha / 2, 1 - ci_alpha / 2])
            intervals.append((float(lo), float(hi)))

        significances = holm_bonferroni(raw_p_values, alpha=alpha)
        adjusted_p_values = holm_adjusted_p_values(raw_p_values)

        # A campaign-stratified bootstrap has as many independent observations
        # as the split has campaigns. Below two, resampling cannot vary the
        # campaign composition at all: every resample contains copies of the
        # same campaign, so each model's Δ* keeps one sign across the whole
        # pass, the tail count is 0, and *every* pair comes back at exactly
        # the resolution floor 2/(R+1) — printed as "outperforms", 21 times
        # out of 21, from a single attack. That is the resolution of the
        # bootstrap being reported as evidence about the models, and it is
        # precisely the claim this project exists to refuse to make.
        if self.n_campaign_blocks < MIN_CAMPAIGNS_FOR_SIGNIFICANCE:
            logger.warning(
                "%d campaign(s) in the resampled split: a campaign-stratified bootstrap has "
                "no campaign-level variation to draw on, so every pairwise p-value collapses "
                "onto its own resolution floor (%.5f). All %d comparisons are reported as "
                "not significant — the point estimates and intervals still stand, the "
                "significance verdict does not.",
                self.n_campaign_blocks,
                self.minimum_resolvable_p_value,
                len(pairs),
            )
            significances = [False] * len(pairs)

        return [
            PairwiseComparison(
                model_a=a,
                model_b=b,
                metric_name=metric_name,
                diff=diff,
                p_value_raw=p,
                significant=sig,
                p_value_holm_adjusted=adjusted,
                at_resolution_floor=floored,
                diff_ci_low=ci[0],
                diff_ci_high=ci[1],
                method="paired_campaign_bootstrap",
            )
            for (a, b), diff, p, adjusted, sig, ci, floored in zip(
                pairs,
                diffs,
                raw_p_values,
                adjusted_p_values,
                significances,
                intervals,
                at_floor,
                strict=True,
            )
        ]


def _shared_draws(
    n_campaigns: int, n_benign: int, n_resamples: int, seed: int, max_attempts: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Campaign multiplicities and benign draw count of each kept resample, and how many were redrawn.

    A resample draws `n_blocks` blocks with replacement from the campaigns and the benign
    singletons. How many land on campaigns is binomial, and given that the campaign
    multiplicities are multinomial and uniform. A resample with no campaign holds no
    positive and is redrawn (see `paired_campaign_bootstrap`).
    """
    rng = np.random.default_rng(seed)
    n_blocks = n_campaigns + n_benign
    campaign_draws: list[np.ndarray] = []
    multiplicities: list[np.ndarray] = []
    n_kept = drawn = 0
    while n_kept < n_resamples and drawn < max_attempts:
        batch = min(n_resamples, max_attempts - drawn)
        n_campaign_draws = rng.binomial(n_blocks, n_campaigns / n_blocks, size=batch)
        campaign_draws.append(n_campaign_draws)
        multiplicities.append(
            rng.multinomial(n_campaign_draws, np.full(n_campaigns, 1 / n_campaigns))
        )
        n_kept += int((n_campaign_draws > 0).sum())
        drawn += batch
    all_draws = np.concatenate(campaign_draws)
    kept = np.flatnonzero(all_draws > 0)
    if kept.size < n_resamples:
        raise ValueError(
            f"Only {kept.size}/{n_resamples} non-degenerate resamples in {drawn} attempts. "
            "The test split has too few campaigns for a campaign-stratified bootstrap to say "
            "anything; report the point estimates without intervals rather than intervals "
            "nobody should trust."
        )
    kept = kept[:n_resamples]
    n_degenerate = int(kept[-1] + 1 - n_resamples)
    return np.concatenate(multiplicities)[kept], n_blocks - all_draws[kept], n_degenerate


def benign_bins(stats: ModelStatistics) -> np.ndarray:
    """Benign events between and level with consecutive thresholds: gap, level, gap, level, ..., rest."""
    above, level = stats.benign_above, stats.benign_level
    gaps = above - np.concatenate([[0], above[:-1] + level[:-1]])
    rest = stats.n_benign - (int(above[-1] + level[-1]) if above.size else 0)
    bins = np.empty(2 * above.size + 1, dtype=np.int64)
    bins[0:-1:2], bins[1:-1:2], bins[-1] = gaps, level, rest
    if (bins < 0).any():
        raise ValueError(f"{stats.model_name}: benign counts are inconsistent with its thresholds.")
    return bins


def paired_campaign_bootstrap(
    statistics: dict[str, ModelStatistics],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    max_attempts_factor: int = 20,
) -> PairedBootstrap:
    """AUC-PR of every model on `n_resamples` campaign-block resamples of the split.

    Drawing every benign singleton's multiplicity costs the size of the split per
    resample, about 4e8 draws on LANL. AUC-PR depends on the benign weights only
    through the benign weight at or above each positive threshold, so each model's
    benign draws are taken directly as a multinomial over its `benign_bins`, which has
    exactly the distribution the per-event draw induces on those counts. The campaign
    multiplicities, the number of benign draws and the redraws are shared by every
    model, and each model's benign split restarts one generator from the same seed, so
    two models with the same bins (a model and itself, in particular) get the same
    draws. The per-model intervals have exactly the distribution of the per-event
    scheme; the comparisons are paired exactly in the campaign component, which carries
    the variance at a positive rate near 1e-7. Measured against the per-event scheme on
    the demo split with 20,000 resamples: intervals equal to four decimals, the same
    significance verdict on 28 of 28 pairs, p-values within 0.006.

    Resamples that draw no campaign hold no positive and are redrawn: AUC-PR over zero
    positives is undefined, and scikit-learn's 0.0 for it would tie every model on such
    a resample and floor every p-value (about 37% of resamples with one campaign).
    `n_degenerate_discarded` records how often that happened.
    """
    if not statistics:
        raise ValueError("paired_campaign_bootstrap needs at least one model to evaluate.")
    names = list(statistics)
    reference = statistics[names[0]]
    if not reference.positives.height:
        raise ValueError(
            "paired_campaign_bootstrap needs positives: the split has none, so every ranking "
            "metric on it is undefined. This is a split bug, not a result."
        )
    for name in names[1:]:
        other = statistics[name]
        if other.n_benign != reference.n_benign or not other.positives["event_id"].equals(
            reference.positives["event_id"]
        ):
            raise ValueError(f"{name} was not evaluated on the same split as {names[0]}.")

    campaigns, campaign_index = np.unique(
        reference.positives["campaign_id"].to_numpy(), return_inverse=True
    )
    multiplicity, benign_draws, n_degenerate = _shared_draws(
        campaigns.size, reference.n_benign, n_resamples, seed, n_resamples * max_attempts_factor
    )
    positive_weight = multiplicity[:, campaign_index]

    resampled = {}
    for name in names:
        stats = statistics[name]
        bins = benign_bins(stats)
        rng = np.random.default_rng([seed, 1])
        benign = (
            rng.multinomial(benign_draws, bins / stats.n_benign)
            if stats.n_benign
            else np.zeros((n_resamples, bins.size), dtype=np.int64)
        )
        at_or_above = np.cumsum(benign, axis=1)[:, 1:-1:2].astype(np.float64)
        resampled[name] = weighted_average_precision(
            positive_weight_by_group(stats, positive_weight), at_or_above
        )
    logger.info(
        "Bootstrap: %d resamples x %d models over %d campaigns, %d degenerate draws redrawn.",
        n_resamples,
        len(names),
        campaigns.size,
        n_degenerate,
    )
    return PairedBootstrap(
        model_names=names,
        point_estimates={name: average_precision(statistics[name]) for name in names},
        resampled=resampled,
        confidence=confidence,
        n_resamples=n_resamples,
        n_degenerate_discarded=n_degenerate,
        n_campaign_blocks=int(campaigns.size),
    )


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns, per original index,
    whether that comparison is significant at the family-wise `alpha`.
    """
    m = len(p_values)
    order = np.argsort(p_values, kind="stable")
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


def holm_adjusted_p_values(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values, returned per original index.

    `holm_bonferroni` answers "reject at *this* alpha?". This answers "at what
    alpha would this comparison start to be rejected?", which is the number a
    reader can hold against any threshold instead of only against the one the
    run happened to use — and the number that belongs under a key called
    `p_value_holm_adjusted`, where this project published the *uncorrected*
    value for most of its history.

    Sorting ascending, the adjusted value at rank `i` is
    `max_{j <= i} (m - j) * p_(j)`, clipped to 1. The running maximum is what
    keeps the sequence monotone, and monotonicity is what makes
    "adjusted <= alpha" agree with the step-down rule at every alpha at once —
    `tests/unit/test_stats_tests.py` pins the two together rather than trusting
    that they cannot drift apart.

    The agreement holds for every `alpha < 1` and not at `alpha == 1`, which is
    the clipping and not a disagreement about the procedure: an adjusted
    p-value is a p-value and may not exceed 1, so a family whose step-down rule
    rejects nothing still ends up with adjusted values of exactly 1.0, and
    `<= 1.0` then admits them. No significance level is 1.
    """
    m = len(p_values)
    if m == 0:
        return []

    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(np.argsort(p_values, kind="stable")):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


@dataclass
class PairwiseComparison:
    model_a: str
    model_b: str
    metric_name: str
    diff: float
    #: The **uncorrected** two-sided p-value for this pair on its own. Named
    #: for what it is: the JSON key used to say `p_value_corrected` and the
    #: sentence used to say "corrected p", while Holm was only ever applied as
    #: a decision rule and never folded into this number.
    p_value_raw: float
    significant: bool
    #: `p_value_raw` corrected for multiplicity across the whole family.
    #: `None` on routes that report no family.
    p_value_holm_adjusted: float | None = None
    #: True when `p_value_raw` is the smallest value the test is *able* to
    #: express rather than one it measured — no resample put
    #: the difference on the far side of zero. The p-value is then an upper
    #: bound and the sentence says `<=` instead of `=`, because reporting a
    #: floor as a measurement claims a precision the run does not have.
    at_resolution_floor: bool = False
    diff_ci_low: float | None = None
    diff_ci_high: float | None = None
    method: str = "paired_campaign_bootstrap"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric_name,
            "diff": self.diff,
            "diff_ci_low": self.diff_ci_low,
            "diff_ci_high": self.diff_ci_high,
            "p_value_raw": self.p_value_raw,
            "p_value_holm_adjusted": self.p_value_holm_adjusted,
            "at_resolution_floor": self.at_resolution_floor,
            "significant": self.significant,
            "method": self.method,
            "sentence": render_comparison_sentence(self),
        }


def render_p_value(comparison: PairwiseComparison) -> str:
    """How a comparison's p-value is allowed to be written down.

    Two things the printed form has to get right, both of which it used to get
    wrong. It reports the **Holm-adjusted** value when there is one, because
    the sentence already says "corrected" and the number beside it was the
    uncorrected one. And it writes `<=` whenever the test bottomed out at its
    own resolution: a bootstrap that never once put the difference on the far
    side of zero has shown that p is *below* `2/(R+1)`, not that it *equals*
    it, and every significant pair in both published runs is exactly that
    case. Printing `=` there turns the number of resamples into a measurement.
    """
    adjusted = comparison.p_value_holm_adjusted
    label, value = (
        ("Holm-adjusted p", adjusted) if adjusted is not None else ("p", comparison.p_value_raw)
    )
    relation = "<=" if comparison.at_resolution_floor else "="
    return f"{label} {relation} {value:.4f}"


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
    p_value = render_p_value(comparison)
    if comparison.significant:
        return (
            f"{better} outperforms {worse} on {comparison.metric_name} "
            f"(Δ={abs(comparison.diff):.4f}{interval}, {p_value})."
        )
    return (
        f"{comparison.model_a} and {comparison.model_b} are not significantly different "
        f"on {comparison.metric_name} (Δ={comparison.diff:.4f}{interval}, {p_value})."
    )
