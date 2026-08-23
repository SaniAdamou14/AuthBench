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
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

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


def _resolve_worker_count(n_jobs: int, n_tasks: int) -> int:
    """How many processes to actually start.

    `n_jobs <= 0` means "one per CPU", capped at the number of models — more
    workers than tasks only adds spawn cost. Each worker holds its own copy of
    the scores and its own rank order, so on a small machine the memory ceiling,
    not the core count, is what should set this.
    """
    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    return max(1, min(n_jobs, n_tasks))


def build_campaign_blocks(frame: pl.DataFrame) -> list[np.ndarray]:
    """Row-index blocks: one block per campaign (all its events together),
    plus one singleton block per benign (non-campaign) event.

    The explicit, one-object-per-block form — readable, and what the causality
    of the scheme actually is. It is *not* what the resampling loops use: at
    LANL's test-split size that list is ~3x10^8 one-element NumPy arrays, tens
    of gigabytes of Python objects before a single resample is drawn. Those go
    through `CampaignBlocks`, which is the same scheme in two arrays.
    """
    blocks = CampaignBlocks.from_frame(frame)
    return [*blocks.campaign_blocks, *(np.array([i]) for i in blocks.singleton_idx)]


@dataclass(frozen=True)
class CampaignBlocks:
    """The campaign-block resampling scheme, stored in the shape it is used in.

    A block bootstrap over `n_blocks` blocks draws `n_blocks` of them with
    replacement. Almost every one of those blocks is a single benign event, so
    materializing them individually costs orders of magnitude more than the
    resample itself. Here the campaigns — a handful of them — stay as explicit
    index arrays and every benign event is one entry of `singleton_idx`.

    Drawing is then done in two steps, which is *exactly* equivalent to drawing
    uniformly from the combined list: each of the `n_blocks` draws lands in the
    campaign set with probability `n_campaigns / n_blocks` (hence a binomial),
    and, given that, is uniform over the campaigns.
    """

    campaign_blocks: list[np.ndarray]
    singleton_idx: np.ndarray
    has_malicious_singleton: bool

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> CampaignBlocks:
        with_idx = frame.with_row_index("_idx")
        campaign_rows = with_idx.filter(pl.col("campaign_id").is_not_null())
        # `.sort("campaign_id")` is load-bearing, not tidiness. Polars'
        # `group_by` makes no ordering guarantee — it is a multi-threaded hash
        # aggregation, and the group order varies between runs of the same
        # process on the same data. The resample then draws block index `j`
        # from a list whose `j`-th entry is a *different campaign* each run,
        # so a seeded bootstrap produced a different sampling distribution
        # every time: point estimates were stable, every confidence interval
        # and every p-value drifted. That is NFR-02 (bit-identical reruns)
        # broken exactly where it matters most, and silently — the numbers
        # stayed plausible. Sorting pins the index-to-campaign mapping.
        campaign_blocks = [
            np.asarray(idx_list, dtype=np.int64)
            for idx_list in campaign_rows.group_by("campaign_id")
            .agg(pl.col("_idx"))
            .sort("campaign_id")["_idx"]
            .to_list()
        ]
        benign = with_idx.filter(pl.col("campaign_id").is_null())
        # A malicious event with no campaign_id shouldn't happen — labeling and
        # campaign attachment join on the same quadruplet — but if one did, the
        # cheap "no campaign block drawn ⇒ no positives" shortcut below would be
        # wrong, so the possibility is checked once rather than assumed away.
        has_malicious_singleton = (
            "is_malicious" in benign.columns and benign.filter(pl.col("is_malicious")).height > 0
        )
        return cls(
            campaign_blocks=campaign_blocks,
            singleton_idx=benign["_idx"].to_numpy().astype(np.int64, copy=False),
            has_malicious_singleton=has_malicious_singleton,
        )

    @property
    def n_campaign_blocks(self) -> int:
        return len(self.campaign_blocks)

    @property
    def n_blocks(self) -> int:
        return self.n_campaign_blocks + int(self.singleton_idx.size)

    def draw_counts(self, rng: np.random.Generator, n_rows: int) -> tuple[np.ndarray, int]:
        """One resample expressed as a per-row multiplicity vector.

        The same draw as `draw`, in the form the ranking-based metrics want. A
        resample of a 50-million-row split is 50 million row indices; gathering
        a Polars frame on them costs more than the metric does. A count vector
        is the same information in a fixed-size array, and every metric here is
        a weighted sum over rows.
        """
        idx, n_campaign_draws = self.draw(rng)
        counts = np.bincount(idx, minlength=n_rows).astype(np.int32, copy=False)
        return counts, n_campaign_draws

    def draw(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        """One resample: the row indices it selects, and how many of the drawn
        blocks were campaigns (0 means the resample holds no campaign event).
        """
        n_campaigns = self.n_campaign_blocks
        n_singletons = int(self.singleton_idx.size)
        n_blocks = self.n_blocks
        if n_blocks == 0:
            return np.empty(0, dtype=np.int64), 0

        n_campaign_draws = int(rng.binomial(n_blocks, n_campaigns / n_blocks)) if n_campaigns else 0
        parts: list[np.ndarray] = []
        if n_campaign_draws:
            chosen = rng.integers(0, n_campaigns, size=n_campaign_draws)
            parts.extend(self.campaign_blocks[j] for j in chosen)
        n_singleton_draws = n_blocks - n_campaign_draws
        if n_singleton_draws and n_singletons:
            parts.append(self.singleton_idx[rng.integers(0, n_singletons, size=n_singleton_draws)])

        if not parts:
            return np.empty(0, dtype=np.int64), n_campaign_draws
        return np.concatenate(parts), n_campaign_draws


@dataclass(frozen=True)
class RankedScores:
    """One model's test scores, pre-sorted once so every resample is O(n).

    The cost of `average_precision_score` is dominated by sorting, and a
    bootstrap sorts the *same* scores again for every resample: 2,000
    resamples times 8 models times an argsort of 50 million rows is tens of
    hours, which is the difference between a benchmark that runs on a real
    dataset overnight and one that does not.

    Sorting once and walking the fixed order under a per-row weight vector
    gives exactly the same number — `average_precision_score(y, s,
    sample_weight=counts)` to machine precision, ties included, which
    `tests/unit/test_fast_auc_pr.py` asserts against scikit-learn directly.
    """

    order: np.ndarray
    y_sorted: np.ndarray
    #: Index of the last row of each tied-score group. Average precision steps
    #: once per distinct score, never per row, so tied events must move
    #: together or precision is read at a threshold that splits them.
    group_end: np.ndarray

    @classmethod
    def from_arrays(cls, y_true: np.ndarray, scores: np.ndarray) -> RankedScores:
        # int32 for the permutation and uint8 for the label mask, not int64 and
        # float64. Both are exact — a row index below 2^31 and a 0/1 label lose
        # nothing — and together they cut this structure from 16 bytes per row
        # to 5. On a split of 36 million events that is the difference between
        # 0.6 GB and 0.2 GB per model, on a machine with 8 GB total.
        order = np.argsort(-scores, kind="stable").astype(np.int32, copy=False)
        scores_sorted = scores[order]
        y_sorted = (y_true[order] != 0).astype(np.uint8, copy=False)

        # Everything ranked below the last positive is dropped, and this is
        # exact rather than an approximation: average precision sums one term
        # per *threshold at which recall increases*, and recall stops
        # increasing once the last positive has been passed. Those rows enter
        # no term at all.
        #
        # It matters because the per-resample cost is dominated by gathering
        # weights in rank order, a cache-hostile random read over the whole
        # split. At a positive rate near 1e-7 a model that ranks the attacks
        # anywhere near the top turns tens of millions of rows into a few
        # hundred thousand. A model that ranks them at random keeps most of
        # them — which is the honest outcome, since the floors are exactly the
        # models with nothing to exploit.
        positives = np.flatnonzero(y_sorted)
        if positives.size:
            last_positive = int(positives[-1])
            # Extend to the end of that score's tie group: those rows share the
            # threshold and so share the final precision term.
            tie_value = scores_sorted[last_positive]
            cut = int(np.searchsorted(-scores_sorted, -tie_value, side="right"))
            order = order[:cut]
            scores_sorted = scores_sorted[:cut]
            y_sorted = y_sorted[:cut]

        is_last = np.empty(scores_sorted.size, dtype=bool)
        is_last[:-1] = scores_sorted[:-1] != scores_sorted[1:]
        if is_last.size:
            is_last[-1] = True
        return cls(
            order=order,
            y_sorted=y_sorted,
            group_end=np.flatnonzero(is_last).astype(np.int32, copy=False),
        )

    def average_precision(self, counts: np.ndarray) -> float:
        """Weighted average precision for a resample given as row multiplicities."""
        weights = counts[self.order].astype(np.float64, copy=False)
        true_positives = np.cumsum(weights * self.y_sorted)  # uint8 mask promotes
        total_positives = true_positives[-1] if true_positives.size else 0.0
        if total_positives <= 0:
            # Undefined, not zero. Callers discard these resamples; returning
            # 0.0 the way scikit-learn does would tie every model together and
            # put a floor under every p-value. See `paired_campaign_bootstrap`.
            return float("nan")

        false_positives = np.cumsum(weights) - true_positives
        tp_at = true_positives[self.group_end]
        fp_at = false_positives[self.group_end]
        recall = tp_at / total_positives
        precision = tp_at / np.maximum(tp_at + fp_at, 1e-12)
        return float(np.sum(np.diff(recall, prepend=0.0) * precision))


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
    blocks = CampaignBlocks.from_frame(frame)
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(frame)

    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx, _ = blocks.draw(rng)
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


@dataclass(frozen=True)
class _ModelBootstrapTask:
    """Everything one worker needs to bootstrap one model, and nothing more.

    A plain dataclass of arrays rather than a closure, because Windows spawns
    worker processes rather than forking them: every argument is pickled, so it
    has to be picklable and it has to be small enough to be worth sending.
    """

    name: str
    y_true: np.ndarray
    scores: np.ndarray
    blocks: CampaignBlocks
    n_resamples: int
    seed: int
    max_attempts: int


def _bootstrap_one_model(task: _ModelBootstrapTask) -> tuple[str, np.ndarray, int, int]:
    """Resample one model. Returns (name, values, n_kept_rows, n_degenerate).

    Module-level and self-contained so a process pool can call it. Every model
    re-seeds from the *same* `seed` and therefore replays the identical draw
    sequence — which is what keeps the comparisons paired no matter how the
    work is distributed, or in what order the workers happen to finish.
    """
    ranked = RankedScores.from_arrays(task.y_true, task.scores)
    rng = np.random.default_rng(task.seed)
    values = np.empty(task.n_resamples)
    n_rows = task.y_true.size

    accepted, attempts = 0, 0
    while accepted < task.n_resamples and attempts < task.max_attempts:
        attempts += 1
        counts, n_campaign_draws = task.blocks.draw_counts(rng, n_rows)
        if n_campaign_draws == 0 and not task.blocks.has_malicious_singleton:
            continue
        value = ranked.average_precision(counts)
        if np.isnan(value):
            continue
        values[accepted] = value
        accepted += 1

    if accepted < task.n_resamples:
        raise ValueError(
            f"{task.name}: only {accepted}/{task.n_resamples} non-degenerate resamples in "
            f"{attempts} attempts. The test split has too few campaigns for a "
            "campaign-stratified bootstrap to say anything — report the point estimates "
            "without intervals rather than intervals nobody should trust."
        )
    return task.name, values, int(ranked.order.size), attempts - accepted


def paired_campaign_bootstrap(
    frame: pl.DataFrame,
    metric_fn: Callable[[pl.DataFrame, str], float],
    score_columns: dict[str, str],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    max_attempts_factor: int = 20,
    fast_auc_pr: bool = False,
    n_jobs: int = 1,
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

    `fast_auc_pr=True` computes each resample's average precision from a
    pre-sorted rank order (`RankedScores`) instead of gathering a full-size
    Polars frame and re-sorting it. It is a specialization, not an
    approximation: the value is weighted average precision, identical to
    scikit-learn's to machine precision, and `tests/unit/test_fast_auc_pr.py`
    asserts the whole bootstrap returns the same numbers either way. Only the
    cost differs — the difference between hours and days on a real split.
    `metric_fn` still supplies the point estimates, and must be AUC-PR for the
    two to describe the same quantity.
    """
    if not score_columns:
        raise ValueError("paired_campaign_bootstrap needs at least one model to evaluate.")
    if frame.filter(pl.col("is_malicious")).height == 0:
        raise ValueError(
            "paired_campaign_bootstrap needs positives: the frame has none, so every "
            "ranking metric on it is undefined. This is a split bug, not a result."
        )

    blocks = CampaignBlocks.from_frame(frame)
    rng = np.random.default_rng(seed)

    model_names = list(score_columns)
    point_estimates = {name: metric_fn(frame, score_columns[name]) for name in model_names}
    resampled = {name: np.empty(n_resamples) for name in model_names}

    max_attempts = n_resamples * max_attempts_factor

    if fast_auc_pr:
        logger.info(
            "Bootstrapping AUC-PR over %d rows x %d models by pre-sorted rank order.",
            frame.height,
            len(model_names),
        )
        y_true = frame["is_malicious"].to_numpy()

        # One task per model, each replaying the same seeded draw sequence.
        #
        # The models are strictly independent — no shared state, and whether a
        # draw is accepted depends only on the counts, never on any model's
        # scores — so distributing them changes nothing about the result and
        # divides the wall time by the number of workers. The first real LANL
        # run spent 4h49 here on a single core while eleven sat idle.
        tasks = [
            _ModelBootstrapTask(
                name=name,
                y_true=y_true,
                scores=frame[score_columns[name]].to_numpy(),
                blocks=blocks,
                n_resamples=n_resamples,
                seed=seed,
                max_attempts=max_attempts,
            )
            for name in model_names
        ]
        workers = _resolve_worker_count(n_jobs, len(tasks))
        logger.info("Bootstrapping %d models across %d worker process(es).", len(tasks), workers)

        degenerate = 0
        started = time.monotonic()
        if workers == 1:
            results = [_bootstrap_one_model(task) for task in tasks]
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_bootstrap_one_model, tasks))

        for name, values, n_kept, n_degenerate in results:
            resampled[name] = values
            degenerate = n_degenerate
            logger.info(
                "  %s: %d of %d rows survived truncation (%.1f%%), %d degenerate draws redrawn",
                name,
                n_kept,
                frame.height,
                100.0 * n_kept / max(1, frame.height),
                n_degenerate,
            )
        logger.info("Bootstrap finished in %.1f s.", time.monotonic() - started)

        return PairedBootstrap(
            model_names=model_names,
            point_estimates=point_estimates,
            resampled=resampled,
            confidence=confidence,
            n_resamples=n_resamples,
            n_degenerate_discarded=degenerate,
            n_campaign_blocks=blocks.n_campaign_blocks,
        )

    accepted, attempts = 0, 0
    while accepted < n_resamples and attempts < max_attempts:
        attempts += 1

        idx, n_campaign_draws = blocks.draw(rng)
        # A resample that drew no campaign block holds no positive, unless the
        # frame has malicious events outside every campaign. Deciding that from
        # the draw itself skips gathering a full-size frame for a resample that
        # is about to be discarded — and with one campaign in the split that is
        # roughly every third draw.
        if n_campaign_draws == 0 and not blocks.has_malicious_singleton:
            continue
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
        n_campaign_blocks=blocks.n_campaign_blocks,
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
