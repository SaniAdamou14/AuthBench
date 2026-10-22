"""Sufficient statistics of a scored split, accumulated one day at a time.

Every metric this benchmark reports depends on the scores only through the
positives: each positive's rank in its own day's alert queue (and under the two
tie-breaks that bracket it), how many benign events of the whole split score
above it and level with it, and, per day, the tie at each budget and the
best-ranked events. `SplitAccumulator` computes these from one day of scores at
a time, so evaluating a split needs the memory of its largest day rather than of
the split. The positives' scores are given up front, from scoring the few
hundred positive rows on their own, and every day checks that it reproduces
them bit for bit; that holds because every model in the catalog is row-local to
the last bit (`tests/unit/test_row_locality.py`).

`tests/unit/test_statistics.py` checks every derived metric against its
whole-frame definition in `evaluate.budget`, `evaluate.campaign`,
`evaluate.ledger` and scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from authbench.evaluate.budget import BudgetCurve, TieExposure
from authbench.evaluate.campaign import TimeToDetectionResult

POSITIVE_KEYS = ["event_id", "campaign_id", "day", "time"]
RANK_COLUMNS = ["rank", "rank_best", "rank_worst", "rank_against", "rank_for"]


@dataclass(frozen=True)
class ModelStatistics:
    """What one model's scores on a split reduce to, for every reported metric."""

    model_name: str
    #: One row per positive in (day, time, event_id) order: `POSITIVE_KEYS`, `score`, the queue
    #: `rank` in its day, the tie bounds `rank_best` and `rank_worst`, and the ranks with ties
    #: broken against the model (`rank_against`) and for it (`rank_for`).
    positives: pl.DataFrame
    #: Distinct positive scores, descending, and the benign events of the split strictly above
    #: and exactly level with each.
    thresholds: np.ndarray
    benign_above: np.ndarray
    benign_level: np.ndarray
    n_benign: int
    #: `day`, `events`, `positives`.
    day_sizes: pl.DataFrame
    tie_exposure: list[TieExposure]
    #: The split's first events in queue order: `score`, `time`, `event_id`, `is_malicious`.
    top: pl.DataFrame
    min_distinct_scores_in_a_day: int
    largest_tie_fraction_in_a_day: float

    @property
    def n_events(self) -> int:
        return self.n_benign + self.positives.height

    def threshold_group(self) -> np.ndarray:
        """Index in `thresholds` of each positive's score."""
        return np.searchsorted(-self.thresholds, -self.positives["score"].to_numpy())


def _position_among_ties(scores: np.ndarray, *keys: np.ndarray) -> np.ndarray:
    """0-based position of each element among those with an equal score, ordered by `keys`."""
    order = np.lexsort((*reversed(keys), -scores))
    ordered = scores[order]
    index = np.arange(scores.size)
    starts = np.ones(scores.size, dtype=bool)
    starts[1:] = ordered[1:] != ordered[:-1]
    position = np.empty(scores.size, dtype=np.int64)
    position[order] = index - np.maximum.accumulate(np.where(starts, index, 0))
    return position


@dataclass
class _Day:
    """One day's columns, shared by every model's update."""

    time: np.ndarray
    event_id: np.ndarray
    malicious: np.ndarray
    arrival: np.ndarray
    positive_rows: np.ndarray
    positive_index: np.ndarray


@dataclass
class _ModelState:
    scores: np.ndarray
    thresholds: np.ndarray
    above: np.ndarray
    level: np.ndarray
    ranks: dict[str, np.ndarray]
    contested: dict[int, list[int]]
    top: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    min_distinct: int | None = None
    largest_tie: float = 0.0


class SplitAccumulator:
    """Folds a split's days, in any order, into one `ModelStatistics` per model."""

    def __init__(
        self,
        positives: pl.DataFrame,
        score_columns: dict[str, str],
        *,
        budgets: list[int],
        top_k: int,
    ) -> None:
        """`positives`: every malicious event of the split, with `POSITIVE_KEYS` and each model's score."""
        if positives["campaign_id"].null_count():
            raise ValueError(
                "A malicious event has no campaign_id. Labelling and campaign attachment join "
                "on the same red-team key, so this is a labelling bug, not a result."
            )
        self._positives = (
            positives.sort("campaign_id", "time", "event_id")
            .with_columns(pl.int_range(pl.len()).over("campaign_id").alias("_campaign_event_index"))
            .sort("day", "time", "event_id")
        )
        self._row = {int(e): i for i, e in enumerate(self._positives["event_id"].to_list())}
        self._campaign_event_index = self._positives["_campaign_event_index"].to_numpy()
        self._expected_per_day = {
            int(day): int(count) for day, count in self._positives.group_by("day").len().iter_rows()
        }
        self._budgets = list(budgets)
        self._top_k = top_k
        self._days: dict[int, tuple[int, int]] = {}
        self._n_benign = 0
        empty_top = (
            np.empty(0),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=bool),
        )
        self._models: dict[str, _ModelState] = {}
        for name, column in score_columns.items():
            scores = self._positives[column].cast(pl.Float64).to_numpy()
            if np.isnan(scores).any():
                raise ValueError(f"{name} gives a positive event a NaN score.")
            thresholds = np.unique(scores)[::-1].copy()
            self._models[name] = _ModelState(
                scores=scores,
                thresholds=thresholds,
                above=np.zeros(thresholds.size, dtype=np.int64),
                level=np.zeros(thresholds.size, dtype=np.int64),
                ranks={key: np.zeros(scores.size, dtype=np.int64) for key in RANK_COLUMNS},
                contested={k: [0, 0, 0] for k in self._budgets},
                top=empty_top,
            )

    @property
    def model_names(self) -> list[str]:
        return list(self._models)

    def drop(self, name: str) -> None:
        """Stop accumulating a model, e.g. one that failed to score a day."""
        self._models.pop(name, None)

    def add_day(self, frame: pl.DataFrame, scores: dict[str, np.ndarray]) -> None:
        """One day's `event_id`, `time`, `day` and `is_malicious`, and each model's scores in row order."""
        days = frame["day"].unique().to_list()
        if len(days) != 1:
            raise ValueError(f"add_day takes exactly one day, got {sorted(days)}.")
        day_number = int(days[0])
        if day_number in self._days:
            raise ValueError(f"Day {day_number} was already added.")
        malicious = frame["is_malicious"].to_numpy().astype(bool)
        time = frame["time"].to_numpy()
        event_id = frame["event_id"].to_numpy()
        positive_rows = np.flatnonzero(malicious)
        expected = self._expected_per_day.get(day_number, 0)
        if positive_rows.size != expected:
            raise ValueError(
                f"Day {day_number} holds {positive_rows.size} malicious events; "
                f"the positives table lists {expected}."
            )
        try:
            positive_index = np.array(
                [self._row[int(e)] for e in event_id[positive_rows]], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError(f"Malicious event {exc} is not in the positives table.") from None
        day = _Day(
            time=time,
            event_id=event_id,
            malicious=malicious,
            arrival=np.lexsort((event_id, time)),
            positive_rows=positive_rows,
            positive_index=positive_index,
        )
        for name, state in self._models.items():
            s = np.asarray(scores[name], dtype=np.float64)
            if s.shape != (frame.height,):
                raise ValueError(f"{name}: {s.size} scores for a day of {frame.height} events.")
            if np.isnan(s).any():
                raise ValueError(f"{name} gives an event of day {day_number} a NaN score.")
            if not np.array_equal(s[positive_rows], state.scores[positive_index]):
                raise ValueError(
                    f"{name} scores the positives of day {day_number} differently among the "
                    "day's events than on their own: it is not row-local, and its ranks "
                    "would depend on how the split was batched."
                )
            self._update(state, s, day)
        self._days[day_number] = (frame.height, int(positive_rows.size))
        self._n_benign += frame.height - int(positive_rows.size)

    def _update(self, state: _ModelState, s: np.ndarray, day: _Day) -> None:
        n = s.size
        # Queue order: score descending, then arrival (time, event_id), as in `evaluate.budget`.
        order = day.arrival[np.argsort(-s[day.arrival], kind="stable")]
        descending = s[order]
        negated = -descending
        benign_negated = negated[~day.malicious[order]]
        queue_rank = np.empty(n, dtype=np.int64)
        queue_rank[order] = np.arange(1, n + 1)

        rows, index = day.positive_rows, day.positive_index
        ps = s[rows]
        above = np.searchsorted(negated, -ps, side="left")
        benign_level = np.searchsorted(benign_negated, -ps, side="right") - np.searchsorted(
            benign_negated, -ps, side="left"
        )
        arrival_keys = (day.time[rows], day.event_id[rows])
        state.ranks["rank"][index] = queue_rank[rows]
        state.ranks["rank_best"][index] = above + 1
        state.ranks["rank_worst"][index] = np.searchsorted(negated, -ps, side="right")
        # Benign events first inside a tie: the fewest positives any tie-break admits.
        state.ranks["rank_against"][index] = (
            above + benign_level + _position_among_ties(ps, *arrival_keys) + 1
        )
        # Positives first, a campaign's earliest event before any campaign's next.
        state.ranks["rank_for"][index] = (
            above + _position_among_ties(ps, self._campaign_event_index[index], *arrival_keys) + 1
        )

        left = np.searchsorted(benign_negated, -state.thresholds, side="left")
        state.above += left
        state.level += np.searchsorted(benign_negated, -state.thresholds, side="right") - left

        for k in self._budgets:
            if k <= 0 or n <= k:
                continue
            threshold = descending[k - 1]
            n_above = int(np.searchsorted(negated, -threshold, side="left"))
            n_tied = int(np.searchsorted(negated, -threshold, side="right")) - n_above
            if n_tied > k - n_above:
                state.contested[k][0] += k - n_above
                state.contested[k][1] += n_tied
                state.contested[k][2] += 1

        head = order[: self._top_k]
        merged = (
            np.concatenate([state.top[0], descending[: self._top_k]]),
            np.concatenate([state.top[1], day.time[head]]),
            np.concatenate([state.top[2], day.event_id[head]]),
            np.concatenate([state.top[3], day.malicious[head]]),
        )
        keep = np.lexsort((merged[2], merged[1], -merged[0]))[: self._top_k]
        state.top = (merged[0][keep], merged[1][keep], merged[2][keep], merged[3][keep])

        if n:
            changes = np.flatnonzero(descending[1:] != descending[:-1])
            runs = np.diff(np.concatenate([[0], changes + 1, [n]]))
            distinct = int(changes.size + 1)
            state.min_distinct = (
                distinct if state.min_distinct is None else min(state.min_distinct, distinct)
            )
            state.largest_tie = max(state.largest_tie, float(runs.max() / n))

    def finish(self) -> dict[str, ModelStatistics]:
        missing = sorted(set(self._expected_per_day) - set(self._days))
        if missing:
            raise ValueError(f"Positives on days {missing}, which were never added.")
        day_sizes = pl.DataFrame(
            [(day, events, positives) for day, (events, positives) in sorted(self._days.items())],
            schema={"day": pl.Int64, "events": pl.Int64, "positives": pl.Int64},
            orient="row",
        )
        keys = self._positives.select(POSITIVE_KEYS)
        return {
            name: ModelStatistics(
                model_name=name,
                positives=keys.with_columns(
                    pl.Series("score", state.scores),
                    *[pl.Series(key, state.ranks[key]) for key in RANK_COLUMNS],
                ),
                thresholds=state.thresholds,
                benign_above=state.above,
                benign_level=state.level,
                n_benign=self._n_benign,
                day_sizes=day_sizes,
                tie_exposure=[
                    TieExposure(
                        budget=k,
                        n_contested_slots=state.contested[k][0],
                        n_contenders=state.contested[k][1],
                        n_contested_days=state.contested[k][2],
                    )
                    for k in self._budgets
                ],
                top=pl.DataFrame(
                    {
                        "score": state.top[0],
                        "time": state.top[1],
                        "event_id": state.top[2],
                        "is_malicious": state.top[3],
                    }
                ),
                min_distinct_scores_in_a_day=state.min_distinct or 0,
                largest_tie_fraction_in_a_day=state.largest_tie,
            )
            for name, state in self._models.items()
        }


def statistics_from_frame(
    scored: pl.DataFrame, score_columns: dict[str, str], *, budgets: list[int], top_k: int
) -> dict[str, ModelStatistics]:
    """The same statistics from a frame already holding every model's scores (the demo, tests)."""
    positives = scored.filter(pl.col("is_malicious")).select(
        *POSITIVE_KEYS, *score_columns.values()
    )
    accumulator = SplitAccumulator(positives, score_columns, budgets=budgets, top_k=top_k)
    for day in scored.partition_by("day", maintain_order=True):
        accumulator.add_day(day, {name: day[col].to_numpy() for name, col in score_columns.items()})
    return accumulator.finish()


def positive_weight_by_group(stats: ModelStatistics, weights: np.ndarray) -> np.ndarray:
    """Positive weight at each threshold; `weights` is (positives,) or (resamples, positives)."""
    w = np.atleast_2d(weights).astype(np.float64)
    out = np.zeros((w.shape[0], stats.thresholds.size))
    np.add.at(out.T, stats.threshold_group(), w.T)
    return out


def weighted_average_precision(tp_by_group: np.ndarray, fp_at_or_above: np.ndarray) -> np.ndarray:
    """Average precision per row, from the positive weight at each threshold and the benign weight at or above it.

    The sum over thresholds of recall gained times precision, as scikit-learn computes it; a
    threshold with no benign event between it and the next changes nothing, so the positives'
    distinct scores are the only thresholds that count. NaN where no positive has weight.
    """
    tp = np.cumsum(tp_by_group, axis=1)
    total = tp[:, -1:] if tp.shape[1] else np.zeros((tp.shape[0], 1))
    denominator = tp + fp_at_or_above
    precision = np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        result: np.ndarray = np.sum(tp_by_group / total * precision, axis=1)
    return np.where(total[:, 0] > 0, result, np.nan)


def average_precision(stats: ModelStatistics, *, campaign_weighted: bool = False) -> float:
    """AUC-PR; `campaign_weighted` gives each positive weight 1/|its campaign| (benign events keep 1)."""
    if not stats.positives.height:
        return 0.0
    weights = np.ones(stats.positives.height)
    if campaign_weighted:
        campaign = stats.positives["campaign_id"].to_numpy()
        _, inverse, counts = np.unique(campaign, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse]
    tp = positive_weight_by_group(stats, weights)
    fp = (stats.benign_above + stats.benign_level).astype(np.float64)[None, :]
    return float(weighted_average_precision(tp, fp)[0])


def roc_auc(stats: ModelStatistics) -> float | None:
    """Probability a positive outscores a benign event, ties counting one half; None without both."""
    n_positives, n_benign = stats.positives.height, stats.n_benign
    if not n_positives or not n_benign:
        return None
    group = stats.threshold_group()
    above, level = stats.benign_above[group], stats.benign_level[group]
    below = n_benign - above - level
    return float(np.sum(below + 0.5 * level) / (n_positives * n_benign))


def tied_pair_fraction(stats: ModelStatistics) -> float | None:
    """Share of (positive, benign) pairs on the same score: the pairs ROC-AUC credits one half."""
    n_positives, n_benign = stats.positives.height, stats.n_benign
    if not n_positives or not n_benign:
        return None
    return float(np.sum(stats.benign_level[stats.threshold_group()]) / (n_positives * n_benign))


def recall_at_fixed_fpr(stats: ModelStatistics, target_fpr: float) -> float:
    """Largest recall at a threshold whose false-positive rate does not exceed `target_fpr`."""
    n_positives, n_benign = stats.positives.height, stats.n_benign
    if not n_positives or not n_benign:
        return 0.0
    recall = np.cumsum(np.bincount(stats.threshold_group(), minlength=stats.thresholds.size))
    fpr = (stats.benign_above + stats.benign_level) / n_benign
    eligible = fpr <= target_fpr
    return float(recall[eligible].max() / n_positives) if eligible.any() else 0.0


def precision_at_k(stats: ModelStatistics, k: int) -> float:
    """Share of positives among the split's first `k` events in queue order."""
    if k <= 0:
        return 0.0
    if k > stats.top.height and stats.n_events > stats.top.height:
        raise ValueError(f"precision@{k} needs the first {k} events; only {stats.top.height} kept.")
    head = stats.top["is_malicious"].head(k)
    return float(head.mean()) if head.len() else 0.0  # type: ignore[arg-type]


def _campaign_recall(positives: pl.DataFrame, rank_column: str, k: int) -> float:
    n_campaigns = positives["campaign_id"].n_unique()
    if not n_campaigns:
        return 0.0
    return positives.filter(pl.col(rank_column) <= k)["campaign_id"].n_unique() / n_campaigns


def budget_curve(stats: ModelStatistics, budgets: list[int]) -> BudgetCurve:
    """Event and campaign recall at each daily budget, the tie bracket, exposure and first detection."""
    positives = stats.positives
    n = positives.height
    return BudgetCurve(
        model_name=stats.model_name,
        budgets=budgets,
        event_recall=[
            positives.filter(pl.col("rank") <= k).height / n if n else 0.0 for k in budgets
        ],
        campaign_recall=[_campaign_recall(positives, "rank", k) for k in budgets],
        campaign_recall_min=[_campaign_recall(positives, "rank_against", k) for k in budgets],
        campaign_recall_max=[_campaign_recall(positives, "rank_for", k) for k in budgets],
        tie_exposure=[e for e in stats.tie_exposure if e.budget in budgets],
        budget_for_first_detection=int(positives["rank"].min()) if n else None,  # type: ignore[arg-type]
    )


def time_to_detection(stats: ModelStatistics, k: int) -> TimeToDetectionResult:
    """Delay from each campaign's first event to its first alert at daily budget `k`."""
    per_campaign = stats.positives.group_by("campaign_id").agg(
        pl.col("time").min().alias("first_event_time"),
        pl.col("time").filter(pl.col("rank") <= k).min().alias("first_alert_time"),
    )
    detected = per_campaign.filter(pl.col("first_alert_time").is_not_null()).sort("campaign_id")
    return TimeToDetectionResult(
        model_name=stats.model_name,
        budget=k,
        detected_delay_seconds=(
            detected["first_alert_time"] - detected["first_event_time"]
        ).to_list(),
        n_never_detected=per_campaign.height - detected.height,
        n_total_campaigns=per_campaign.height,
    )
