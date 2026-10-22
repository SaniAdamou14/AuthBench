"""Positives ledger: the day rank of every red-team event under every model, and what follows from it.

Every budget-shaped quantity (recall at any budget, per-campaign detection
budgets, their quantiles and log-area, regrouped campaigns, resampling, the
random-ranking null) is a function of these ranks and the day sizes alone. The
ledger is written once per run, so these analyses never re-score a model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from authbench.evaluate.budget import _RANK, _ranked

LEDGER_COLUMNS = [
    "model",
    "event_id",
    "campaign_id",
    "day",
    "time",
    "rank",
    "rank_best",
    "rank_worst",
]
QUANTILES = (0.1, 0.25, 0.5)


def positives_ledger(scored: pl.DataFrame, score_col: str, model_name: str) -> pl.DataFrame:
    """Queue rank of each positive in its day, with the best and worst rank its tie group allows."""
    ranked = _ranked(scored, score_col, "queue").with_columns(
        pl.col(score_col).rank("min", descending=True).over("day").alias("rank_best"),
        pl.col(score_col).rank("max", descending=True).over("day").alias("rank_worst"),
    )
    return (
        ranked.filter(pl.col("is_malicious"))
        .select(
            pl.lit(model_name).alias("model"),
            "event_id",
            "campaign_id",
            "day",
            "time",
            pl.col(_RANK).cast(pl.Int64).alias("rank"),
            pl.col("rank_best").cast(pl.Int64),
            pl.col("rank_worst").cast(pl.Int64),
        )
        .sort("day", "time", "event_id")
    )


def day_sizes(scored: pl.DataFrame) -> pl.DataFrame:
    """Events and positives per day: the denominator of every rank."""
    return (
        scored.group_by("day")
        .agg(pl.len().alias("events"), pl.col("is_malicious").sum().alias("positives"))
        .sort("day")
    )


def detection_budgets(ledger: pl.DataFrame, rank_col: str = "rank") -> np.ndarray:
    """b(c) = min rank over a campaign's events, one value per campaign, for one model."""
    per_campaign = ledger.group_by("campaign_id").agg(pl.col(rank_col).min()).sort("campaign_id")
    return per_campaign[rank_col].to_numpy().astype(np.int64)


def recall_at(budgets: np.ndarray, k: int) -> float:
    return float(np.mean(budgets <= k)) if budgets.size else 0.0


def budget_quantile(budgets: np.ndarray, q: float) -> int:
    """Smallest daily budget detecting at least a fraction q of the campaigns."""
    ordered = np.sort(budgets)
    return int(ordered[max(0, math.ceil(q * ordered.size) - 1)])


def log_area(budgets: np.ndarray, k_max: int) -> float:
    """Area under campaign recall against log budget on [1, k_max], normalised to [0, 1].

    R(k) is a step function of the b(c), so the integral of R(k) dk / k is exactly
    the mean over campaigns of ln(k_max / b(c)) for b(c) <= k_max.
    """
    if not budgets.size or k_max <= 1:
        return 0.0
    within = budgets[budgets <= k_max].astype(np.float64)
    area = np.sum(np.log(k_max / within)) / (budgets.size * math.log(k_max))
    # numpy's vectorised log differs in the last bit between CPUs; 12 decimals are machine-independent.
    return round(float(area), 12)


@dataclass(frozen=True)
class BudgetSummary:
    first_detection: int
    quantiles: dict[float, int]
    log_area: float

    @classmethod
    def from_budgets(cls, budgets: np.ndarray, k_max: int) -> BudgetSummary:
        return cls(
            first_detection=int(budgets.min()),
            quantiles={q: budget_quantile(budgets, q) for q in QUANTILES},
            log_area=log_area(budgets, k_max),
        )

    def as_vector(self) -> np.ndarray:
        return np.array([self.first_detection, *self.quantiles.values(), self.log_area])


def resample_summaries(
    budgets: np.ndarray, groups: np.ndarray, k_max: int, n_resamples: int, seed: int
) -> np.ndarray:
    """Summaries over a cluster bootstrap that resamples whole groups (accounts) with replacement."""
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(labels.size)]
    out = np.empty((n_resamples, 2 + len(QUANTILES)))
    for r in range(n_resamples):
        drawn = rng.integers(0, labels.size, labels.size)
        sample = budgets[np.concatenate([members[g] for g in drawn])]
        out[r] = BudgetSummary.from_budgets(sample, k_max).as_vector()
    return out


def random_ranking_summaries(
    campaign_of_positive: np.ndarray,
    day_of_positive: np.ndarray,
    events_per_day: dict[int, int],
    k_max: int,
    n_simulations: int,
    seed: int,
) -> np.ndarray:
    """Summaries of the same campaigns under uniformly random rankings of each day."""
    rng = np.random.default_rng(seed)
    campaigns, campaign_index = np.unique(campaign_of_positive, return_inverse=True)
    days = np.unique(day_of_positive)
    positions = {d: np.flatnonzero(day_of_positive == d) for d in days}
    out = np.empty((n_simulations, 2 + len(QUANTILES)))
    ranks = np.empty(day_of_positive.size, dtype=np.int64)
    for s in range(n_simulations):
        for d in days:
            where = positions[d]
            ranks[where] = rng.choice(events_per_day[int(d)], size=where.size, replace=False) + 1
        budgets = np.full(campaigns.size, np.iinfo(np.int64).max)
        np.minimum.at(budgets, campaign_index, ranks)
        out[s] = BudgetSummary.from_budgets(budgets, k_max).as_vector()
    return out


def summary_record(
    ledger: pl.DataFrame,
    sizes: pl.DataFrame,
    users: pl.DataFrame,
    *,
    k_max: int,
    n_resamples: int,
    n_simulations: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    """Point summaries, account-bootstrap intervals and random-ranking percentiles for one model."""
    by_campaign = (
        ledger.group_by("campaign_id")
        .agg(pl.col("rank").min().alias("b"), pl.col("rank_best").min(), pl.col("rank_worst").min())
        .join(users, on="campaign_id", how="left")
        .sort("campaign_id")
    )
    budgets = by_campaign["b"].to_numpy().astype(np.int64)
    point = BudgetSummary.from_budgets(budgets, k_max)
    resampled = resample_summaries(
        budgets, by_campaign["user"].to_numpy(), k_max, n_resamples, seed
    )
    null = random_ranking_summaries(
        ledger["campaign_id"].to_numpy(),
        ledger["day"].to_numpy(),
        dict(zip(sizes["day"].to_list(), sizes["events"].to_list(), strict=True)),
        k_max,
        n_simulations,
        seed,
    )
    alpha = (1 - confidence) / 2
    low, high = np.quantile(resampled, [alpha, 1 - alpha], axis=0)
    names = ["first_detection", *[f"budget_q{int(100 * q)}" for q in QUANTILES], "log_area"]
    observed = point.as_vector()
    # Share of random rankings at least as good: lower budgets, or a higher log-area, are better.
    better_or_equal = [
        float(np.mean(null[:, i] >= observed[i]))
        if name == "log_area"
        else float(np.mean(null[:, i] <= observed[i]))
        for i, name in enumerate(names)
    ]
    return {
        "n_campaigns": int(budgets.size),
        "n_accounts": int(by_campaign["user"].n_unique()),
        "k_max": k_max,
        "tie_range_first_detection": [
            int(by_campaign["rank_best"].to_numpy().min()),
            int(by_campaign["rank_worst"].to_numpy().min()),
        ],
        "statistics": {
            name: {
                "value": float(observed[i]),
                "ci_low": float(low[i]),
                "ci_high": float(high[i]),
                "random_ranking_at_least_as_good": better_or_equal[i],
            }
            for i, name in enumerate(names)
        },
    }


def write_ledger_outputs(
    scored: pl.DataFrame,
    model_names: list[str],
    score_columns: dict[str, str],
    users: pl.DataFrame,
    tables_dir: Path,
    *,
    k_max: int,
    n_resamples: int,
    n_simulations: int,
    confidence: float,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Write positives_ledger.csv, day_sizes.csv and budget_summary.json; return the summaries."""
    ledger = pl.concat(
        [positives_ledger(scored, score_columns[name], name) for name in model_names]
    )
    sizes = day_sizes(scored)
    ledger.write_csv(tables_dir / "positives_ledger.csv")
    sizes.write_csv(tables_dir / "day_sizes.csv")
    summaries = {
        name: summary_record(
            ledger.filter(pl.col("model") == name),
            sizes,
            users,
            k_max=k_max,
            n_resamples=n_resamples,
            n_simulations=n_simulations,
            confidence=confidence,
            seed=seed,
        )
        for name in model_names
    }
    (tables_dir / "budget_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def campaign_users(labeled: pl.DataFrame) -> pl.DataFrame:
    """campaign_id -> account, the resampling unit of the budget summaries."""
    return (
        labeled.filter(pl.col("is_malicious"))
        .group_by("campaign_id")
        .agg(pl.col("src_user").cast(pl.String).first().alias("user"))
        .sort("campaign_id")
    )
