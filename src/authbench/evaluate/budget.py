"""Recall under a realistic per-day alert budget (US-126) — the project's central metric.

Alerts are the top-k scored events **per day**, never the top-k over the
whole test period — a SOC analyst has a daily capacity, not a period-long
one, and taking a global top-k would let a model "spend" its whole budget on
one easy day and ignore the rest (spec section 6.3 / US-126 acceptance
criteria).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

DEFAULT_BUDGETS: list[int] = [10, 50, 100, 500]


def alerts_at_budget(frame: pl.DataFrame, score_col: str, k: int) -> pl.DataFrame:
    """The top-`k` scored events of each day — the alert set a budget-`k` SOC
    analyst would actually see.
    """
    ranked = frame.with_columns(
        pl.col(score_col).rank(method="ordinal", descending=True).over("day").alias("_rank_in_day")
    )
    return ranked.filter(pl.col("_rank_in_day") <= k)


def event_recall_at_budget(frame: pl.DataFrame, score_col: str, k: int) -> float:
    """Fraction of malicious *events* (not campaigns) captured by the daily top-k alerts."""
    total_positives = frame.filter(pl.col("is_malicious")).height
    if total_positives == 0:
        return 0.0
    alerts = alerts_at_budget(frame, score_col, k)
    n_true_positives = alerts.filter(pl.col("is_malicious")).height
    return n_true_positives / total_positives


def campaign_recall_at_budget(frame: pl.DataFrame, score_col: str, k: int) -> float:
    """Fraction of campaigns with at least one event captured by the daily
    top-k alerts — the metric that reflects a defender's actual objective
    (US-106): detecting the attack, not every one of its events.
    """
    total_campaigns = frame.filter(pl.col("campaign_id").is_not_null())["campaign_id"].n_unique()
    if total_campaigns == 0:
        return 0.0
    alerts = alerts_at_budget(frame, score_col, k)
    detected_campaigns = alerts.filter(pl.col("campaign_id").is_not_null())[
        "campaign_id"
    ].n_unique()
    return detected_campaigns / total_campaigns


@dataclass
class BudgetCurve:
    model_name: str
    budgets: list[int]
    event_recall: list[float]
    campaign_recall: list[float]


def compute_budget_curve(
    frame: pl.DataFrame, score_col: str, model_name: str, budgets: list[int] | None = None
) -> BudgetCurve:
    """Event- and campaign-recall at every budget in `budgets` — this is what
    the report's principal figure (campaign recall vs. budget, one curve per
    model) is built from (US-126).
    """
    budgets = budgets or DEFAULT_BUDGETS
    return BudgetCurve(
        model_name=model_name,
        budgets=budgets,
        event_recall=[event_recall_at_budget(frame, score_col, k) for k in budgets],
        campaign_recall=[campaign_recall_at_budget(frame, score_col, k) for k in budgets],
    )
