"""The ledger must reproduce every budget metric computed from the full scored frame, exactly."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.integrate import trapezoid

from authbench.evaluate.budget import (
    campaign_detection_budgets,
    campaign_recall_at_budget,
    random_first_detection_survival,
)
from authbench.evaluate.ledger import (
    budget_quantile,
    day_sizes,
    detection_budgets,
    log_area,
    positives_ledger,
    random_ranking_summaries,
    recall_at,
    resample_summaries,
    summary_record,
)


def _scored(seed: int = 0, n_days: int = 3, per_day: int = 400) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    malicious = rng.random(n) < 0.03
    campaign = np.where(malicious, rng.integers(1, 9, n), 0)
    return pl.DataFrame(
        {
            "event_id": np.arange(n),
            "time": np.arange(n) * 60,
            "day": np.repeat(np.arange(n_days), per_day),
            "is_malicious": malicious,
            "campaign_id": [int(c) if c else None for c in campaign],
            "score": rng.integers(0, 50, n).astype(float),
        }
    )


def test_ledger_budgets_match_the_full_frame_computation() -> None:
    frame = _scored()
    ledger = positives_ledger(frame, "score", "M")
    expected = campaign_detection_budgets(frame, "score").sort("campaign_id")
    assert detection_budgets(ledger).tolist() == expected["budget_for_detection"].to_list()


@pytest.mark.parametrize("k", [1, 5, 20, 100, 400])
def test_ledger_recall_matches_campaign_recall_at_every_budget(k: int) -> None:
    frame = _scored(seed=1)
    budgets = detection_budgets(positives_ledger(frame, "score", "M"))
    assert recall_at(budgets, k) == campaign_recall_at_budget(frame, "score", k)


def test_rank_bounds_bracket_the_queue_rank_within_ties() -> None:
    ledger = positives_ledger(_scored(seed=2), "score", "M")
    assert (ledger["rank_best"] <= ledger["rank"]).all()
    assert (ledger["rank"] <= ledger["rank_worst"]).all()


def test_log_area_closed_form_equals_numerical_integration() -> None:
    budgets = np.array([1, 3, 40, 900, 20_000])
    k_max = 500
    grid = np.exp(np.linspace(0, math.log(k_max), 200_001))
    recall = np.array([np.mean(budgets <= k) for k in grid])
    numeric = trapezoid(recall, np.log(grid)) / math.log(k_max)
    assert log_area(budgets, k_max) == pytest.approx(numeric, abs=1e-4)
    assert log_area(np.array([1, 1]), k_max) == pytest.approx(1.0)
    assert log_area(np.array([600, 900]), k_max) == 0.0


def test_budget_quantile_is_the_smallest_budget_reaching_the_fraction() -> None:
    budgets = np.array([7, 3, 50, 9])
    assert budget_quantile(budgets, 0.25) == 3
    assert budget_quantile(budgets, 0.5) == 7
    assert budget_quantile(budgets, 0.75) == 9
    assert recall_at(budgets, budget_quantile(budgets, 0.5)) >= 0.5


def test_account_bootstrap_keeps_campaigns_of_one_account_together() -> None:
    budgets = np.array([1, 2, 100, 200, 300])
    users = np.array(["a", "a", "b", "c", "d"])
    out = resample_summaries(budgets, users, k_max=500, n_resamples=300, seed=0)
    assert set(np.unique(out[:, 0])) <= {1.0, 100.0, 200.0, 300.0}


def test_simulated_null_first_detection_matches_the_exact_distribution() -> None:
    days = {0: 2_000, 1: 3_000}
    day_of_positive = np.array([0] * 5 + [1] * 3)
    campaign_of_positive = np.array([1, 1, 2, 3, 3, 4, 5, 5])
    sims = random_ranking_summaries(campaign_of_positive, day_of_positive, days, 500, 20_000, 7)
    exact_days = [(2_000, 5), (3_000, 3)]
    for x in (50, 200, 600):
        empirical = float(np.mean(sims[:, 0] <= x))
        exact = 1 - random_first_detection_survival(exact_days, x)
        assert empirical == pytest.approx(exact, abs=0.015)


def test_summary_record_reports_intervals_and_null_shares() -> None:
    frame = _scored(seed=3)
    ledger = positives_ledger(frame, "score", "M")
    users = ledger.select(
        "campaign_id", (pl.lit("u") + pl.col("campaign_id").cast(pl.String)).alias("user")
    ).unique()
    record = summary_record(
        ledger,
        day_sizes(frame),
        users,
        k_max=100,
        n_resamples=200,
        n_simulations=200,
        confidence=0.95,
        seed=0,
    )
    first = record["statistics"]["first_detection"]  # type: ignore[index]
    assert first["ci_low"] <= first["value"] <= first["ci_high"]
    assert 0.0 <= first["random_ranking_at_least_as_good"] <= 1.0
    best, worst = record["tie_range_first_detection"]  # type: ignore[misc]
    assert best <= first["value"] <= worst


def test_regrouping_at_the_labelling_gap_recovers_the_campaigns_and_finer_gaps_split_them() -> None:
    from authbench.evaluate.ledger import gap_sensitivity, regroup_campaigns

    hour = 3600
    ledger = pl.DataFrame(
        {
            "model": ["m"] * 6,
            "event_id": [1, 2, 3, 4, 5, 6],
            # Account A: events 0 h, 3 h, then 40 h later; account B: one campaign at 2 h spacing.
            "campaign_id": [1, 1, 2, 3, 3, 3],
            "time": [0, 3 * hour, 43 * hour, 10 * hour, 12 * hour, 14 * hour],
            "rank": [50, 7, 900, 3, 400, 80],
        }
    )
    users = pl.DataFrame({"campaign_id": [1, 2, 3], "user": ["A", "A", "B"]})

    def partition(gap: float) -> set[frozenset[int]]:
        mapping = regroup_campaigns(ledger, users, gap)
        groups = mapping.group_by("campaign_id").agg(pl.col("event_id"))
        return {frozenset(g) for g in groups["event_id"].to_list()}

    assert partition(24) == {frozenset({1, 2}), frozenset({3}), frozenset({4, 5, 6})}
    assert partition(1) == {frozenset({i}) for i in range(1, 7)}
    assert partition(72) == {frozenset({1, 2, 3}), frozenset({4, 5, 6})}

    rows = gap_sensitivity(ledger, users, [1, 24, 72], [10, 100], k_max=1000)
    assert [r["n_campaigns"] for r in rows] == [6, 3, 2]
    at_24 = rows[1]["models"]["m"]
    assert at_24["campaign_recall"] == {"10": 2 / 3, "100": 2 / 3}
    assert at_24["first_detection"] == 3
