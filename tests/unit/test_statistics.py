"""Every metric derived from day-by-day statistics equals its whole-frame definition."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from authbench.evaluate import statistics as st
from authbench.evaluate.budget import compute_budget_curve
from authbench.evaluate.campaign import time_to_detection
from authbench.evaluate.ledger import day_sizes, positives_ledger
from authbench.evaluate.metrics import campaign_weighted_auc_pr, precision_at_k

BUDGETS = [1, 3, 10, 40]


def _frame(seed: int, levels: int) -> pl.DataFrame:
    """Four days, scores on `levels` values (heavy ties), campaigns spanning days, shared seconds."""
    rng = np.random.default_rng(seed)
    n = 600
    day = np.sort(rng.integers(0, 4, n))
    time = day * 86_400 + rng.integers(0, 40, n) * 60
    malicious = rng.random(n) < 0.06
    campaign = np.where(malicious, rng.integers(1, 6, n), -1)
    scores = {
        "coarse": rng.integers(0, levels, n).astype(float) + malicious * rng.integers(0, 2, n),
        "fine": rng.normal(size=n) + 1.5 * malicious,
        "constant": np.zeros(n),
    }
    frame = pl.DataFrame(
        {
            "event_id": rng.permutation(n).astype(np.int64),
            "time": time.astype(np.int64),
            "day": day.astype(np.int64),
            "is_malicious": malicious,
            "campaign_id": pl.Series(campaign).replace(-1, None),
            **{f"s_{name}": values for name, values in scores.items()},
        }
    )
    return frame.sample(fraction=1.0, shuffle=True, seed=seed)


COLUMNS = {"coarse": "s_coarse", "fine": "s_fine", "constant": "s_constant"}


@pytest.fixture(params=[(0, 3), (1, 2), (2, 8)], ids=["seed0", "seed1", "seed2"])
def case(request: pytest.FixtureRequest) -> tuple[pl.DataFrame, dict[str, st.ModelStatistics]]:
    frame = _frame(*request.param)
    return frame, st.statistics_from_frame(frame, COLUMNS, budgets=BUDGETS, top_k=max(BUDGETS))


@pytest.mark.parametrize("model", list(COLUMNS))
def test_budget_metrics_equal_the_whole_frame_definitions(
    case: tuple[pl.DataFrame, dict[str, st.ModelStatistics]], model: str
) -> None:
    frame, stats = case
    expected = compute_budget_curve(frame, COLUMNS[model], model, budgets=BUDGETS)
    assert st.budget_curve(stats[model], BUDGETS) == expected
    for k in BUDGETS:
        reference = time_to_detection(frame, COLUMNS[model], k, model)
        actual = st.time_to_detection(stats[model], k)
        assert sorted(actual.detected_delay_seconds) == sorted(reference.detected_delay_seconds)
        assert actual.n_never_detected == reference.n_never_detected
        assert actual.n_total_campaigns == reference.n_total_campaigns


@pytest.mark.parametrize("model", list(COLUMNS))
def test_the_ledger_columns_equal_the_whole_frame_ledger(
    case: tuple[pl.DataFrame, dict[str, st.ModelStatistics]], model: str
) -> None:
    frame, stats = case
    reference = positives_ledger(frame, COLUMNS[model], model)
    actual = stats[model].positives
    for column in ("event_id", "day", "time", "rank", "rank_best", "rank_worst"):
        assert actual[column].to_list() == reference[column].to_list(), column
    assert stats[model].day_sizes.rows() == day_sizes(frame).rows()


@pytest.mark.parametrize("model", list(COLUMNS))
def test_ranking_metrics_equal_scikit_learn(
    case: tuple[pl.DataFrame, dict[str, st.ModelStatistics]], model: str
) -> None:
    frame, stats = case
    y = frame["is_malicious"].to_numpy()
    s = frame[COLUMNS[model]].to_numpy()
    assert st.average_precision(stats[model]) == pytest.approx(
        average_precision_score(y, s), rel=1e-12
    )
    assert st.average_precision(stats[model], campaign_weighted=True) == pytest.approx(
        campaign_weighted_auc_pr(y, s, frame["campaign_id"].to_numpy()), rel=1e-12
    )
    assert st.roc_auc(stats[model]) == pytest.approx(roc_auc_score(y, s), rel=1e-12)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    for target in (0.0, 0.01, 0.1, 0.5):
        assert st.recall_at_fixed_fpr(stats[model], target) == pytest.approx(
            float(tpr[fpr <= target].max()), abs=1e-15
        )


@pytest.mark.parametrize("model", list(COLUMNS))
def test_precision_at_k_takes_ties_in_arrival_order(
    case: tuple[pl.DataFrame, dict[str, st.ModelStatistics]], model: str
) -> None:
    frame, stats = case
    in_arrival_order = frame.sort("time", "event_id")
    y = in_arrival_order["is_malicious"].to_numpy()
    s = in_arrival_order[COLUMNS[model]].to_numpy()
    for k in BUDGETS:
        assert st.precision_at_k(stats[model], k) == precision_at_k(y, s, k)


def test_day_order_does_not_matter(
    case: tuple[pl.DataFrame, dict[str, st.ModelStatistics]],
) -> None:
    frame, stats = case
    positives = frame.filter(pl.col("is_malicious")).select(*st.POSITIVE_KEYS, *COLUMNS.values())
    accumulator = st.SplitAccumulator(positives, COLUMNS, budgets=BUDGETS, top_k=max(BUDGETS))
    for day in reversed(frame.partition_by("day", maintain_order=True)):
        accumulator.add_day(day, {name: day[col].to_numpy() for name, col in COLUMNS.items()})
    for name, reordered in accumulator.finish().items():
        assert reordered.positives.equals(stats[name].positives)
        assert reordered.top.equals(stats[name].top)
        assert np.array_equal(reordered.benign_above, stats[name].benign_above)
        assert reordered.tie_exposure == stats[name].tie_exposure


def test_a_model_that_is_not_row_local_is_caught() -> None:
    frame = _frame(0, 3)
    positives = frame.filter(pl.col("is_malicious")).select(*st.POSITIVE_KEYS, "s_fine")
    accumulator = st.SplitAccumulator(positives, {"fine": "s_fine"}, budgets=BUDGETS, top_k=40)
    day = frame.filter(pl.col("day") == frame.filter(pl.col("is_malicious"))["day"][0])
    with pytest.raises(ValueError, match="not row-local"):
        accumulator.add_day(day, {"fine": day["s_fine"].to_numpy() + 1e-15})


def test_a_missing_day_or_a_stray_positive_is_an_error() -> None:
    frame = _frame(0, 3)
    positives = frame.filter(pl.col("is_malicious")).select(*st.POSITIVE_KEYS, "s_fine")
    accumulator = st.SplitAccumulator(positives, {"fine": "s_fine"}, budgets=BUDGETS, top_k=40)
    with pytest.raises(ValueError, match="never added"):
        accumulator.finish()
    stray = frame.filter(pl.col("day") == 0).with_columns(pl.lit(True).alias("is_malicious"))
    with pytest.raises(ValueError, match="malicious events"):
        accumulator.add_day(stray, {"fine": stray["s_fine"].to_numpy()})


def test_a_positive_without_a_campaign_is_refused() -> None:
    frame = _frame(0, 3).with_columns(
        pl.when(pl.col("is_malicious"))
        .then(None)
        .otherwise(pl.col("campaign_id"))
        .alias("campaign_id")
    )
    with pytest.raises(ValueError, match="no campaign_id"):
        st.statistics_from_frame(frame, COLUMNS, budgets=BUDGETS, top_k=40)


def test_a_fixed_threshold_counts_alerts_and_recall_as_a_brute_force_count_does() -> None:
    calibration, frame = _frame(4, 1000), _frame(5, 1000)
    columns = {"fine": "s_fine"}
    budgets = [1, 3, 10]
    calibrated = st.statistics_from_frame(calibration, columns, budgets=[1], top_k=40)["fine"]
    thresholds = st.alert_thresholds(calibrated, budgets)
    # On its own calibration split a continuous score gives exactly k alerts a day on average.
    own = st.statistics_from_frame(
        calibration, columns, budgets=budgets, top_k=40, alert_thresholds={"fine": thresholds}
    )["fine"]
    assert own.alerts_per_day.sum(axis=0).tolist() == [k * 4 for k in budgets]

    stats = st.statistics_from_frame(
        frame, columns, budgets=budgets, top_k=40, alert_thresholds={"fine": thresholds}
    )["fine"]
    rows = st.fixed_threshold(stats, budgets)
    for i, (k, row) in enumerate(zip(budgets, rows, strict=True)):
        above = frame.filter(pl.col("s_fine") >= thresholds[i])
        per_day = [above.filter(pl.col("day") == d).height for d in sorted(frame["day"].unique())]
        assert stats.alerts_per_day[:, i].tolist() == per_day
        assert row["alerts_per_day_mean"] == pytest.approx(sum(per_day) / len(per_day))
        hit = above.filter(pl.col("is_malicious"))
        assert (
            row["campaign_recall"]
            == hit["campaign_id"].n_unique()
            / frame.filter(pl.col("is_malicious"))["campaign_id"].n_unique()
        )
        assert row["budget"] == k


def test_without_thresholds_no_fixed_threshold_is_reported() -> None:
    stats = st.statistics_from_frame(_frame(0, 3), COLUMNS, budgets=BUDGETS, top_k=40)
    assert stats["fine"].alert_thresholds.size == 0
    assert st.fixed_threshold(stats["fine"], []) == []


def _with_entities(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frame = _frame(seed, 3)
    n = frame.height
    return frame.with_columns(
        pl.Series("src_computer", [f"C{i}" for i in rng.integers(0, 6, n)]),
        pl.Series("dst_computer", [f"D{i}" for i in rng.integers(0, 6, n)]),
        pl.Series("src_user", [f"U{i}" for i in rng.integers(0, 9, n)]),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("column", ["s_coarse", "s_fine"])
def test_unit_ranks_equal_a_brute_force_ranking_of_units(seed: int, column: str) -> None:
    frame = _with_entities(seed)
    stats = st.statistics_from_frame(frame, {"m": column}, budgets=[1], top_k=1, with_units=True)[
        "m"
    ]
    for unit, keys in st.UNITS.items():
        keyed = frame.with_columns(
            pl.concat_str(
                [
                    (pl.col("time") // 3600).cast(pl.Utf8) if k == "hour" else pl.col(k)
                    for k in keys
                ],
                separator="|",
            ).alias("_unit")
        )
        units = (
            keyed.group_by("day", "_unit")
            .agg(pl.col(column).max().alias("score"), pl.col("time").min().alias("first"))
            .sort("day", "score", "first", "_unit", descending=[False, True, False, False])
            .with_columns(pl.int_range(1, pl.len() + 1).over("day").alias("rank"))
        )
        expected = (
            keyed.filter(pl.col("is_malicious"))
            .join(units, on=["day", "_unit"])
            .sort("day", "time", "event_id")["rank"]
            .to_list()
        )
        assert stats.positives[f"rank_{unit}"].to_list() == expected, unit
        counts = units.group_by("day").len().sort("day")["len"].to_list()
        assert stats.day_sizes[f"{unit}s"].to_list() == counts
