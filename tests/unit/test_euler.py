"""M5c Euler: signal, causality, determinism and row-locality on a small synthetic network."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch")

from authbench.models.euler import EulerScorer  # noqa: E402

SNAPSHOT = 1800


def _routine(snapshots: range, seed: int = 0) -> pl.DataFrame:
    """A chain C0 -> C1 -> ... -> C9 and a ticket server, a few events per edge per snapshot."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in snapshots:
        for i in range(9):
            for _ in range(rng.integers(1, 4)):
                rows.append((s * SNAPSHOT + int(rng.integers(0, SNAPSHOT)), f"C{i}", f"C{i + 1}"))
        rows.append((s * SNAPSHOT + 7, "C0", "TGT"))
    frame = pl.DataFrame(rows, schema=["time", "src_computer", "dst_computer"], orient="row")
    return frame.sort("time").with_row_index("event_id")


def _fitted(seed: int = 3) -> tuple[EulerScorer, pl.DataFrame, pl.DataFrame]:
    train, test = _routine(range(0, 30)), _routine(range(30, 36), seed=1)
    anomaly = pl.DataFrame(
        {"time": [33 * SNAPSHOT + 100], "src_computer": ["C2"], "dst_computer": ["C8"]}
    ).with_columns(pl.lit(10_000, dtype=pl.UInt32).alias("event_id"))
    test = pl.concat([test, anomaly.select(test.columns)]).sort("time")
    model = EulerScorer(epochs=15, patience=5, window=10, seed=seed, learning_rate=0.02)
    model.fit(train.lazy())
    model.prepare([train.lazy(), test.lazy()])
    return model, train, test


def test_a_never_seen_edge_scores_above_the_routine_ones_of_its_snapshot() -> None:
    model, _, test = _fitted()
    scores = model.score(test.lazy()).to_numpy()
    assert np.isfinite(scores).all()
    snapshot = test["time"].to_numpy() // SNAPSHOT == 33
    anomalous = (test["src_computer"] == "C2").to_numpy() & (
        test["dst_computer"] == "C8"
    ).to_numpy()
    assert scores[anomalous][0] > np.max(scores[snapshot & ~anomalous])


def test_scores_do_not_depend_on_later_snapshots() -> None:
    model, train, test = _fitted()
    before = model.score(test.filter(pl.col("time") < 34 * SNAPSHOT).lazy()).to_numpy()
    later = pl.DataFrame(
        {"time": [35 * SNAPSHOT + 5] * 20, "src_computer": ["C5"] * 20, "dst_computer": ["C0"] * 20}
    ).with_columns(pl.lit(20_000, dtype=pl.UInt32).alias("event_id"))
    model.prepare([train.lazy(), pl.concat([test, later.select(test.columns)]).lazy()])
    after = model.score(test.filter(pl.col("time") < 34 * SNAPSHOT).lazy()).to_numpy()
    np.testing.assert_array_equal(before, after)


def test_two_fits_with_one_seed_score_identically_and_scoring_is_row_local() -> None:
    first, _, test = _fitted(seed=5)
    second, _, _ = _fitted(seed=5)
    whole = first.score(test.lazy()).to_numpy()
    np.testing.assert_array_equal(whole, second.score(test.lazy()).to_numpy())
    alone = np.array(
        [first.score(test.slice(i, 1).lazy()).to_numpy()[0] for i in range(0, test.height, 7)]
    )
    np.testing.assert_array_equal(whole[::7], alone)


def test_a_computer_unseen_in_training_is_scored_through_the_shared_unseen_node() -> None:
    model, _, _ = _fitted()
    unseen = pl.DataFrame(
        {"time": [32 * SNAPSHOT + 1], "src_computer": ["C77"], "dst_computer": ["C1"]}
    )
    assert np.isfinite(model.score(unseen.lazy()).to_numpy()).all()


def test_training_stops_early_and_keeps_its_history() -> None:
    model, _, _ = _fitted()
    assert 1 <= len(model.history) <= 15
    assert all(np.isfinite(loss) for _, _, loss in model.history)


def test_snapshot_edges_aggregated_by_day_equal_one_aggregation() -> None:
    from authbench.models.euler import _snapshot_edges

    frame = pl.concat([_routine(range(0, 40)), _routine(range(40, 100), seed=2)])
    expected = (
        frame.select(
            (pl.col("time") // SNAPSHOT).cast(pl.Int64).alias("snapshot"),
            pl.col("src_computer").alias("src"),
            pl.col("dst_computer").alias("dst"),
        )
        .group_by("snapshot", "src", "dst")
        .len()
        .sort("snapshot", "src", "dst")
    )
    assert _snapshot_edges(frame.lazy(), SNAPSHOT).equals(expected)
