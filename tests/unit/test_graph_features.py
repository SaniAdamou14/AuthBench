from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("networkx")

from authbench.features.graph import (  # noqa: E402
    GraphFeatureConfig,
    attach_graph_features,
    build_daily_graph,
)

CONFIG = GraphFeatureConfig(
    window_days=2, node2vec_dim=4, node2vec_walk_length=5, node2vec_num_walks=2
)


def _events(rows: list[tuple[int, str, str]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "event_id": list(range(len(rows))),
            "day": [r[0] for r in rows],
            "src_user": [r[1] for r in rows],
            "dst_computer": [r[2] for r in rows],
        },
        schema={
            "event_id": pl.Int64,
            "day": pl.Int64,
            "src_user": pl.Categorical,
            "dst_computer": pl.Categorical,
        },
    )


def test_edge_weight_counts_repeated_authentications() -> None:
    events = pl.DataFrame(
        {"src_user": ["U1", "U1", "U1", "U2"], "dst_computer": ["C1", "C1", "C2", "C1"]}
    )
    graph = build_daily_graph(events)
    assert graph["u:U1"]["m:C1"]["weight"] == 2
    assert graph["u:U1"]["m:C2"]["weight"] == 1
    assert graph["u:U2"]["m:C1"]["weight"] == 1
    assert graph.number_of_edges() == 3


def test_a_day_sees_only_its_trailing_window_never_itself_or_the_future() -> None:
    events = _events(
        [
            (0, "U1", "C1"),
            (1, "U1", "C2"),
            (2, "U2", "C3"),
            (3, "U1", "C1"),
            (5, "U9", "C9"),
        ]
    )
    result = attach_graph_features(events, events, CONFIG).collect().sort("event_id")
    degree = dict(
        zip(result["event_id"].to_list(), result["graph_user_degree"].to_list(), strict=True)
    )

    assert degree[0] == 0, "day 0 has no past"
    assert degree[1] == 1, "day 1 sees day 0 only, not its own C2 edge"
    assert degree[2] == 0, "U2 first appears on day 2"
    assert degree[3] == 1, "day 3 reads days 1-2: U1 reached C2 only, day 0 is outside the window"
    assert degree[4] == 0, "day 5 reads days 3-4 and nothing of U9"


def test_clustering_is_the_bipartite_coefficient_and_not_identically_zero() -> None:
    events = _events(
        [(0, "U1", "C1"), (0, "U1", "C2"), (0, "U2", "C1"), (0, "U2", "C2"), (1, "U1", "C1")]
    )
    result = attach_graph_features(events, events, CONFIG).collect().filter(pl.col("day") == 1)
    assert result["graph_user_clustering"][0] == pytest.approx(1.0)


def test_m5a_scores_every_event_and_is_row_local() -> None:
    pytest.importorskip("pyod")
    from authbench.models.graph import Node2VecIsolationForestScorer

    events = _events([(d, f"U{i % 3}", f"C{(i + d) % 4}") for d in range(4) for i in range(6)])
    featured = attach_graph_features(events, events, CONFIG).collect()
    model = Node2VecIsolationForestScorer()
    model.fit(featured.lazy())
    whole = model.score(featured.lazy()).to_numpy()
    halves = [model.score(featured.slice(0, 12).lazy()), model.score(featured.slice(12).lazy())]
    assert model.scores_row_locally
    assert whole.shape == (featured.height,)
    assert whole.tolist() == pl.concat(halves).to_list()
