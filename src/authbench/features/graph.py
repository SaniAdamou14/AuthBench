"""F5 — authentication-graph features (US-113, `authbench[graph]` extra).

The bipartite user→machine graph is rebuilt **per day**, from a strictly
trailing `window_days`-day window — never once over the full 58 days. That
distinction is not cosmetic: an embedding fit on the whole period leaks every
future edge into every past node's representation, and RQ3 (does the graph
add signal independent of frequency/novelty?) becomes unanswerable (spec
section 4.1).

Day-level granularity (rebuild once per day, applied to every event of that
day) is a deliberate cost/leakage trade-off: per-event graph reconstruction
would be exact but is computationally infeasible at LANL scale, and it buys
nothing extra in causality since no event contributes to its own day's graph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional `authbench[graph]` dependency
    nx = None


@dataclass(frozen=True)
class GraphFeatureConfig:
    window_days: int = 7
    node2vec_dim: int = 64
    node2vec_walk_length: int = 40
    node2vec_num_walks: int = 10
    seed: int = 42


@dataclass
class DailyGraphFeatures:
    day: int
    degree: dict[str, int]
    pagerank: dict[str, float]
    clustering: dict[str, float]
    community: dict[str, int]
    embeddings: dict[str, list[float]]
    adamic_adar: dict[tuple[str, str], float]


def _require_networkx() -> None:
    if nx is None:
        raise ImportError(
            "F5 graph features require the optional 'graph' extra: "
            "install with `uv pip install -e '.[graph]'`."
        )


def build_daily_graph(events: pl.DataFrame) -> nx.Graph:
    """Bipartite user/machine graph over `events` (already restricted to the
    trailing window by the caller). User nodes are prefixed `u:`, machine
    nodes `m:`, so the same identifier string can never collide across sides.
    """
    _require_networkx()
    graph = nx.Graph()
    for row in events.select(["src_user", "dst_computer"]).unique().iter_rows(named=True):
        u = f"u:{row['src_user']}"
        m = f"m:{row['dst_computer']}"
        if graph.has_edge(u, m):
            graph[u][m]["weight"] += 1
        else:
            graph.add_edge(u, m, weight=1)
    return graph


def compute_daily_graph_features(
    graph: nx.Graph, day: int, config: GraphFeatureConfig
) -> DailyGraphFeatures:
    """Degree, PageRank, local clustering, Louvain community, and node2vec
    embeddings for every node in `graph`.
    """
    _require_networkx()

    degree = dict(graph.degree())
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {}
    clustering = nx.clustering(graph, weight="weight") if graph.number_of_edges() else {}

    try:
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(graph, seed=config.seed)
        community = {node: idx for idx, comm in enumerate(communities) for node in comm}
    except ImportError:  # pragma: no cover - networkx<3.0 fallback
        community = {}

    embeddings: dict[str, list[float]] = {}
    try:
        from node2vec import Node2Vec

        n2v = Node2Vec(
            graph,
            dimensions=config.node2vec_dim,
            walk_length=config.node2vec_walk_length,
            num_walks=config.node2vec_num_walks,
            seed=config.seed,
            quiet=True,
        )
        model = n2v.fit(seed=config.seed)
        embeddings = {node: model.wv[node].tolist() for node in graph.nodes()}
    except ImportError:  # pragma: no cover - optional dependency
        embeddings = {}

    adamic_adar = {
        (u, v): score
        for u, v, score in nx.adamic_adar_index(graph, [(u, v) for u, v in graph.edges()])
    }

    return DailyGraphFeatures(
        day=day,
        degree=degree,
        pagerank=pagerank,
        clustering=clustering,
        community=community,
        embeddings=embeddings,
        adamic_adar=adamic_adar,
    )


def _lookup_int_factory(table: dict[str, int], prefix: str) -> Callable[[str], int]:
    return lambda key: table.get(f"{prefix}{key}", 0)


def _lookup_float_factory(table: dict[str, float], prefix: str) -> Callable[[str], float]:
    return lambda key: table.get(f"{prefix}{key}", 0.0)


def _adamic_adar_factory(
    table: dict[tuple[str, str], float],
) -> Callable[[dict[str, str]], float]:
    def lookup(row: dict[str, str]) -> float:
        forward = (f"u:{row['src_user']}", f"m:{row['dst_computer']}")
        backward = (f"m:{row['dst_computer']}", f"u:{row['src_user']}")
        return table.get(forward, table.get(backward, 0.0))

    return lookup


def _embedding_factory(table: dict[str, list[float]], dim: int) -> Callable[[str], list[float]]:
    default = [0.0] * dim
    return lambda user: table.get(f"u:{user}", default)


def attach_graph_features(
    events: pl.LazyFrame, all_events: pl.LazyFrame, config: GraphFeatureConfig
) -> pl.LazyFrame:
    """For every day D present in `events`, build the trailing-window graph
    from `all_events` restricted to `[D - window_days, D - 1]`, and join the
    resulting node-level features onto `src_user` and `dst_computer`.

    `all_events` must be a superset that includes at least the
    `window_days` days preceding the earliest day in `events` — the caller
    is responsible for not pointing this at data beyond the current split's
    time horizon (US-107's leakage guard applies here as much as anywhere).
    """
    _require_networkx()

    days = sorted(events.select("day").unique().collect()["day"].to_list())
    per_day_frames: list[pl.DataFrame] = []

    for day in days:
        window_lo, window_hi = day - config.window_days, day - 1
        window_events = (
            all_events.filter(pl.col("day").is_between(window_lo, window_hi))
            .select(["src_user", "dst_computer"])
            .collect()
        )

        graph = build_daily_graph(window_events)
        feats = compute_daily_graph_features(graph, day, config)

        day_events = events.filter(pl.col("day") == day).collect()
        enriched = day_events.with_columns(
            [
                pl.col("src_user")
                .cast(pl.Utf8)
                .map_elements(_lookup_int_factory(feats.degree, "u:"), return_dtype=pl.Int64)
                .alias("graph_user_degree"),
                pl.col("dst_computer")
                .cast(pl.Utf8)
                .map_elements(_lookup_int_factory(feats.degree, "m:"), return_dtype=pl.Int64)
                .alias("graph_host_degree"),
                pl.col("src_user")
                .cast(pl.Utf8)
                .map_elements(_lookup_float_factory(feats.pagerank, "u:"), return_dtype=pl.Float64)
                .alias("graph_user_pagerank"),
                pl.col("dst_computer")
                .cast(pl.Utf8)
                .map_elements(_lookup_float_factory(feats.pagerank, "m:"), return_dtype=pl.Float64)
                .alias("graph_host_pagerank"),
                pl.col("src_user")
                .cast(pl.Utf8)
                .map_elements(
                    _lookup_float_factory(feats.clustering, "u:"), return_dtype=pl.Float64
                )
                .alias("graph_user_clustering"),
                pl.struct(["src_user", "dst_computer"])
                .map_elements(_adamic_adar_factory(feats.adamic_adar), return_dtype=pl.Float64)
                .alias("graph_pair_adamic_adar"),
                pl.col("src_user")
                .cast(pl.Utf8)
                .map_elements(
                    _embedding_factory(feats.embeddings, config.node2vec_dim),
                    return_dtype=pl.List(pl.Float64),
                )
                .alias("graph_user_embedding"),
            ]
        )
        per_day_frames.append(enriched)

    return pl.concat(per_day_frames).lazy() if per_day_frames else events
