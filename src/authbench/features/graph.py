"""F5: authentication-graph features (US-113, `authbench[graph]` extra).

The bipartite user-machine graph is rebuilt per day from a strictly trailing
window of `window_days` days, never once over the whole period: an embedding
fitted on all days leaks every future edge into every past node and makes
RQ3 unanswerable (spec section 4.1). Day granularity is a cost trade-off;
no event contributes to its own day's graph.

Because the graph is bipartite, measures built on triangles or on common
neighbours are identically zero. Clustering is therefore the bipartite
coefficient of Latapy, Magnien and Del Vecchio (2008), and no common-neighbour
link score (Adamic-Adar, Jaccard) is computed between a user and a machine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

try:
    import networkx as nx
    from networkx.algorithms import bipartite
except ImportError:  # pragma: no cover - optional `authbench[graph]` dependency
    nx = None
    bipartite = None


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


GRAPH_FEATURE_COLUMNS = (
    "graph_user_degree",
    "graph_host_degree",
    "graph_user_pagerank",
    "graph_host_pagerank",
    "graph_user_clustering",
    "graph_user_community",
    "graph_user_embedding",
)


def _require_networkx() -> None:
    if nx is None:
        raise ImportError(
            "F5 graph features require the optional 'graph' extra: `uv sync --extra graph`."
        )


def build_daily_graph(events: pl.DataFrame) -> nx.Graph:
    """Bipartite graph over `events`, nodes prefixed `u:` and `m:`, edges weighted by authentication count."""
    _require_networkx()
    graph = nx.Graph()
    pairs = events.group_by(["src_user", "dst_computer"]).agg(pl.len().alias("weight"))
    for user, machine, weight in pairs.iter_rows():
        graph.add_edge(f"u:{user}", f"m:{machine}", weight=int(weight))
    return graph


def compute_daily_graph_features(
    graph: nx.Graph, day: int, config: GraphFeatureConfig
) -> DailyGraphFeatures:
    """Degree, weighted PageRank, bipartite clustering, Louvain community and node2vec embedding per node."""
    _require_networkx()
    # An empty trailing window (the first day of a split) has nothing to embed; node2vec would raise.
    if not graph.number_of_edges():
        return DailyGraphFeatures(day, {}, {}, {}, {}, {})

    from networkx.algorithms.community import louvain_communities
    from node2vec import Node2Vec

    communities = louvain_communities(graph, weight="weight", seed=config.seed)
    n2v = Node2Vec(
        graph,
        dimensions=config.node2vec_dim,
        walk_length=config.node2vec_walk_length,
        num_walks=config.node2vec_num_walks,
        seed=config.seed,
        workers=1,
        quiet=True,
    )
    model = n2v.fit(seed=config.seed, workers=1, min_count=1)
    return DailyGraphFeatures(
        day=day,
        degree=dict(graph.degree()),
        pagerank=nx.pagerank(graph, weight="weight"),
        clustering={node: float(c) for node, c in bipartite.clustering(graph).items()},
        community={node: index for index, members in enumerate(communities) for node in members},
        embeddings={node: model.wv[node].tolist() for node in graph.nodes()},
    )


def _lookup(
    table: dict[str, float] | dict[str, int], prefix: str, default: float
) -> Callable[[str], float]:
    return lambda key: float(table.get(f"{prefix}{key}", default))


def attach_graph_features(
    events: pl.LazyFrame, all_events: pl.LazyFrame, config: GraphFeatureConfig
) -> pl.LazyFrame:
    """Join, for each day D of `events`, the features of the graph built from `all_events` on days [D - window_days, D - 1].

    The caller must not point `all_events` beyond the current split's horizon (US-107).
    Unseen nodes take the values of an isolated node: degree, PageRank, clustering 0, community -1, zero embedding.
    """
    _require_networkx()
    days = sorted(events.select("day").unique().collect()["day"].to_list())
    zero_embedding = [0.0] * config.node2vec_dim
    per_day: list[pl.DataFrame] = []
    for day in days:
        window = (
            all_events.filter(pl.col("day").is_between(day - config.window_days, day - 1))
            .select(["src_user", "dst_computer"])
            .collect()
        )
        feats = compute_daily_graph_features(build_daily_graph(window), day, config)
        user = pl.col("src_user").cast(pl.String)
        host = pl.col("dst_computer").cast(pl.String)
        per_day.append(
            events.filter(pl.col("day") == day)
            .collect()
            .with_columns(
                user.map_elements(_lookup(feats.degree, "u:", 0), return_dtype=pl.Float64)
                .cast(pl.Int64)
                .alias("graph_user_degree"),
                host.map_elements(_lookup(feats.degree, "m:", 0), return_dtype=pl.Float64)
                .cast(pl.Int64)
                .alias("graph_host_degree"),
                user.map_elements(
                    _lookup(feats.pagerank, "u:", 0.0), return_dtype=pl.Float64
                ).alias("graph_user_pagerank"),
                host.map_elements(
                    _lookup(feats.pagerank, "m:", 0.0), return_dtype=pl.Float64
                ).alias("graph_host_pagerank"),
                user.map_elements(
                    _lookup(feats.clustering, "u:", 0.0), return_dtype=pl.Float64
                ).alias("graph_user_clustering"),
                user.map_elements(_lookup(feats.community, "u:", -1), return_dtype=pl.Float64)
                .cast(pl.Int64)
                .alias("graph_user_community"),
                user.map_elements(
                    lambda key: feats.embeddings.get(f"u:{key}", zero_embedding),
                    return_dtype=pl.List(pl.Float64),
                ).alias("graph_user_embedding"),
            )
        )
    return pl.concat(per_day).lazy() if per_day else events
