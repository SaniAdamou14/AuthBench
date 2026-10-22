"""M5c — Euler (King and Huang, 2022): GCN snapshots, a GRU across them, temporal link prediction.

Reimplemented from the paper, since its code carries no licence. From the paper (sections IV
and VI): a directed, weighted graph from source to destination computers per 30-minute
snapshot; edge weight `sigmoid((C - mu) / sigma)` over the snapshot's edge counts; node
features a one-hot identifier concatenated with a one-hot node type; two GCN layers of 32
units, tanh, a 32-unit GRU and a projection to 16; decoder `sigmoid(z_u . z_v)`; predictor
mode, each snapshot's edges scored from the previous snapshot's embeddings; binary
cross-entropy against as many random non-edges; Adam; early stopping after 10 epochs without
improvement on 5% of each snapshot's edges held out. Not stated there, chosen here: dropout
0.1 on edges and between the GCN layers, a single linear projection, learning rate 0.01 (from
its grid), at most 100 epochs, back-propagation truncated to one day of snapshots, and one
shared node for computers unseen in training.

An event scores the negated log-odds of its edge under the previous snapshot's embeddings: an
unlikely edge scores high. `prepare` runs the trained model once along the whole timeline and
keeps every snapshot's embeddings on disk, so scoring is a lookup: causal, and the same for an
event alone or among a day of events.
"""

from __future__ import annotations

import copy
import re
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from authbench.models.base import BaseAnomalyScorer
from authbench.models.rowwise import row_sum
from authbench.models.torch_base import nn, torch

COLUMNS = ["time", "src_computer", "dst_computer"]
_COMPUTER = re.compile(r"^C\d+$")


def _snapshot_edges(frame: pl.LazyFrame, snapshot_seconds: int) -> pl.DataFrame:
    """Distinct (snapshot, source, destination) edges and their event counts.

    Aggregated one day at a time: a snapshot never spans two days, so this is exact, and
    memory is bounded by a day of events rather than the timeline (all of LANL at once
    needed more than 50 GB).
    """
    bounds = frame.select(pl.col("time").min().alias("first"), pl.col("time").max().alias("last"))
    first, last = bounds.collect().row(0)
    if first is None:
        return pl.DataFrame(
            schema={"snapshot": pl.Int64, "src": pl.Utf8, "dst": pl.Utf8, "len": pl.UInt32}
        )
    days = []
    for day in range(int(first) // 86_400, int(last) // 86_400 + 1):
        days.append(
            frame.filter(pl.col("time").is_between(day * 86_400, (day + 1) * 86_400, closed="left"))
            .select(
                (pl.col("time") // snapshot_seconds).cast(pl.Int64).alias("snapshot"),
                pl.col("src_computer").cast(pl.Utf8).alias("src"),
                pl.col("dst_computer").cast(pl.Utf8).alias("dst"),
            )
            .group_by("snapshot", "src", "dst")
            .len()
            .collect()
        )
    return pl.concat(days).sort("snapshot", "src", "dst")


class _Graph:
    """One snapshot: edge sources, targets and Euler's standardised-count weights."""

    def __init__(self, source: torch.Tensor, target: torch.Tensor, counts: torch.Tensor) -> None:
        self.source, self.target = source, target
        if counts.numel():
            spread = counts.std(unbiased=False)
            standard = (counts - counts.mean()) / spread if spread > 0 else torch.zeros_like(counts)
            self.weight = torch.sigmoid(standard)
        else:
            self.weight = counts

    def without(self, keep: torch.Tensor) -> _Graph:
        graph = copy.copy(self)
        graph.source, graph.target, graph.weight = (
            self.source[keep],
            self.target[keep],
            self.weight[keep],
        )
        return graph


def _propagate(graph: _Graph, h: torch.Tensor) -> torch.Tensor:
    """GCN aggregation from sources to targets with self-loops and symmetric degree normalisation."""
    n = h.shape[0]
    loops = torch.arange(n)
    source = torch.cat([graph.source, loops])
    target = torch.cat([graph.target, loops])
    weight = torch.cat([graph.weight, torch.ones(n)])
    degree = torch.zeros(n).index_add_(0, target, weight)
    norm = weight * degree[source].rsqrt() * degree[target].rsqrt()
    return torch.zeros_like(h).index_add_(0, target, h[source] * norm[:, None])


class _Euler(nn.Module):
    def __init__(self, n_nodes: int, hidden: int, embedding: int, dropout: float) -> None:
        super().__init__()
        # One-hot identifiers times the first weight matrix: a row per node, plus a shared unseen node.
        self.ids = nn.Embedding(n_nodes + 1, hidden)
        self.types = nn.Embedding(2, hidden)
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.second = nn.Linear(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.project = nn.Linear(hidden, embedding)
        self.dropout = dropout

    def step(
        self, graph: _Graph, node_type: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and graph.source.numel():
            graph = graph.without(torch.rand(graph.source.numel()) >= self.dropout)
        x = self.ids.weight + self.types(node_type)
        h = torch.relu(_propagate(graph, x) + self.bias)
        h = nn.functional.dropout(h, self.dropout, self.training)
        h = torch.tanh(_propagate(graph, self.second(h)))
        state = self.gru(h, state)
        return self.project(state), state


class EulerScorer(BaseAnomalyScorer):
    """M5c — Euler's GCN + GRU predictor, scored per event from the previous snapshot."""

    name = "M5c_euler"
    requires_labels = False
    scores_row_locally = True

    def __init__(
        self,
        snapshot_seconds: int = 1800,
        hidden: int = 32,
        embedding: int = 16,
        dropout: float = 0.1,
        learning_rate: float = 0.01,
        epochs: int = 100,
        patience: int = 10,
        validation_fraction: float = 0.05,
        window: int = 48,
        seed: int = 42,
        cache_dir: str | None = None,
    ) -> None:
        self.snapshot_seconds = snapshot_seconds
        self.hidden = hidden
        self.embedding = embedding
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.window = window
        self.seed = seed
        self.cache_dir = cache_dir
        self.history: list[tuple[int, float, float]] = []
        self._nodes: dict[str, int] = {}
        self._module: _Euler | None = None
        self._embeddings: np.ndarray | None = None
        self._first_snapshot = 0

    def _index(self, names: pl.Series) -> torch.Tensor:
        unseen = len(self._nodes)
        mapped = names.replace_strict(self._nodes, default=unseen, return_dtype=pl.Int64)
        return torch.from_numpy(mapped.to_numpy().astype(np.int64))

    def _node_type(self) -> torch.Tensor:
        names = [*self._nodes, "unseen"]
        return torch.tensor([0 if _COMPUTER.match(n) else 1 for n in names], dtype=torch.long)

    def _graphs(self, edges: pl.DataFrame, first: int, last: int) -> list[_Graph]:
        """One graph per snapshot in [first, last], empty snapshots included."""
        by_snapshot = {int(k[0]): g for k, g in edges.group_by("snapshot")} if edges.height else {}
        empty = torch.empty(0, dtype=torch.long)
        graphs = []
        for snapshot in range(first, last + 1):
            group = by_snapshot.get(snapshot)
            if group is None:
                graphs.append(_Graph(empty, empty, torch.empty(0)))
                continue
            graphs.append(
                _Graph(
                    self._index(group["src"]),
                    self._index(group["dst"]),
                    torch.from_numpy(group["len"].to_numpy().astype(np.float32)),
                )
            )
        return graphs

    def fit(self, train: pl.LazyFrame) -> None:
        deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            self._fit(train)
        finally:
            torch.use_deterministic_algorithms(deterministic)

    def _fit(self, train: pl.LazyFrame) -> None:
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        edges = _snapshot_edges(train, self.snapshot_seconds)
        names = sorted(set(edges["src"].to_list()) | set(edges["dst"].to_list()))
        self._nodes = {n: i for i, n in enumerate(names)}
        n_nodes = len(names)
        first, last = int(edges["snapshot"].min()), int(edges["snapshot"].max())  # type: ignore[arg-type]
        graphs = self._graphs(edges, first, last)
        held_out = [
            torch.rand(g.source.numel(), generator=generator) < self.validation_fraction
            for g in graphs
        ]
        node_type = self._node_type()
        validation_negatives = [
            torch.randint(0, n_nodes, (2, int(mask.sum())), generator=generator)
            for mask in held_out
        ]

        module = _Euler(n_nodes, self.hidden, self.embedding, self.dropout)
        optimizer = torch.optim.Adam(module.parameters(), lr=self.learning_rate)
        best_loss, best_state, stale = float("inf"), copy.deepcopy(module.state_dict()), 0
        self.history = []
        for epoch in range(1, self.epochs + 1):
            module.train()
            state = torch.zeros(n_nodes + 1, self.hidden)
            fit_loss, terms = 0.0, 0
            for start in range(0, len(graphs) - 1, self.window):
                optimizer.zero_grad()
                losses = []
                for t in range(start, min(start + self.window, len(graphs) - 1)):
                    z, state = module.step(graphs[t].without(~held_out[t]), node_type, state)
                    target = graphs[t + 1].without(~held_out[t + 1])
                    if target.source.numel():
                        negatives = torch.randint(
                            0, n_nodes, (2, target.source.numel()), generator=generator
                        )
                        losses.append(_link_loss(z, target.source, target.target, negatives))
                if losses:
                    loss = torch.stack(losses).mean()
                    loss.backward()  # type: ignore[no-untyped-call]
                    optimizer.step()
                    fit_loss += loss.item() * len(losses)
                    terms += len(losses)
                state = state.detach()
            holdout = self._holdout_loss(module, graphs, held_out, validation_negatives, node_type)
            self.history.append((epoch, fit_loss / max(terms, 1), holdout))
            if holdout < best_loss:
                best_loss, best_state, stale = holdout, copy.deepcopy(module.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        module.load_state_dict(best_state)
        self._module = module

    def _holdout_loss(
        self,
        module: _Euler,
        graphs: list[_Graph],
        held_out: list[torch.Tensor],
        negatives: list[torch.Tensor],
        node_type: torch.Tensor,
    ) -> float:
        module.eval()
        state = torch.zeros(len(self._nodes) + 1, self.hidden)
        losses = []
        with torch.no_grad():
            for t in range(len(graphs) - 1):
                z, state = module.step(graphs[t].without(~held_out[t]), node_type, state)
                mask = held_out[t + 1]
                if mask.any():
                    target = graphs[t + 1]
                    losses.append(
                        _link_loss(z, target.source[mask], target.target[mask], negatives[t + 1])
                    )
        return float(torch.stack(losses).mean()) if losses else float("inf")

    def prepare(self, timeline: list[pl.LazyFrame]) -> None:
        """Run the trained model along the timeline, in time order, and keep every snapshot's embeddings."""
        if self._module is None:
            raise RuntimeError(f"{self.name} must be fitted before it is prepared.")
        frames = [f.select(COLUMNS) for f in timeline]
        edges = pl.concat([_snapshot_edges(f, self.snapshot_seconds) for f in frames])
        first, last = int(edges["snapshot"].min()), int(edges["snapshot"].max())  # type: ignore[arg-type]
        graphs = self._graphs(edges, first, last)
        directory = Path(self.cache_dir or tempfile.mkdtemp(prefix="authbench_euler_"))
        directory.mkdir(parents=True, exist_ok=True)
        store = np.memmap(
            directory / f"{self.name}_{self.seed}.f32",
            dtype=np.float32,
            mode="w+",
            shape=(len(graphs), len(self._nodes) + 1, self.embedding),
        )
        node_type = self._node_type()
        self._module.eval()
        state = torch.zeros(len(self._nodes) + 1, self.hidden)
        with torch.no_grad():
            for t, graph in enumerate(graphs):
                z, state = self._module.step(graph, node_type, state)
                store[t] = z.numpy()
        store.flush()
        self._embeddings, self._first_snapshot = store, first

    def score(self, data: pl.LazyFrame) -> pl.Series:
        if self._embeddings is None:
            raise RuntimeError(f"{self.name} must be prepared on the timeline before scoring.")
        frame = data.select(COLUMNS).collect()
        previous = frame["time"].to_numpy() // self.snapshot_seconds - 1 - self._first_snapshot
        source = self._index(frame["src_computer"].cast(pl.Utf8)).numpy()
        target = self._index(frame["dst_computer"].cast(pl.Utf8)).numpy()
        known = (previous >= 0) & (previous < self._embeddings.shape[0])
        logits = np.zeros(frame.height)
        if known.any():
            at = previous[known]
            z_source = self._embeddings[at, source[known]].astype(np.float64)
            z_target = self._embeddings[at, target[known]].astype(np.float64)
            logits[known] = row_sum(z_source * z_target)
        # No previous snapshot: the edge has no history to be judged against, so it scores neutral.
        return pl.Series(self.name, -logits)


def _link_loss(
    z: torch.Tensor, source: torch.Tensor, target: torch.Tensor, negatives: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy of edges against as many random non-edges (Euler, equation 5)."""
    positive = (z[source] * z[target]).sum(dim=1)
    negative = (z[negatives[0]] * z[negatives[1]]).sum(dim=1)
    return nn.functional.binary_cross_entropy_with_logits(
        positive, torch.ones_like(positive)
    ) + nn.functional.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
