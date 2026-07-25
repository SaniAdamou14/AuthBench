"""M5 — graph-based anomaly scores (US-123/US-124, `authbench[graph]` extra).

M5a scores the node2vec embedding of the acting user (from
`features.graph.attach_graph_features`, rebuilt per trailing 7-day window —
never on the full period) with an Isolation Forest. This is the model that
answers RQ3: does the graph carry signal independent of F2/F3?

M5b (GNN link prediction) is `Could`-priority in the spec (section 5, 21
points, "only if time permits") and is stubbed with a clear `NotImplementedError`
rather than a fake implementation — an honest gap beats a decorative one.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from authbench.models.base import BaseAnomalyScorer


class Node2VecIsolationForestScorer(BaseAnomalyScorer):
    """M5a — Isolation Forest over the per-event node2vec embedding of the
    acting user's node in that day's trailing-window authentication graph.
    """

    name = "M5a_node2vec_iforest"
    requires_labels = False

    def __init__(self, embedding_col: str = "graph_user_embedding", seed: int = 42) -> None:
        self.embedding_col = embedding_col
        self.seed = seed
        self._model: object | None = None

    def _matrix(self, data: pl.LazyFrame) -> np.ndarray:
        embeddings = data.select(self.embedding_col).collect()[self.embedding_col].to_list()
        return np.array(embeddings, dtype=np.float64)

    def fit(self, train: pl.LazyFrame) -> None:
        from pyod.models.iforest import IForest

        self._model = IForest(random_state=self.seed)
        self._model.fit(self._matrix(train))  # type: ignore[union-attr]

    def score(self, data: pl.LazyFrame) -> pl.Series:
        assert self._model is not None
        raw_scores = self._model.decision_function(self._matrix(data))  # type: ignore[attr-defined]
        return pl.Series(self.name, raw_scores)


class GNNLinkPredictionScorer(BaseAnomalyScorer):
    """M5b — GNN-based link prediction (spec: Could, 21 points, optional).

    Not implemented: this stage is explicitly conditional on schedule in the
    spec ("seulement si le temps le permet"). Left as a clearly-marked gap
    rather than a shallow implementation that would misrepresent M5b as
    measured when it is not.
    """

    name = "M5b_gnn_link_prediction"
    requires_labels = False

    def fit(self, train: pl.LazyFrame) -> None:
        raise NotImplementedError(
            "M5b is out of scope for the current sprint (spec section 5: Could, "
            "'seulement si le temps le permet'). Implement with PyTorch Geometric "
            "if S7/S8 schedule allows."
        )

    def score(self, data: pl.LazyFrame) -> pl.Series:
        raise NotImplementedError("See fit().")
