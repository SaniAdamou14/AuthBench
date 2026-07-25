"""M4c — LSTM next-event prediction (US-122, `authbench[deep]` extra).

Consumes the F6 sequence encoding (the N most recent strictly-prior events of
the same user) and predicts the destination computer of the *current* event.
The anomaly score is the surprise of the actual outcome under the model:
`-log p(actual dst | context)` — an unbounded, rank-comparable score, exactly
like every other model in the catalog (spec section 3.5).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from authbench.models.base import BaseAnomalyScorer

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional `authbench[deep]` dependency
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "M4c requires the optional 'deep' extra: install with `uv pip install -e '.[deep]'`."
        )


class _NextEventLSTM(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        # index 0 is reserved for the padding sentinel (features.sequence.PAD_ID + 1 shift).
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequence)
        _, (h_n, _) = self.lstm(embedded)
        logits = self.output(h_n[-1])
        return logits


class NextEventLSTMScorer(BaseAnomalyScorer):
    """M4c. Vocabulary size is fixed at fit time from the training period's
    destination computers; unseen destinations at scoring time map to a
    single reserved out-of-vocabulary index rather than crashing.
    """

    name = "M4c_lstm_next_event"
    requires_labels = False

    def __init__(
        self,
        context_length: int = 32,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 1,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        _require_torch()
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self._model: _NextEventLSTM | None = None
        self._vocab_size: int = 0

    def _sequences_and_targets(self, data: pl.LazyFrame) -> tuple[np.ndarray, np.ndarray]:
        collected = data.select(["sequence_dst", "dst_computer"]).collect()
        sequences = np.array(collected["sequence_dst"].to_list(), dtype=np.int64)
        # Shift every id up by 1 so that PAD_ID (-1) becomes 0, the embedding's padding index.
        sequences = sequences + 1
        targets = collected["dst_computer"].to_physical().to_numpy().astype(np.int64) + 1
        return sequences, targets

    def fit(self, train: pl.LazyFrame) -> None:
        torch.manual_seed(self.seed)
        sequences, targets = self._sequences_and_targets(train)
        self._vocab_size = int(max(sequences.max(), targets.max())) + 1

        self._model = _NextEventLSTM(
            self._vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        seq_tensor = torch.tensor(sequences, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(seq_tensor, target_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for _ in range(self.epochs):
            for seq_batch, target_batch in loader:
                optimizer.zero_grad()
                logits = self._model(seq_batch)
                loss = loss_fn(logits, target_batch.clamp(max=self._vocab_size - 1))
                loss.backward()
                optimizer.step()

    def score(self, data: pl.LazyFrame) -> pl.Series:
        assert self._model is not None
        sequences, targets = self._sequences_and_targets(data)
        targets_clamped = np.clip(targets, 0, self._vocab_size - 1)

        seq_tensor = torch.tensor(sequences, dtype=torch.long)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(seq_tensor)
            log_probs = torch.log_softmax(logits, dim=1)
            target_tensor = torch.tensor(targets_clamped, dtype=torch.long)
            surprise = -log_probs.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
        return pl.Series(self.name, surprise.numpy())
