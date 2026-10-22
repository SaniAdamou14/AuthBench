"""M4c: LSTM next-destination model over the F6 sequences (US-122).

The score is the surprise of the event's actual destination given the user's
prior context, `-log p(dst_code | sequence_dst)`. Codes come from the
train-fitted vocabulary, so a destination never seen in training maps to
`OOV_ID`, a class no training target ever reinforces, instead of being
aliased to a known machine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from authbench.features.vocabulary import PAD_ID
from authbench.models.torch_base import Batch, TorchScorer, nn, torch


class _NextDestinationLSTM(nn.Module):
    def __init__(self, n_tokens: int, embedding_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_tokens)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(self.embedding(sequence))
        return self.output(hidden[-1])  # type: ignore[no-any-return]


class NextEventLSTMScorer(TorchScorer):
    """M4c. `n_tokens` is `features.vocabulary.n_tokens` of the destination vocabulary."""

    name = "M4c_lstm_next_event"
    requires_labels = False

    def __init__(
        self,
        n_tokens: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 1,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.n_tokens = n_tokens
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _inputs(self, data: pl.LazyFrame, *, fitting: bool) -> Batch:
        frame = data.select("sequence_dst", "dst_code").collect()
        sequences = np.asarray(frame["sequence_dst"].to_list(), dtype=np.int64)
        targets = frame["dst_code"].to_numpy().astype(np.int64)
        if sequences.size and (sequences.max() >= self.n_tokens or targets.max() >= self.n_tokens):
            raise ValueError(
                f"Token id outside [0, {self.n_tokens}): the sequences were encoded with a different vocabulary."
            )
        return torch.from_numpy(sequences), torch.from_numpy(targets)

    def _build(self, inputs: Batch) -> nn.Module:
        return _NextDestinationLSTM(
            self.n_tokens, self.embedding_dim, self.hidden_dim, self.num_layers
        )

    def _loss(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        sequences, targets = batch
        return nn.functional.cross_entropy(module(sequences), targets, reduction="none")

    def _score(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        return self._loss(module, batch)
