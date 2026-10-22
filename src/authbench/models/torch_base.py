"""Shared training and scoring loop for the PyTorch models (M4a, M4b, M4c), `authbench[deep]` extra.

Fitting is bounded, seeded and label-free: a seeded sample of the training
partition is split into a fitting set and a holdout, and training stops when
the holdout loss has not improved for `patience` epochs, keeping the best
weights. Scoring is row-local and batched, so `train_eval` can chunk it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import polars as pl

from authbench.models.base import BaseAnomalyScorer, sample_for_fit

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on the optional extra
    raise ImportError(
        "The deep models need the optional 'deep' extra: `uv sync --extra deep`."
    ) from exc

Batch = tuple[torch.Tensor, ...]

__all__ = ["Batch", "EpochRecord", "TorchScorer", "nn", "torch"]


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    fit_loss: float
    holdout_loss: float


class TorchScorer(BaseAnomalyScorer):
    """Template: subclasses define `_inputs`, `_build`, `_loss` (per sample) and `_score` (per sample)."""

    scores_row_locally = True

    def __init__(
        self,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int = 3,
        holdout_fraction: float = 0.1,
        fit_sample_size: int | None = None,
        score_batch_rows: int = 262_144,
        seed: int = 42,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.holdout_fraction = holdout_fraction
        self.fit_sample_size = fit_sample_size
        self.score_batch_rows = score_batch_rows
        self.seed = seed
        self.history: list[EpochRecord] = []
        self._module: nn.Module | None = None

    def _inputs(self, data: pl.LazyFrame, *, fitting: bool) -> Batch:
        raise NotImplementedError

    def _build(self, inputs: Batch) -> nn.Module:
        raise NotImplementedError

    def _loss(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        raise NotImplementedError

    def _score(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        raise NotImplementedError

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
        inputs = self._inputs(sample_for_fit(train, self.fit_sample_size, self.seed), fitting=True)
        n_rows = inputs[0].shape[0]
        order = torch.randperm(n_rows, generator=generator)
        n_holdout = max(1, int(n_rows * self.holdout_fraction)) if n_rows > 1 else 0
        holdout = tuple(t[order[:n_holdout]] for t in inputs)
        fitting = tuple(t[order[n_holdout:]] for t in inputs)

        module = self._build(inputs)
        optimizer = torch.optim.Adam(module.parameters(), lr=self.learning_rate)
        best_loss, best_state, stale = float("inf"), copy.deepcopy(module.state_dict()), 0
        self.history = []
        for epoch in range(1, self.epochs + 1):
            module.train()
            fit_loss = 0.0
            permutation = torch.randperm(fitting[0].shape[0], generator=generator)
            for start in range(0, permutation.shape[0], self.batch_size):
                index = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = self._loss(module, tuple(t[index] for t in fitting)).mean()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                fit_loss += loss.item() * index.shape[0]
            fit_loss /= max(1, permutation.shape[0])
            holdout_loss = self._mean_loss(module, holdout) if n_holdout else fit_loss
            self.history.append(EpochRecord(epoch, fit_loss, holdout_loss))
            if holdout_loss < best_loss:
                best_loss, best_state, stale = holdout_loss, copy.deepcopy(module.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        module.load_state_dict(best_state)
        self._module = module

    def _mean_loss(self, module: nn.Module, data: Batch) -> float:
        module.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, data[0].shape[0], self.score_batch_rows):
                batch = tuple(t[start : start + self.score_batch_rows] for t in data)
                total += float(self._loss(module, batch).sum())
        return total / data[0].shape[0]

    def score(self, data: pl.LazyFrame) -> pl.Series:
        assert self._module is not None, "fit() must run before score()"
        inputs = self._inputs(data, fitting=False)
        self._module.eval()
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, inputs[0].shape[0], self.score_batch_rows):
                batch = tuple(t[start : start + self.score_batch_rows] for t in inputs)
                parts.append(self._score(self._module, batch).to(torch.float64).numpy())
        scores = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        return pl.Series(self.name, scores)
