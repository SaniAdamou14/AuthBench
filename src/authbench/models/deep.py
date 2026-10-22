"""M4a dense autoencoder and M4b variational autoencoder over the shared design matrix (US-121).

Both are fitted on standardized `MODEL_FEATURE_COLUMNS` and scored by the
quantity they were trained to minimise, evaluated deterministically: the
squared reconstruction error for M4a, the negative ELBO at the posterior mean
for M4b. Only the rank of a score is ever used (spec section 3.5).

Scoring runs the trained network in float64 `models.rowwise` arithmetic rather
than in batched float32 torch, whose results moved by up to 1e-6 with the batch
an event was scored in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from authbench.models.rowwise import affine, row_sum
from authbench.models.torch_base import Batch, TorchScorer, nn, torch


def _mlp(dims: list[int], *, activate_last: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(in_dim, out_dim))
        if activate_last or index < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _numpy(parameter: torch.Tensor) -> np.ndarray:
    return parameter.detach().to(torch.float64).numpy()


def _forward(sequential: nn.Sequential, x: np.ndarray) -> np.ndarray:
    """A Linear/ReLU stack evaluated in float64 row-wise arithmetic."""
    for layer in sequential:
        if isinstance(layer, nn.Linear):
            x = affine(x, _numpy(layer.weight), _numpy(layer.bias))
        elif isinstance(layer, nn.ReLU):
            x = np.maximum(x, 0.0)
        else:
            raise TypeError(f"No row-wise evaluation for {type(layer).__name__}.")
    return x


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.net = _mlp([input_dim, *hidden_dims, input_dim], activate_last=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore[no-any-return]


class _VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        self.encoder = _mlp([input_dim, *hidden_dims], activate_last=True)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = _mlp([latent_dim, *reversed(hidden_dims), input_dim], activate_last=False)

    def posterior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)


class _TabularScorer(TorchScorer):
    """Standardizes the design matrix with statistics fitted on the training sample only."""

    def __init__(self, feature_cols: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()

    scores_row_locally = True

    def _scaled(self, data: pl.LazyFrame, *, fitting: bool) -> np.ndarray:
        matrix = data.select(self.feature_cols).collect().to_numpy().astype(np.float64)
        scaled: np.ndarray = (
            self.scaler.fit_transform(matrix) if fitting else self.scaler.transform(matrix)
        )
        return scaled

    def _inputs(self, data: pl.LazyFrame, *, fitting: bool) -> Batch:
        scaled = self._scaled(data, fitting=fitting)
        return (torch.from_numpy(np.ascontiguousarray(scaled, dtype=np.float32)),)

    def _row_scores(self, module: nn.Module, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score(self, data: pl.LazyFrame) -> pl.Series:
        if self._module is None:
            raise RuntimeError(f"{self.name} must be fitted before scoring.")
        return pl.Series(
            self.name, self._row_scores(self._module, self._scaled(data, fitting=False))
        )


class AutoencoderScorer(_TabularScorer):
    """M4a: dense autoencoder, scored by the per-event mean squared reconstruction error."""

    name = "M4a_autoencoder"
    requires_labels = False

    def __init__(
        self,
        feature_cols: list[str],
        hidden_dims: list[int] | None = None,
        epochs: int = 30,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature_cols,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.hidden_dims = hidden_dims or [64, 16, 64]

    def _build(self, inputs: Batch) -> nn.Module:
        return _Autoencoder(inputs[0].shape[1], self.hidden_dims)

    def _loss(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        (x,) = batch
        reconstruction: torch.Tensor = module(x)
        return ((reconstruction - x) ** 2).mean(dim=1)

    def _score(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        return self._loss(module, batch)

    def _row_scores(self, module: nn.Module, x: np.ndarray) -> np.ndarray:
        assert isinstance(module, _Autoencoder)
        return row_sum((_forward(module.net, x) - x) ** 2) / x.shape[1]


class VAEScorer(_TabularScorer):
    """M4b: VAE with a unit-variance Gaussian decoder, scored by the negative ELBO at z = mu."""

    name = "M4b_vae"
    requires_labels = False

    def __init__(
        self,
        feature_cols: list[str],
        hidden_dims: list[int] | None = None,
        latent_dim: int = 8,
        epochs: int = 30,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        kl_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature_cols,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.hidden_dims = hidden_dims or [64, 16]
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def _build(self, inputs: Batch) -> nn.Module:
        return _VariationalAutoencoder(inputs[0].shape[1], self.hidden_dims, self.latent_dim)

    def _negative_elbo(self, module: nn.Module, x: torch.Tensor, *, sample: bool) -> torch.Tensor:
        assert isinstance(module, _VariationalAutoencoder)
        mu, logvar = module.posterior(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        decoded: torch.Tensor = module.decoder(z)
        reconstruction = 0.5 * ((decoded - x) ** 2).sum(dim=1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return reconstruction + self.kl_weight * kl

    def _loss(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        # Reparameterised sample in training mode; the posterior mean on the holdout keeps early stopping deterministic.
        return self._negative_elbo(module, batch[0], sample=module.training)

    def _score(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        return self._negative_elbo(module, batch[0], sample=False)

    def _row_scores(self, module: nn.Module, x: np.ndarray) -> np.ndarray:
        assert isinstance(module, _VariationalAutoencoder)
        hidden = _forward(module.encoder, x)
        mu = affine(hidden, _numpy(module.mu.weight), _numpy(module.mu.bias))
        logvar = affine(hidden, _numpy(module.logvar.weight), _numpy(module.logvar.bias))
        reconstruction = 0.5 * row_sum((_forward(module.decoder, mu) - x) ** 2)
        kl = -0.5 * row_sum(1 + logvar - mu**2 - np.exp(logvar))
        return reconstruction + self.kl_weight * kl
