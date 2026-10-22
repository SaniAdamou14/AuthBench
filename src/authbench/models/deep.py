"""M4a dense autoencoder and M4b variational autoencoder over the shared design matrix (US-121).

Both are fitted on standardized `MODEL_FEATURE_COLUMNS` and scored by the
quantity they were trained to minimise, evaluated deterministically: the
squared reconstruction error for M4a, the negative ELBO at the posterior mean
for M4b. Only the rank of a score is ever used (spec section 3.5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from authbench.models.torch_base import Batch, TorchScorer, nn, torch


def _mlp(dims: list[int], *, activate_last: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(in_dim, out_dim))
        if activate_last or index < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


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

    def _inputs(self, data: pl.LazyFrame, *, fitting: bool) -> Batch:
        matrix = data.select(self.feature_cols).collect().to_numpy().astype(np.float64)
        scaled = self.scaler.fit_transform(matrix) if fitting else self.scaler.transform(matrix)
        return (torch.from_numpy(np.ascontiguousarray(scaled, dtype=np.float32)),)


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
