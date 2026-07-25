"""M4a/M4b — dense autoencoder and variational autoencoder (US-121, `authbench[deep]` extra).

Both score by reconstruction error / negative ELBO — never by a calibrated
probability — keeping them comparable to every other model under the
rank-only evaluation contract (spec section 3.5).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

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
            "M4 deep models require the optional 'deep' extra: "
            "install with `uv pip install -e '.[deep]'`."
        )


class _DenseAutoencoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers += [nn.Linear(dims[-1], input_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoencoderScorer(BaseAnomalyScorer):
    """M4a — dense autoencoder, scored by per-event mean squared reconstruction error."""

    name = "M4a_autoencoder"
    requires_labels = False

    def __init__(
        self,
        feature_cols: list[str],
        hidden_dims: list[int] | None = None,
        epochs: int = 30,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        _require_torch()
        self.feature_cols = feature_cols
        self.hidden_dims = hidden_dims or [64, 16, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.scaler = StandardScaler()
        self._model: _DenseAutoencoder | None = None

    def _matrix(self, data: pl.LazyFrame) -> np.ndarray:
        return data.select(self.feature_cols).collect().to_numpy().astype(np.float64)

    def fit(self, train: pl.LazyFrame) -> None:
        torch.manual_seed(self.seed)
        x = self.scaler.fit_transform(self._matrix(train))
        self._model = _DenseAutoencoder(x.shape[1], self.hidden_dims)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        x_tensor = torch.tensor(x, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstruction = self._model(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()

    def score(self, data: pl.LazyFrame) -> pl.Series:
        assert self._model is not None
        x = self.scaler.transform(self._matrix(data))
        x_tensor = torch.tensor(x, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            reconstruction = self._model(x_tensor)
            error = torch.mean((reconstruction - x_tensor) ** 2, dim=1).numpy()
        return pl.Series(self.name, error)


class _VAE(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            encoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(dims[-1], latent_dim)
        self.logvar = nn.Linear(dims[-1], latent_dim)

        decoder_dims = [latent_dim, *reversed(hidden_dims)]
        decoder_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(decoder_dims[:-1], decoder_dims[1:], strict=True):
            decoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        decoder_layers += [nn.Linear(decoder_dims[-1], input_dim)]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


class VAEScorer(BaseAnomalyScorer):
    """M4b — variational autoencoder, scored by the negative ELBO (reconstruction
    term + KL divergence) — a probabilistic score, still rank-only comparable.
    """

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
        seed: int = 42,
    ) -> None:
        _require_torch()
        self.feature_cols = feature_cols
        self.hidden_dims = hidden_dims or [64, 16]
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.seed = seed
        self.scaler = StandardScaler()
        self._model: _VAE | None = None

    def _matrix(self, data: pl.LazyFrame) -> np.ndarray:
        return data.select(self.feature_cols).collect().to_numpy().astype(np.float64)

    def fit(self, train: pl.LazyFrame) -> None:
        torch.manual_seed(self.seed)
        x = self.scaler.fit_transform(self._matrix(train))
        self._model = _VAE(x.shape[1], self.hidden_dims, self.latent_dim)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                x_hat, mu, logvar = self._model(batch)
                recon_loss = torch.mean((x_hat - batch) ** 2)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.kl_weight * kl
                loss.backward()
                optimizer.step()

    def score(self, data: pl.LazyFrame) -> pl.Series:
        assert self._model is not None
        x = self.scaler.transform(self._matrix(data))
        x_tensor = torch.tensor(x, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            x_hat, mu, logvar = self._model(x_tensor)
            recon_error = torch.mean((x_hat - x_tensor) ** 2, dim=1)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            neg_elbo = recon_error + self.kl_weight * kl
        return pl.Series(self.name, neg_elbo.numpy())
