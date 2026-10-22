"""Linear algebra whose result for a row depends on that row alone, to the last bit.

BLAS, MKL and oneDNN choose their accumulation order from the shape of the whole
matrix, so one event scored alone and the same event scored among millions can
differ (measured on the demo: one ULP for PCA, 1.6e-7 for the autoencoder). A
score that depends on its batch makes ranks depend on `chunk_rows`, a memory
setting. Accumulating one input column at a time with separate multiply and add
fixes the order of every sum, whatever the batch size, alignment or kernel.
"""

from __future__ import annotations

import numpy as np


def affine(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """`x @ weight.T + bias`, each output summed over the inputs in column order."""
    n_rows, n_out = x.shape[0], weight.shape[0]
    out = (
        np.zeros((n_rows, n_out)) if bias is None else np.tile(bias.astype(np.float64), (n_rows, 1))
    )
    weight = weight.astype(np.float64)
    for k in range(x.shape[1]):
        out += x[:, k, None] * weight[None, :, k]
    return out


def row_sum(x: np.ndarray) -> np.ndarray:
    """Sum of each row, added in column order."""
    out = np.zeros(x.shape[0])
    for k in range(x.shape[1]):
        out += x[:, k]
    return out
