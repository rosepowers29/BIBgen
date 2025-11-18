from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

def whitening_matrices(features : ArrayLike):
    r"""
    Compute whitening matrix $W$ and centering vector $\mu$,
    such that applying the transformation $x \to W (x - \mu)$
    yields features which are zero-mean, uncorrelated, and with normalized variance.

    Parameters
    ----------
    features : numpy.typing.ArrayLike
        Input data with shape `(n_samples, n_features)`

    Returns
    -------
    W : numpy.ndarray
        Matrix with dimensions `(n_features, n_features)`, corresponding with $W$ above.
    W_inv : numpy.ndarray
        Inverse of `W`, also with dimensions `(n_features, n_features)`
    mu : numpy.ndarray
        Mean of features, equal to `np.mean(features, axis=0)`, corresponding with $\mu$ above.

    Examples
    --------
    >>> import numpy as np
    >>> import pytest
    >>> from BIBgen.preprocessing import whitening_matrices
    >>> data = np.random.rand(50, 5)
    >>> W, W_inv, mu = whitening_matrices(data)
    >>> transformed = (data - mu) @ W.T
    >>> np.mean(transformed, axis=0) == pytest.approx(np.zeros(5), abs=1e-4)
    True
    >>> np.cov(transformed, rowvar=False) == pytest.approx(np.identity(5), abs=1e-4)
    True
    """
    mu = np.mean(features, axis=0)
    centered = features - mu

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    eps = 1e-6
    D_inv_sqrt = np.diag(1 / np.sqrt(eigvals + eps))
    D_sqrt = np.diag(np.sqrt(eigvals + eps))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T
    W_inv = eigvecs @ D_sqrt @ eigvecs.T

    return W, W_inv, mu

class Sphering:
    def __init__(self, mu : ArrayLike, std : ArrayLike):
        self.mu = mu
        self.std = std

    @classmethod
    def from_data(cls, features : ArrayLike):
        return cls(np.mean(features, axis=0), np.std(features, axis=0))

    @classmethod
    def from_npy(cls, path : str):
        assert path.endswith(".npy")
        arr = np.load(path)
        return cls(arr[0], arr[1])

    def save_npy(self, path : str):
        assert path.endswith(".npy")
        np.save(path, np.stack((self.mu, self.std)))

    def transform(self, unsphered):
        return (new_data - self.mu) / self.std

    def untransform(self, sphered):
        return self.std * sphered + self.mu

def diffuse(features : ArrayLike, betas : Sequence) -> ArrayLike:
    r"""
    Applies diffusion by iteratively adding Gaussian noise.
    At each timestep, the features are perturbed by $x_{\tau+1} = \sqrt{1 - beta_\tau} x_\tau + \sqrt{\beta_\tau} z_\tau$,
    where $z \sim \mathcal{N}(0, I)$.

    Parameters
    ----------
    features : np.typing.ArrayLike
        Normalized input data with dimensions `(n_samples, n_features)`
    betas : typing.Sequence
        Noise schedule which prescribes $\beta_t$ above. Dimensions are `(n_timesteps,)`

    Returns
    -------
    result : numpy.ndarray
        Noisy samples for each iteration with dimensions `(n_timesteps + 1, n_samples, n_features)`.
        The first element `result[0]` is the same as `features` for convenience.
    """
    result = np.empty((len(betas) + 1, *np.shape(features)))
    result[0] = features
    z = np.random.normal(size=(len(betas), *np.shape(features)))

    for tau in range(len(betas)):
        result[tau + 1] = np.sqrt(1 - betas[tau]) * result[tau] + np.sqrt(betas[tau]) * z[tau]

    return result