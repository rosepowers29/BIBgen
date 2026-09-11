from typing import Sequence, Iterable
import warnings

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
        """
        Class for normalizing and centering data.

        Parameters
        ----------
        mu : numpy.typing.ArrayLike
            Mean of data along axis 0
        std : numpy.typing.ArrayLike
            Standard deviation of data along axis 0

        Returns
        -------
        self : Sphering
            object for transforming and untransforming data
        """
        self.mu = mu
        self.std = std

    @classmethod
    def from_data(cls, features : ArrayLike):
        """
        Constructor from data

        Parameters
        ----------
        features : numpy.typing.ArrayLike
            Data array at least 1d. Mu and std will be computed along axis 0

        Returns
        -------
        self : Sphering
            object for transforming and untransforming data
        """
        return cls(np.mean(features, axis=0), np.std(features, axis=0))

    @classmethod
    def from_spherings(cls, instances : Iterable):
        """
        Constructor from collection of Sphering objects.
        Averages their mu and std.

        Parameters
        ----------
        instances : Iterable[Sphering]
            Collection of Sphering objects with the same shape for mu and std

        Returns
        -------
        self : Sphering
            object for transforming and untransforming data
        """
        mu_sum, std_sum = instances[0].mu, instances[0].std
        for i in range(1, len(instances)):
            mu_sum += instances[i].mu
            std_sum += instances[i].std
        return cls(mu_sum / len(instances), std_sum / len(instances))

    def transform(self, unsphered : ArrayLike):
        """
        Transform data to have zero mean and unit standard deviation.

        Parameters
        ----------
        unsphered : numpy.typing.ArrayLike
            Raw data with all dimensions of axes >= 1 equal to mu and std

        Returns
        -------
        sphered : numpy.ndarray
            Transformed data

        Examples
        --------
        >>> import numpy as np
        >>> import pytest
        >>> from BIBgen.preprocessing import Sphering
        >>> data = np.random.rand(50, 5)
        >>> sphering = Sphering.from_data(data)
        >>> sphered = sphering.transform(data)
        >>> np.mean(transformed, axis=0) == pytest.approx(np.zeros(5), abs=1e-4)
        True
        >>> np.std(transformed, axis=0) == pytest.approx(np.ones(5), abs=1e-4)
        True
        """
        return (unsphered - self.mu) / self.std

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

def quadratic_beta_schedule(n_timesteps : int, scale : float = 3e-5) -> np.ndarray:
    r"""
    Quadratic noise schedule: $\beta_\tau = \text{scale} \cdot \tau^2$ for $\tau = 1 \dots T$.

    With the defaults (``n_timesteps=100, scale=3e-5``), this reproduces ``config/noise_schedule.csv``
    exactly ($\beta_{max} = 0.3$, $\bar\alpha_T \approx 1.3\times 10^{-5}$).

    Parameters
    ----------
    n_timesteps : int
        Number of diffusion timesteps $T$.
    scale : float
        Coefficient multiplying $\tau^2$.

    Returns
    -------
    betas : numpy.ndarray
        Array of shape `(n_timesteps,)` with $\beta_\tau$ for $\tau = 1 \dots T$.
    """
    tau = np.arange(1, n_timesteps + 1)
    return scale * tau**2

def cosine_beta_schedule(
    n_timesteps : int,
    s : float = 0.008,
    target_alpha_bar_T : float = 1e-5,
    beta_clip : float = 0.999,
) -> np.ndarray:
    r"""
    Cosine-derived noise schedule (Nichol & Dhariwal, "Improved Denoising Diffusion
    Probabilistic Models", 2021), adapted to hit an explicit terminal $\bar\alpha_T$ rather
    than driving it asymptotically to zero as in the original (which assumes $T \approx 1000$).

    Defines $f(t) = \cos^2(\theta(t))$ for $t = 0 \dots T$, where $\theta(t)$ is linearly
    interpolated between $\theta_0 = \frac{\pi}{2}\frac{s}{1+s}$ and $\theta_1$, with $\theta_1$
    solved so that $\bar\alpha_T = f(T)/f(0)$ equals ``target_alpha_bar_T`` exactly. Then
    $\beta_\tau = 1 - \bar\alpha_\tau / \bar\alpha_{\tau - 1}$, clipped at ``beta_clip`` as a
    defensive numerical fuse.

    Parameters
    ----------
    n_timesteps : int
        Number of diffusion timesteps $T$.
    s : float
        Small offset controlling the schedule's curvature near $\tau=0$, as in the original paper.
    target_alpha_bar_T : float
        Desired cumulative product $\bar\alpha_T = \prod_\tau (1-\beta_\tau)$ at the final timestep.
    beta_clip : float
        Maximum allowed $\beta_\tau$; values above this are clipped and a warning is raised.

    Returns
    -------
    betas : numpy.ndarray
        Array of shape `(n_timesteps,)` with $\beta_\tau$ for $\tau = 1 \dots T$.
    """
    theta0 = (np.pi / 2) * (s / (1 + s))
    theta1 = np.arccos(np.sqrt(target_alpha_bar_T) * np.cos(theta0))

    t = np.arange(0, n_timesteps + 1)
    theta = theta0 + (theta1 - theta0) * t / n_timesteps
    f = np.cos(theta) ** 2
    alpha_bar = f / f[0]

    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    n_clipped = int(np.sum(beta > beta_clip))
    if n_clipped:
        warnings.warn(
            f"{n_clipped} beta value(s) exceeded beta_clip={beta_clip} and were clipped; "
            "the requested (n_timesteps, s, target_alpha_bar_T) combination pushes the "
            "cosine schedule past what beta_clip can represent honestly — consider raising "
            "target_alpha_bar_T or reducing n_timesteps.",
            stacklevel=2,
        )
    return np.minimum(beta, beta_clip)