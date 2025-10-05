"""
Sample generators for functional data simulations.

Implements:
- KL-based 5D random process using Fourier cosine basis with dimension-specific variances
- Mean functions per dimension
- Symmetric Toeplitz dependence across dimensions via matrix A
- Sparse sampling with uniform time points and SNR-controlled Gaussian noise

All code and comments are in English.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional

from .functional_data import Trajectory, FunctionalData, ObservationType


def toeplitz_first_row_to_matrix(first_row: np.ndarray) -> np.ndarray:
    """Create a symmetric Toeplitz matrix from its first row.

    Parameters
    ----------
    first_row : np.ndarray
        First row values. The matrix will be symmetric Toeplitz.

    Returns
    -------
    np.ndarray
        Symmetric Toeplitz matrix of shape (d, d).
    """
    d = len(first_row)
    A = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            A[i, j] = first_row[abs(i - j)]
    return A


def mean_functions(t: np.ndarray) -> np.ndarray:
    """Return 5D mean functions evaluated at t.

    mu_1(t) = 1
    mu_2(t) = sin(2πt)
    mu_3(t) = exp(t)
    mu_4(t) = 4t(1-t)
    mu_5(t) = 10 t^3 - 15 t^4 + 6 t^5

    Parameters
    ----------
    t : np.ndarray
        Time points in [0, 1].

    Returns
    -------
    np.ndarray
        Array of shape (len(t), 5) with mean values per dimension.
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros((t.shape[0], 5), dtype=float)
    out[:, 0] = 1.0
    out[:, 1] = np.sin(2 * np.pi * t)
    out[:, 2] = np.exp(t)
    out[:, 3] = 4.0 * t * (1.0 - t)
    out[:, 4] = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    return out


def cosine_basis(t: np.ndarray, K: int) -> np.ndarray:
    """Fourier cosine basis with L2 normalization on [0,1].

    f_k(t) = sqrt(2) * cos(pi * k * t), for k=1..K
    ∫_0^1 f_k(t)^2 dt = 1

    Parameters
    ----------
    t : np.ndarray
        Time points.
    K : int
        Number of basis functions.

    Returns
    -------
    np.ndarray
        Basis matrix of shape (len(t), K).
    """
    t = np.asarray(t, dtype=float)
    k = np.arange(1, K + 1, dtype=float)
    return np.sqrt(2.0) * np.cos(np.pi * np.outer(t, k))


def lambda_sequences(K: int) -> np.ndarray:
    """Return (5, K) array of lambda_i,k per dimension.

    Dimension-specific choices:
    - dim 1: decays ~ 1 / (1 + k^2) to emulate exp(-|t-s|) covariance on cosine basis
    - dim 2: k^{-2}
    - dim 3: 0.5 * k^{-2}
    - dim 4: k^{-1.5}
    - dim 5: 0.5 * k^{-1.5}

    Parameters
    ----------
    K : int
        Number of KL modes.

    Returns
    -------
    np.ndarray
        Lambda array of shape (5, K).
    """
    k = np.arange(1, K + 1, dtype=float)
    lam = np.zeros((5, K), dtype=float)
    lam[0, :] = 1.0 / (1.0 + k**2)
    lam[1, :] = k**-2
    lam[2, :] = 0.5 * k**-2
    lam[3, :] = k**-1.5
    lam[4, :] = 0.5 * k**-1.5
    return lam


def kl_random_process(t: np.ndarray, K: int, rng: np.random.Generator, lam: np.ndarray) -> np.ndarray:
    """Generate centered 5D random process via KL representation.

    X_d(t) = sum_{k=1..K} u_{d,k} f_k(t),
    where u_{d,k} ~ Uniform(-a, a) with a = sqrt(3 * lambda_{d,k}) to ensure Var(u_{d,k}) = lambda_{d,k}.

    Parameters
    ----------
    t : np.ndarray
        Time points.
    K : int
        Number of KL modes.
    rng : np.random.Generator
        Random generator.
    lam : np.ndarray
        Lambda sequences of shape (5, K).

    Returns
    -------
    np.ndarray
        Centered process values of shape (len(t), 5).
    """
    B = cosine_basis(t, K)  # (m, K)
    a = np.sqrt(3.0 * lam)  # (5, K)
    U = rng.uniform(low=-a, high=a)  # (5, K), element-wise bounds
    # Compute X(t) = B @ U^T for each dimension
    X = B @ U.T  # (m, 5)
    return X


def compute_signal_moment(domain: Tuple[float, float], lam: np.ndarray, A: np.ndarray, grid_size: int = 501) -> float:
    """Approximate the average second-order moment over dimensions.

    We combine KL variances and mean energy:
    - KL part: for each source dim j, S_j = sum_k lambda_{j,k}
      Output dim d contribution: sum_j A[d,j]^2 * S_j
    - Mean part: ∫ mu_d(t)^2 dt approximated on a dense grid

    Returns
    -------
    float
        Average over output dimensions of the total second moment.
    """
    a, b = domain
    t_grid = np.linspace(a, b, grid_size)
    mu = mean_functions((t_grid - a) / (b - a))  # scale to [0,1] for mu
    mean_energy = np.trapz(mu**2, x=t_grid, axis=0) / (b - a)

    S = lam.sum(axis=1)  # (5,)
    kl_energy = (A**2 @ S)  # (5,)
    total = mean_energy + kl_energy
    return float(total.mean())


def generate_sparse_dataset(
    n: int,
    m: int,
    snr: float,
    domain: Tuple[float, float] = (0.0, 1.0),
    K: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[FunctionalData, List[Trajectory], float]:
    """Generate a sparse FunctionalData dataset under the specified simulation design.

    Parameters
    ----------
    n : int
        Number of trajectories.
    m : int
        Baseline observations per trajectory. Each m_i is drawn uniformly from [m-2, m+2].
    snr : float
        Target signal-to-noise ratio between 2 and 5.
    domain : tuple[float, float]
        Time domain, default (0,1).
    K : int
        Number of KL modes for the random process.
    rng : np.random.Generator, optional
        Random generator. If None, uses default_rng(2025).

    Returns
    -------
    FunctionalData, list[Trajectory], float
        The dataset, list of trajectories, and the noise standard deviation used.
    """
    if rng is None:
        rng = np.random.default_rng(2025)

    # Dependence matrix A (symmetric Toeplitz)
    first_row = np.array([1.0, 0.4, 0.3, 0.2, 0.1], dtype=float)
    A = toeplitz_first_row_to_matrix(first_row)

    # Lambda sequences for KL per dimension
    lam = lambda_sequences(K)

    # Calibrate noise to achieve desired SNR using moment approximation
    signal_moment = compute_signal_moment(domain, lam, A)
    sigma2 = signal_moment / snr
    sigma = float(np.sqrt(sigma2))

    a, b = domain
    trajectories: List[Trajectory] = []

    for _ in range(n):
        # Random m_i in [m-2, m+2]
        m_i = int(rng.integers(low=max(1, m - 2), high=m + 3))
        # Uniform time points in [a, b]
        t_i = rng.uniform(low=a, high=b, size=m_i)
        t_i.sort()

        # Centered random process on scaled t in [0,1]
        t_unit = (t_i - a) / (b - a)
        X = kl_random_process(t_unit, K, rng, lam)  # (m_i, 5)

        # Apply dependence and add mean
        mu = mean_functions(t_unit)  # (m_i, 5)
        zeta = X @ A.T + mu  # (m_i, 5)

        # Add Gaussian noise with variance sigma2, independent across samples and dims
        eps = rng.normal(loc=0.0, scale=sigma, size=zeta.shape)
        Y = zeta + eps

        traj = Trajectory(
            observation_type=ObservationType.SPARSE,
            domain=(a, b),
            t=t_i,
            y=Y,
            regular=False,
        )
        trajectories.append(traj)

    fd = FunctionalData(trajectories)
    return fd, trajectories, sigma