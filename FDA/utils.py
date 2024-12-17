import numpy as np
from scipy.stats import poisson

def generate_random_time_points(lambda_poisson=10, interval=(0, 1)):
    """
    Generate random observation time points in a given interval.

    Parameters:
        lambda_poisson (float): The expected value (rate) for the Poisson distribution to determine `m`, the number of time points.
        interval (tuple): The interval (start, end) within which time points are generated.

    Returns:
        np.ndarray: A sorted array of random observation time points.
    """
    m = poisson.rvs(mu=lambda_poisson)  # Sample m from a Poisson distribution
    time_points = np.sort(np.random.uniform(interval[0], interval[1], m))  # Uniformly distributed time points
    return time_points

def kl_representation_fourier(n_terms, eigenvalues, interval=(0, 1)):
    """
    Generate a stochastic process using Karhunen-Loève representation with Fourier basis functions.

    Parameters:
        n_terms (int): Number of terms in the KL expansion.
        eigenvalues (np.ndarray): Array of eigenvalues for the process.
        interval (tuple): The interval (start, end) for the process domain.

    Returns:
        callable: A function representing the stochastic process. For a given input `t`, it returns the process value.
    """
    if len(eigenvalues) != n_terms:
        raise ValueError("The length of eigenvalues must match n_terms.")

    # Independent standard normal random variables
    z = np.random.randn(n_terms)

    # Define Fourier basis functions for the interval
    def fourier_basis(k, t):
        L = interval[1] - interval[0]
        if k % 2 == 1:  # Odd index -> sine term
            return np.sqrt(2 / L) * np.sin((k // 2 + 1) * np.pi * t / L)
        else:  # Even index -> cosine term
            return np.sqrt(2 / L) * np.cos((k // 2) * np.pi * t / L)

    # Define the stochastic process as a function of time
    def stochastic_process(t):
        value = 0
        for k in range(n_terms):
            value += np.sqrt(eigenvalues[k]) * z[k] * fourier_basis(k, t)
        return value

    return stochastic_process

def generate_non_gaussian_process_fourier(lambda_poisson=10, n_terms=5, interval=(0, 1)):
    """
    Generate a non-Gaussian stochastic process with random time points and Fourier basis KL expansion.

    Parameters:
        lambda_poisson (float): The expected value (rate) for the Poisson distribution to determine `m`, the number of time points.
        n_terms (int): Number of terms in the KL expansion.
        interval (tuple): The interval (start, end) for time points.

    Returns:
        tuple: (t, y)
            t (np.ndarray): Random observation time points.
            y (np.ndarray): Observed values of the process at the time points.
    """
    # Generate random time points
    t = generate_random_time_points(lambda_poisson, interval)

    # Define eigenvalues for the process
    eigenvalues = 1 / (np.arange(1, n_terms + 1)**2)  # 1/k^2 decay for eigenvalues

    # Define the stochastic process using KL representation with Fourier basis
    process = kl_representation_fourier(n_terms, eigenvalues, interval)

    # Evaluate the process at the random time points
    y = np.array([process(ti) for ti in t])
    return t, y
