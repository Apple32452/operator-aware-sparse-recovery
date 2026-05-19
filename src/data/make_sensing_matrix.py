import numpy as np


def normalize_columns(A: np.ndarray) -> np.ndarray:
    """Normalize matrix columns to unit norm."""
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)


def gaussian_matrix(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian sensing matrix with normalized columns."""
    A = rng.normal(0.0, 1.0, size=(m, n))
    return normalize_columns(A)


def coherent_gaussian_matrix(
    m: int,
    n: int,
    coherence_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a harder coherent sensing matrix.

    A shared latent direction is mixed into every column.
    Larger coherence_strength makes recovery harder.
    """
    A = gaussian_matrix(m, n, rng)
    u = rng.normal(size=(m, 1))
    u = u / (np.linalg.norm(u) + 1e-12)
    shared = u @ rng.normal(size=(1, n))
    A = (1.0 - coherence_strength) * A + coherence_strength * shared
    return normalize_columns(A)


def make_measurements(
    A: np.ndarray,
    x: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate y = A x + noise."""
    y_clean = A @ x
    noise = rng.normal(0.0, noise_std, size=y_clean.shape)
    return y_clean + noise
