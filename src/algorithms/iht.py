import numpy as np


def hard_threshold(x: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k largest-magnitude entries."""
    if k >= len(x):
        return x.copy()

    out = np.zeros_like(x)
    support = np.argsort(np.abs(x))[-k:]
    out[support] = x[support]
    return out


def iht(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    max_iter: int = 100,
    step_size: float | None = None,
    tol: float = 1e-8,
):
    """
    Iterative Hard Thresholding for sparse recovery.

    Solves approximately:
        min_x 0.5 ||y - A x||_2^2
        subject to ||x||_0 <= k
    """
    _, n = A.shape
    x = np.zeros(n)

    if step_size is None:
        # Conservative step size using spectral norm.
        L = np.linalg.norm(A, ord=2) ** 2
        step_size = 1.0 / (L + 1e-12)

    for _ in range(max_iter):
        grad = A.T @ (y - A @ x)
        x_new = hard_threshold(x + step_size * grad, k)

        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12) < tol:
            x = x_new
            break

        x = x_new

    support = np.flatnonzero(np.abs(x) > 1e-10)
    return x, support
