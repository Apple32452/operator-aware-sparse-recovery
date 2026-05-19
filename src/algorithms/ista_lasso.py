import numpy as np


def soft_threshold(z: np.ndarray, lam: float) -> np.ndarray:
    """Coordinate-wise soft thresholding."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def ista_lasso(
    A: np.ndarray,
    y: np.ndarray,
    lam: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-7,
):
    """
    ISTA solver for the LASSO objective:

        0.5 ||y - A x||_2^2 + lam ||x||_1
    """
    _, n = A.shape
    x = np.zeros(n)

    # Lipschitz constant for gradient of 0.5 ||y - A x||^2.
    L = np.linalg.norm(A, ord=2) ** 2
    step = 1.0 / (L + 1e-12)

    for _ in range(max_iter):
        grad = A.T @ (A @ x - y)
        x_new = soft_threshold(x - step * grad, lam * step)

        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12) < tol:
            x = x_new
            break

        x = x_new

    support = np.flatnonzero(np.abs(x) > 1e-8)
    return x, support
