import numpy as np


def omp(A: np.ndarray, y: np.ndarray, k: int, tol: float = 1e-10):
    """
    Orthogonal Matching Pursuit.

    Returns
    -------
    x_hat:
        Recovered signal.
    support:
        Selected support indices.
    """
    _, n = A.shape
    residual = y.copy()
    support = []

    for _ in range(k):
        corr = np.abs(A.T @ residual)

        if support:
            corr[np.array(support)] = -np.inf

        idx = int(np.argmax(corr))
        support.append(idx)

        As = A[:, support]
        coef, *_ = np.linalg.lstsq(As, y, rcond=None)
        residual = y - As @ coef

        if np.linalg.norm(residual) < tol:
            break

    x_hat = np.zeros(n)
    if support:
        x_hat[np.array(support)] = coef

    return x_hat, np.array(support, dtype=int)
