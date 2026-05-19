import numpy as np


def cosamp(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    max_iter: int = 20,
    tol: float = 1e-10,
):
    """
    CoSaMP sparse recovery algorithm.

    This is a standard greedy baseline for compressed sensing.
    """
    _, n = A.shape
    x_hat = np.zeros(n)
    residual = y.copy()
    support = np.array([], dtype=int)

    for _ in range(max_iter):
        proxy = A.T @ residual
        omega = np.argsort(np.abs(proxy))[-2 * k:]

        merged = np.union1d(support, omega)
        As = A[:, merged]
        coef, *_ = np.linalg.lstsq(As, y, rcond=None)

        keep_local = np.argsort(np.abs(coef))[-k:]
        support = merged[keep_local]

        As_keep = A[:, support]
        coef_keep, *_ = np.linalg.lstsq(As_keep, y, rcond=None)

        x_new = np.zeros(n)
        x_new[support] = coef_keep

        residual = y - A @ x_new

        if np.linalg.norm(x_new - x_hat) < tol:
            x_hat = x_new
            break

        x_hat = x_new

        if np.linalg.norm(residual) < tol:
            break

    return x_hat, np.array(support, dtype=int)
