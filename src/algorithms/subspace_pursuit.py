import numpy as np


def subspace_pursuit(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    max_iter: int = 20,
    tol: float = 1e-10,
):
    """
    Subspace Pursuit sparse recovery algorithm.

    Parameters
    ----------
    A:
        Sensing matrix of shape (m, n).
    y:
        Measurement vector of shape (m,).
    k:
        Sparsity level.
    max_iter:
        Maximum number of iterations.
    tol:
        Residual stopping tolerance.

    Returns
    -------
    x_hat:
        Recovered signal.
    support:
        Estimated support.
    """
    _, n = A.shape

    # Initial support from top-k correlations.
    proxy = A.T @ y
    support = np.argsort(np.abs(proxy))[-k:]

    x_hat = np.zeros(n)
    residual = y.copy()
    prev_residual_norm = np.inf

    for _ in range(max_iter):
        # Least squares on current support.
        As = A[:, support]
        coef, *_ = np.linalg.lstsq(As, y, rcond=None)

        x_temp = np.zeros(n)
        x_temp[support] = coef

        residual = y - A @ x_temp
        residual_norm = np.linalg.norm(residual)

        if residual_norm < tol:
            x_hat = x_temp
            break

        # Identify new candidates from residual correlation.
        proxy = A.T @ residual
        candidate_support = np.argsort(np.abs(proxy))[-k:]

        # Merge old and new supports.
        merged_support = np.union1d(support, candidate_support)

        # Least squares on merged support.
        A_merged = A[:, merged_support]
        coef_merged, *_ = np.linalg.lstsq(A_merged, y, rcond=None)

        x_merged = np.zeros(n)
        x_merged[merged_support] = coef_merged

        # Prune back to k largest coefficients.
        new_support = np.argsort(np.abs(x_merged))[-k:]

        A_new = A[:, new_support]
        coef_new, *_ = np.linalg.lstsq(A_new, y, rcond=None)

        x_new = np.zeros(n)
        x_new[new_support] = coef_new

        new_residual = y - A @ x_new
        new_residual_norm = np.linalg.norm(new_residual)

        # Stop if no improvement.
        if new_residual_norm >= prev_residual_norm - tol:
            x_hat = x_new
            support = new_support
            break

        x_hat = x_new
        support = new_support
        prev_residual_norm = new_residual_norm

    return x_hat, np.array(support, dtype=int)
