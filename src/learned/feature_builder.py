import numpy as np


def _safe_standardize(features: np.ndarray) -> np.ndarray:
    return (features - features.mean(axis=0, keepdims=True)) / (
        features.std(axis=0, keepdims=True) + 1e-12
    )


def _omp_residual_features(A: np.ndarray, y: np.ndarray, steps: int = 3):
    """
    Compute OMP-style diagnostic features:
    - whether coordinate was selected in early OMP steps
    - selection step
    - residual correlations after early greedy updates
    """
    _, n = A.shape
    residual = y.copy()
    support = []

    selected_flag = np.zeros(n)
    selection_step = np.zeros(n)

    residual_corrs = []

    for step in range(steps):
        corr = np.abs(A.T @ residual)

        if support:
            corr[np.array(support)] = -np.inf

        idx = int(np.argmax(corr))
        support.append(idx)

        selected_flag[idx] = 1.0
        selection_step[idx] = 1.0 / (step + 1)

        As = A[:, support]
        coef, *_ = np.linalg.lstsq(As, y, rcond=None)
        residual = y - As @ coef

        residual_corrs.append(np.abs(A.T @ residual))

    return selected_flag, selection_step, residual_corrs


def build_operator_features(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Build coordinate-wise operator-aware features.

    Features include:
    - absolute marginal correlation |a_i^T y|
    - signed marginal correlation a_i^T y
    - column norm
    - mean/max coherence
    - correlation rank score
    - early OMP selection diagnostics
    - residual correlations after OMP steps
    """
    _, n = A.shape

    corr = A.T @ y
    abs_corr = np.abs(corr)
    col_norm = np.linalg.norm(A, axis=0)

    gram = np.abs(A.T @ A)
    np.fill_diagonal(gram, 0.0)
    mean_coh = gram.mean(axis=1)
    max_coh = gram.max(axis=1)

    order = np.argsort(np.argsort(-abs_corr))
    rank_score = 1.0 - order / max(n - 1, 1)

    selected_flag, selection_step, residual_corrs = _omp_residual_features(
        A, y, steps=3
    )

    features = np.column_stack([
        abs_corr,
        corr,
        col_norm,
        mean_coh,
        max_coh,
        rank_score,
        selected_flag,
        selection_step,
        residual_corrs[0],
        residual_corrs[1],
        residual_corrs[2],
    ])

    features = _safe_standardize(features)
    return features.astype(np.float32)