import numpy as np


def nrmse(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Normalized root mean squared error."""
    return float(np.linalg.norm(x_true - x_hat) / (np.linalg.norm(x_true) + 1e-12))


def support_metrics(true_support, pred_support):
    """Compute support precision, recall, F1, and exact match."""
    true_set = set(map(int, true_support))
    pred_set = set(map(int, pred_support))

    tp = len(true_set & pred_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(true_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    exact = int(true_set == pred_set)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_support": exact,
    }


def summarize_results(rows):
    """Average a list of metric dictionaries."""
    if not rows:
        return {}

    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
