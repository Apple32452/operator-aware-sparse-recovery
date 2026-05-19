import numpy as np


def make_sparse_signal(n: int, k: int, rng: np.random.Generator):
    """Create an exactly k-sparse signal x in R^n."""
    if k > n:
        raise ValueError("k cannot be larger than n.")

    x = np.zeros(n)
    support = rng.choice(n, size=k, replace=False)
    x[support] = rng.normal(loc=0.0, scale=1.0, size=k)
    return x, np.sort(support)


def make_compressible_signal(
    n: int,
    k: int,
    tail_scale: float,
    rng: np.random.Generator,
):
    """
    Create a signal with k dominant entries plus small off-support noise.

    This is closer to realistic sparse recovery than a perfectly sparse signal.
    """
    if k > n:
        raise ValueError("k cannot be larger than n.")

    x = rng.normal(0.0, tail_scale, size=n)
    support = rng.choice(n, size=k, replace=False)
    x[support] = rng.normal(0.0, 1.0, size=k)
    return x, np.sort(support)
