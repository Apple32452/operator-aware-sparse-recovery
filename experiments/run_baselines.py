import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np

from src.data.make_sparse_signal import make_sparse_signal, make_compressible_signal
from src.data.make_sensing_matrix import gaussian_matrix, coherent_gaussian_matrix, make_measurements
from src.algorithms.omp import omp
from src.algorithms.cosamp import cosamp
from src.algorithms.ista_lasso import ista_lasso
from src.evaluation.metrics import nrmse, support_metrics


def run_one_trial(n, m, k, noise_std, matrix_type, signal_type, rng):
    if matrix_type == "gaussian":
        A = gaussian_matrix(m, n, rng)
    elif matrix_type == "coherent":
        A = coherent_gaussian_matrix(m, n, coherence_strength=0.25, rng=rng)
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    if signal_type == "sparse":
        x, support = make_sparse_signal(n, k, rng)
    elif signal_type == "compressible":
        x, support = make_compressible_signal(n, k, tail_scale=0.05, rng=rng)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    y = make_measurements(A, x, noise_std, rng)

    methods = {}

    x_omp, s_omp = omp(A, y, k)
    methods["OMP"] = (x_omp, s_omp)

    x_cosamp, s_cosamp = cosamp(A, y, k)
    methods["CoSaMP"] = (x_cosamp, s_cosamp)

    x_lasso, _ = ista_lasso(A, y, lam=0.01)
    s_lasso_topk = np.argsort(np.abs(x_lasso))[-k:]
    methods["ISTA_LASSO"] = (x_lasso, s_lasso_topk)

    rows = []
    for name, (x_hat, pred_support) in methods.items():
        sm = support_metrics(support, pred_support)
        rows.append({
            "method": name,
            "n": n,
            "m": m,
            "k": k,
            "noise_std": noise_std,
            "matrix_type": matrix_type,
            "signal_type": signal_type,
            "nrmse": nrmse(x, x_hat),
            **sm,
        })

    return rows


def main():
    rng = np.random.default_rng(42)

    settings = [
        {"n": 256, "m": 128, "k": 32, "noise_std": 0.01},
        {"n": 256, "m": 128, "k": 55, "noise_std": 0.01},
    ]

    all_rows = []
    trials = 20

    for setting in settings:
        for matrix_type in ["gaussian", "coherent"]:
            for signal_type in ["sparse", "compressible"]:
                for trial in range(trials):
                    rows = run_one_trial(
                        matrix_type=matrix_type,
                        signal_type=signal_type,
                        rng=rng,
                        **setting,
                    )
                    for row in rows:
                        row["trial"] = trial
                    all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = ROOT / "results" / "baseline_results.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved results to {out_path}")
    print()
    print(df.groupby(["n", "m", "k", "matrix_type", "signal_type", "method"])[
        ["nrmse", "precision", "recall", "f1", "exact_support"]
    ].mean().round(4))


if __name__ == "__main__":
    main()
