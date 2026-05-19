import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data.make_sparse_signal import make_sparse_signal, make_compressible_signal
from src.algorithms.iht import iht
from src.data.make_sparse_signal import make_sparse_signal
from src.data.make_sensing_matrix import (
    gaussian_matrix,
    coherent_gaussian_matrix,
    make_measurements,
)
from src.learned.feature_builder import build_operator_features
from src.learned.support_mlp import SupportMLP, predict_topk_support
from src.algorithms.omp import omp
from src.algorithms.cosamp import cosamp
from src.algorithms.subspace_pursuit import subspace_pursuit
from src.evaluation.metrics import nrmse, support_metrics
from src.utils.seed import set_seed


def least_squares_on_support(A, y, support, n):
    x_hat = np.zeros(n)
    if len(support) == 0:
        return x_hat
    As = A[:, support]
    coef, *_ = np.linalg.lstsq(As, y, rcond=None)
    x_hat[support] = coef
    return x_hat


def make_matrix(matrix_type, m, n, rng):
    if matrix_type == "gaussian":
        return gaussian_matrix(m, n, rng)
    if matrix_type == "coherent":
        return coherent_gaussian_matrix(m, n, coherence_strength=0.25, rng=rng)
    raise ValueError(f"Unknown matrix_type: {matrix_type}")


def make_dataset(num_instances, n, m, k, noise_std, matrix_type, rng):
    X_list = []
    y_list = []

    for _ in range(num_instances):
        A = make_matrix(matrix_type, m, n, rng)
        x, support = make_sparse_signal(n, k, rng)
        y = make_measurements(A, x, noise_std, rng)

        features = build_operator_features(A, y)
        labels = np.zeros(n, dtype=np.float32)
        labels[support] = 1.0

        X_list.append(features)
        y_list.append(labels)

    X = np.vstack(X_list).astype(np.float32)
    labels = np.concatenate(y_list).astype(np.float32)
    return X, labels


def train_model(X_train, y_train, input_dim, epochs=20, batch_size=512, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SupportMLP(input_dim=input_dim, hidden_dim=64).to(device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(xb)

        if epoch in {0, epochs - 1}:
            print(f"Epoch {epoch + 1:02d} | loss = {total_loss / len(ds):.6f}")

    return model, device


def evaluate_setting(
    model,
    device,
    n,
    m,
    k,
    noise_std,
    matrix_type,
    num_trials,
    rng,
):
    rows = []

    for trial in range(num_trials):
        A = make_matrix(matrix_type, m, n, rng)
        x, support = make_compressible_signal(n, k, tail_scale=0.05, rng=rng)
        y = make_measurements(A, x, noise_std, rng)

        # Correlation TopK
        proxy = A.T @ y
        s_corr = np.argsort(np.abs(proxy))[-k:]
        x_corr = least_squares_on_support(A, y, s_corr, n)

        # Learned support
        features = build_operator_features(A, y)
        s_learned = predict_topk_support(model, features, k, device=device)
        x_learned = least_squares_on_support(A, y, s_learned, n)

        # OMP and CoSaMP
        x_omp, s_omp = omp(A, y, k)
        x_cosamp, s_cosamp = cosamp(A, y, k)
        x_sp, s_sp = subspace_pursuit(A, y, k)
        x_iht, s_iht = iht(A, y, k)

        # OMP + learned union
        candidate_support = np.union1d(s_omp, s_learned)
        x_candidate = least_squares_on_support(A, y, candidate_support, n)
        s_union = np.argsort(np.abs(x_candidate))[-k:]
        x_union = least_squares_on_support(A, y, s_union, n)

        methods = {
            "CorrelationTopK": (x_corr, s_corr),
            "LearnedSupportMLP": (x_learned, s_learned),
            "IHT": (x_iht, s_iht),
            "CoSaMP": (x_cosamp, s_cosamp),
            "SubspacePursuit": (x_sp, s_sp),
            "OMP": (x_omp, s_omp),
            "OMP_LearnedUnion": (x_union, s_union),
        }

        for method, (x_hat, pred_support) in methods.items():
            sm = support_metrics(support, pred_support)
            rows.append({
                "trial": trial,
                "matrix_type": matrix_type,
                "n": n,
                "m": m,
                "k": k,
                "m_over_n": m / n,
                "k_over_m": k / m,
                "noise_std": noise_std,
                "method": method,
                "nrmse": nrmse(x, x_hat),
                **sm,
            })

    return rows


def main():
    set_seed(42)
    rng = np.random.default_rng(42)

    n = 256
    noise_std = 0.01
    matrix_types = ["gaussian", "coherent"]

    # Keep this grid small first so it runs reasonably fast.
    m_values = [96, 128, 160]
    k_over_m_values = [0.25, 0.35, 0.45]

    train_instances = 1000
    eval_trials = 100

    all_rows = []

    for matrix_type in matrix_types:
        for m in m_values:
            for k_ratio in k_over_m_values:
                k = int(round(k_ratio * m))
                print("=" * 80)
                print(f"matrix={matrix_type}, n={n}, m={m}, k={k}, k/m={k/m:.3f}")

                X_train, y_train = make_dataset(
                    num_instances=train_instances,
                    n=n,
                    m=m,
                    k=k,
                    noise_std=noise_std,
                    matrix_type=matrix_type,
                    rng=rng,
                )

                model, device = train_model(
                    X_train,
                    y_train,
                    input_dim=X_train.shape[1],
                    epochs=20,
                )

                rows = evaluate_setting(
                    model=model,
                    device=device,
                    n=n,
                    m=m,
                    k=k,
                    noise_std=noise_std,
                    matrix_type=matrix_type,
                    num_trials=eval_trials,
                    rng=rng,
                )
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = ROOT / "results" / "compressible_phase_transition_results.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved phase-transition results to {out_path}")
    print(
        df.groupby(["matrix_type", "m", "k", "method"])[["nrmse", "f1"]]
        .mean()
        .round(4)
    )


if __name__ == "__main__":
    main()
