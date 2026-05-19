import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.make_sparse_signal import make_compressible_signal
from src.data.make_sensing_matrix import coherent_gaussian_matrix, make_measurements
from src.learned.feature_builder import build_operator_features
from src.learned.support_mlp import SupportMLP, predict_topk_support
from src.algorithms.omp import omp
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


def union_prune_refit(A, y, base_support, extra_support, k, n):
    """
    Union base support with extra candidates, fit LS on candidate set,
    prune to top-k coefficients, then refit LS.
    """
    candidate_support = np.union1d(base_support, extra_support)
    x_candidate = least_squares_on_support(A, y, candidate_support, n)
    final_support = np.argsort(np.abs(x_candidate))[-k:]
    x_final = least_squares_on_support(A, y, final_support, n)
    return x_final, final_support


def make_dataset(num_instances, n, m, k, noise_std, coherence_strength, rng):
    X_list = []
    y_list = []

    for _ in range(num_instances):
        A = coherent_gaussian_matrix(
            m,
            n,
            coherence_strength=coherence_strength,
            rng=rng,
        )

        x, support = make_compressible_signal(
            n,
            k,
            tail_scale=0.05,
            rng=rng,
        )

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


def evaluate_ablation(
    model,
    device,
    n,
    m,
    k,
    noise_std,
    coherence_strength,
    num_trials,
    rng,
):
    rows = []

    for trial in range(num_trials):
        A = coherent_gaussian_matrix(
            m,
            n,
            coherence_strength=coherence_strength,
            rng=rng,
        )

        x, support = make_compressible_signal(
            n,
            k,
            tail_scale=0.05,
            rng=rng,
        )

        y = make_measurements(A, x, noise_std, rng)

        # Base OMP
        x_omp, s_omp = omp(A, y, k)

        # Learned candidates
        features = build_operator_features(A, y)
        s_learned = predict_topk_support(model, features, k, device=device)

        # Correlation candidates
        proxy = A.T @ y
        s_corr = np.argsort(np.abs(proxy))[-k:]

        # Random candidates, excluding OMP support where possible
        all_idx = np.arange(n)
        non_omp = np.setdiff1d(all_idx, s_omp)
        if len(non_omp) >= k:
            s_random = rng.choice(non_omp, size=k, replace=False)
        else:
            s_random = rng.choice(all_idx, size=k, replace=False)

        # Ablations
        x_random_union, s_random_union = union_prune_refit(
            A, y, s_omp, s_random, k, n
        )

        x_corr_union, s_corr_union = union_prune_refit(
            A, y, s_omp, s_corr, k, n
        )

        x_learned_union, s_learned_union = union_prune_refit(
            A, y, s_omp, s_learned, k, n
        )

        methods = {
            "OMP": (x_omp, s_omp),
            "OMP_RandomUnion": (x_random_union, s_random_union),
            "OMP_CorrelationUnion": (x_corr_union, s_corr_union),
            "OMP_LearnedUnion": (x_learned_union, s_learned_union),
        }

        for method, (x_hat, pred_support) in methods.items():
            sm = support_metrics(support, pred_support)
            rows.append({
                "trial": trial,
                "method": method,
                "matrix_type": "coherent",
                "signal_type": "compressible",
                "coherence_strength": coherence_strength,
                "n": n,
                "m": m,
                "k": k,
                "m_over_n": m / n,
                "k_over_m": k / m,
                "noise_std": noise_std,
                "nrmse": nrmse(x, x_hat),
                **sm,
            })

    return rows


def main():
    set_seed(42)
    rng = np.random.default_rng(42)

    # Hard high-coherence compressible setting.
    n = 256
    m = 128
    k = 58
    noise_std = 0.01
    coherence_strength = 0.50

    train_instances = 1000
    eval_trials = 300

    print("Training learned support model...")
    X_train, y_train = make_dataset(
        num_instances=train_instances,
        n=n,
        m=m,
        k=k,
        noise_std=noise_std,
        coherence_strength=coherence_strength,
        rng=rng,
    )

    model, device = train_model(
        X_train,
        y_train,
        input_dim=X_train.shape[1],
        epochs=20,
    )

    print("Running ablation evaluation...")
    rows = evaluate_ablation(
        model=model,
        device=device,
        n=n,
        m=m,
        k=k,
        noise_std=noise_std,
        coherence_strength=coherence_strength,
        num_trials=eval_trials,
        rng=rng,
    )

    df = pd.DataFrame(rows)
    out_path = ROOT / "results" / "union_ablation_results.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved ablation results to {out_path}")
    print()
    print(df.groupby("method")[["nrmse", "precision", "recall", "f1", "exact_support"]].mean().round(4))


if __name__ == "__main__":
    main()
