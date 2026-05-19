import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.algorithms.iht import iht
from src.data.make_sparse_signal import make_sparse_signal
from src.data.make_sensing_matrix import gaussian_matrix, coherent_gaussian_matrix, make_measurements
from src.learned.feature_builder import build_operator_features
from src.learned.support_mlp import SupportMLP, predict_topk_support
from src.algorithms.omp import omp
from src.algorithms.cosamp import cosamp
from src.algorithms.subspace_pursuit import subspace_pursuit
from src.evaluation.metrics import nrmse, support_metrics
from src.utils.seed import set_seed


def make_dataset(num_instances, n, m, k, noise_std, matrix_type, rng):
    X_list = []
    y_list = []

    for _ in range(num_instances):
        if matrix_type == "gaussian":
            A = gaussian_matrix(m, n, rng)
        elif matrix_type == "coherent":
            A = coherent_gaussian_matrix(m, n, coherence_strength=0.25, rng=rng)
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")

        x, support = make_sparse_signal(n, k, rng)
        meas = make_measurements(A, x, noise_std, rng)

        features = build_operator_features(A, meas)
        labels = np.zeros(n, dtype=np.float32)
        labels[support] = 1.0

        X_list.append(features)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X.astype(np.float32), y.astype(np.float32)


def train_model(X_train, y_train, input_dim, epochs=10, batch_size=512, lr=1e-3):
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

    model.train()
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

        print(f"Epoch {epoch + 1:02d} | loss = {total_loss / len(ds):.6f}")

    return model, device


def least_squares_on_support(A, y, support, n):
    x_hat = np.zeros(n)
    As = A[:, support]
    coef, *_ = np.linalg.lstsq(As, y, rcond=None)
    x_hat[support] = coef
    return x_hat


def evaluate_model(model, device, num_trials, n, m, k, noise_std, matrix_type, rng):
    rows = []

    for trial in range(num_trials):
        if matrix_type == "gaussian":
            A = gaussian_matrix(m, n, rng)
        else:
            A = coherent_gaussian_matrix(m, n, coherence_strength=0.25, rng=rng)

        x, support = make_sparse_signal(n, k, rng)
        y = make_measurements(A, x, noise_std, rng)

        features = build_operator_features(A, y)
        learned_support = predict_topk_support(model, features, k, device=device)
        x_learned = least_squares_on_support(A, y, learned_support, n)

        # Simple correlation top-k baseline.
        proxy = A.T @ y
        s_corr = np.argsort(np.abs(proxy))[-k:]
        x_corr = least_squares_on_support(A, y, s_corr, n)

        # Classical greedy baselines.
        x_omp, s_omp = omp(A, y, k)
        x_cosamp, s_cosamp = cosamp(A, y, k)
        x_sp, s_sp = subspace_pursuit(A, y, k)
        x_iht, s_iht = iht(A, y, k)

        # New method: OMP + learned union.
        # Step 1: combine OMP support and learned support.
        candidate_support = np.union1d(s_omp, learned_support)

        # Step 2: least-squares fit on the enlarged candidate set.
        x_candidate = least_squares_on_support(A, y, candidate_support, n)

        # Step 3: prune back to k coordinates using coefficient magnitude.
        s_union = np.argsort(np.abs(x_candidate))[-k:]

        # Step 4: refit least squares on the final selected support.
        x_union = least_squares_on_support(A, y, s_union, n)

        methods = {
            "CorrelationTopK": (x_corr, s_corr),
            "LearnedSupportMLP": (x_learned, learned_support),
            "IHT": (x_iht, s_iht),
            "CoSaMP": (x_cosamp, s_cosamp),
            "SubspacePursuit": (x_sp, s_sp),
            "OMP": (x_omp, s_omp),
            "OMP_LearnedUnion": (x_union, s_union),
        }

        for name, (x_hat, pred_support) in methods.items():
            sm = support_metrics(support, pred_support)
            rows.append({
                "trial": trial,
                "method": name,
                "n": n,
                "m": m,
                "k": k,
                "noise_std": noise_std,
                "matrix_type": matrix_type,
                "nrmse": nrmse(x, x_hat),
                **sm,
            })

    return rows


def main():
    set_seed(42)
    rng = np.random.default_rng(42)

    n = 256
    m = 128
    k = 55
    noise_std = 0.01
    matrix_type = "coherent"

    print("Building training data...")
    X_train, y_train = make_dataset(
        num_instances=3000,
        n=n,
        m=m,
        k=k,
        noise_std=noise_std,
        matrix_type=matrix_type,
        rng=rng,
    )

    print("Training learned support model...")
    model, device = train_model(
        X_train,
        y_train,
        input_dim=X_train.shape[1],
        epochs=30,
    )

    print("Evaluating...")
    rows = evaluate_model(
        model=model,
        device=device,
        num_trials=200,
        n=n,
        m=m,
        k=k,
        noise_std=noise_std,
        matrix_type=matrix_type,
        rng=rng,
    )

    df = pd.DataFrame(rows)
    out_path = ROOT / "results" / "learned_support_results.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved results to {out_path}")
    print()
    print(df.groupby("method")[["nrmse", "precision", "recall", "f1", "exact_support"]].mean().round(4))


if __name__ == "__main__":
    main()
