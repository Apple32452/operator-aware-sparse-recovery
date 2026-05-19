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
from src.algorithms.cosamp import cosamp
from src.algorithms.subspace_pursuit import subspace_pursuit
from src.algorithms.iht import iht
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


def analyze_cases():
    set_seed(42)
    rng = np.random.default_rng(42)

    n = 256
    m = 128
    k = 58
    noise_std = 0.01
    coherence_strength = 0.50

    print("Training model for failure-case analysis...")
    X_train, y_train = make_dataset(
        num_instances=1000,
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

    rows = []
    examples = []

    num_trials = 200

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

        true_set = set(map(int, support))

        features = build_operator_features(A, y)
        learned_support = predict_topk_support(model, features, k, device=device)

        x_omp, s_omp = omp(A, y, k)

        candidate_support = np.union1d(s_omp, learned_support)
        x_candidate = least_squares_on_support(A, y, candidate_support, n)
        s_union = np.argsort(np.abs(x_candidate))[-k:]
        x_union = least_squares_on_support(A, y, s_union, n)

        omp_set = set(map(int, s_omp))
        learned_set = set(map(int, learned_support))
        union_set = set(map(int, s_union))

        omp_metrics = support_metrics(support, s_omp)
        union_metrics = support_metrics(support, s_union)

        omp_nrmse = nrmse(x, x_omp)
        union_nrmse = nrmse(x, x_union)

        missed_by_omp = true_set - omp_set
        recovered_by_learned = missed_by_omp & learned_set
        recovered_by_union = missed_by_omp & union_set

        false_omp_removed = (omp_set - true_set) - union_set

        row = {
            "trial": trial,
            "omp_nrmse": omp_nrmse,
            "union_nrmse": union_nrmse,
            "nrmse_gain": omp_nrmse - union_nrmse,
            "omp_f1": omp_metrics["f1"],
            "union_f1": union_metrics["f1"],
            "f1_gain": union_metrics["f1"] - omp_metrics["f1"],
            "num_missed_by_omp": len(missed_by_omp),
            "num_missed_recovered_by_learned": len(recovered_by_learned),
            "num_missed_recovered_by_union": len(recovered_by_union),
            "num_false_omp_removed": len(false_omp_removed),
        }

        rows.append(row)

        if row["f1_gain"] > 0 or row["nrmse_gain"] > 0.02:
            examples.append({
                **row,
                "true_support": sorted(true_set),
                "omp_support": sorted(omp_set),
                "learned_support": sorted(learned_set),
                "union_support": sorted(union_set),
                "missed_by_omp": sorted(missed_by_omp),
                "recovered_by_learned": sorted(recovered_by_learned),
                "recovered_by_union": sorted(recovered_by_union),
                "false_omp_removed": sorted(false_omp_removed),
            })

    df = pd.DataFrame(rows)
    out_path = ROOT / "results" / "failure_case_summary.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved failure-case summary to {out_path}")
    print()
    print("Aggregate explanation statistics")
    print("--------------------------------")
    print(f"Average NRMSE gain: {df['nrmse_gain'].mean():.6f}")
    print(f"Average F1 gain:    {df['f1_gain'].mean():.6f}")
    print(f"Trials with NRMSE improvement: {(df['nrmse_gain'] > 0).sum()} / {len(df)}")
    print(f"Trials with F1 improvement:    {(df['f1_gain'] > 0).sum()} / {len(df)}")
    print(f"Average OMP missed support count: {df['num_missed_by_omp'].mean():.3f}")
    print(
        "Average missed OMP coords recovered by learned support: "
        f"{df['num_missed_recovered_by_learned'].mean():.3f}"
    )
    print(
        "Average missed OMP coords recovered by final union: "
        f"{df['num_missed_recovered_by_union'].mean():.3f}"
    )
    print(
        "Average false OMP coords removed by union pruning: "
        f"{df['num_false_omp_removed'].mean():.3f}"
    )

    examples = sorted(examples, key=lambda r: r["nrmse_gain"], reverse=True)
    ex_path = ROOT / "results" / "failure_case_examples.txt"

    with open(ex_path, "w") as f:
        for i, ex in enumerate(examples[:10]):
            f.write("=" * 80 + "\n")
            f.write(f"Example {i + 1}\n")
            f.write(f"Trial: {ex['trial']}\n")
            f.write(f"OMP NRMSE: {ex['omp_nrmse']:.6f}\n")
            f.write(f"Union NRMSE: {ex['union_nrmse']:.6f}\n")
            f.write(f"NRMSE gain: {ex['nrmse_gain']:.6f}\n")
            f.write(f"OMP F1: {ex['omp_f1']:.6f}\n")
            f.write(f"Union F1: {ex['union_f1']:.6f}\n")
            f.write(f"F1 gain: {ex['f1_gain']:.6f}\n")
            f.write(f"Missed by OMP: {ex['missed_by_omp']}\n")
            f.write(f"Recovered by learned support: {ex['recovered_by_learned']}\n")
            f.write(f"Recovered by final union: {ex['recovered_by_union']}\n")
            f.write(f"False OMP coords removed: {ex['false_omp_removed']}\n")
            f.write("\n")

    print(f"Saved top examples to {ex_path}")


if __name__ == "__main__":
    analyze_cases()
