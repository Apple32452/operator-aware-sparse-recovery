import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv(ROOT / "results" / "compressible_phase_transition_results.csv")
figures_dir = ROOT / "figures"
figures_dir.mkdir(exist_ok=True)

summary = (
    df.groupby(["matrix_type", "m_over_n", "k_over_m", "method"])[["nrmse", "f1"]]
    .mean()
    .reset_index()
)

# Compare OMP_LearnedUnion against OMP.
omp = summary[summary["method"] == "OMP"].copy()
union = summary[summary["method"] == "OMP_LearnedUnion"].copy()

merged = omp.merge(
    union,
    on=["matrix_type", "m_over_n", "k_over_m"],
    suffixes=("_omp", "_union"),
)

# Positive gain means OMP_LearnedUnion is better.
merged["nrmse_gain"] = merged["nrmse_omp"] - merged["nrmse_union"]
merged["f1_gain"] = merged["f1_union"] - merged["f1_omp"]

# Clean plotting labels.
merged["m_over_n_label"] = merged["m_over_n"].round(3).astype(str)


def clean_k_label(v):
    if v < 0.30:
        return "0.25"
    elif v < 0.40:
        return "0.35"
    return "0.45"


merged["k_over_m_label"] = merged["k_over_m"].apply(clean_k_label)

print(
    merged[
        [
            "matrix_type",
            "m_over_n",
            "k_over_m",
            "nrmse_gain",
            "f1_gain",
        ]
    ]
)

m_order = ["0.375", "0.5", "0.625"]
k_order = ["0.25", "0.35", "0.45"]


def plot_gain_heatmap(sub, metric, title, colorbar_label, filename):
    pivot = sub.pivot(
        index="k_over_m_label",
        columns="m_over_n_label",
        values=metric,
    )

    pivot = pivot.reindex(index=k_order)
    pivot = pivot.reindex(columns=m_order)

    values = pivot.values.astype(float)

    # Use symmetric scale around zero so negative gains are visible.
    vmax = np.nanmax(np.abs(values))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1e-6

    plt.figure(figsize=(6, 4))
    plt.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(label=colorbar_label)
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel("m/n")
    plt.ylabel("k/m")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=250)
    plt.close()


for matrix_type in ["gaussian", "coherent"]:
    sub = merged[merged["matrix_type"] == matrix_type]

    plot_gain_heatmap(
        sub=sub,
        metric="f1_gain",
        title=f"Support F1 Gain Heatmap ({matrix_type})",
        colorbar_label="F1 Gain: OMP_LearnedUnion - OMP",
        filename=f"compressible_phase_transition_f1_gain_{matrix_type}.png",
    )

    plot_gain_heatmap(
        sub=sub,
        metric="nrmse_gain",
        title=f"NRMSE Gain Heatmap ({matrix_type})",
        colorbar_label="NRMSE Reduction: OMP - OMP_LearnedUnion",
        filename=f"compressible_phase_transition_nrmse_gain_{matrix_type}.png",
    )

print(f"Saved phase-transition heatmaps to {figures_dir}")
