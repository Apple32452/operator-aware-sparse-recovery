from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(".")
df = pd.read_csv(ROOT / "results" / "coherence_sweep_results.csv")

figures_dir = ROOT / "figures"
figures_dir.mkdir(exist_ok=True)

summary = (
    df.groupby(["coherence_strength", "method"])[["nrmse", "f1"]]
    .mean()
    .reset_index()
)

omp = summary[summary["method"] == "OMP"].copy()
union = summary[summary["method"] == "OMP_LearnedUnion"].copy()

merged = omp.merge(
    union,
    on="coherence_strength",
    suffixes=("_omp", "_union"),
)

merged["nrmse_gain"] = merged["nrmse_omp"] - merged["nrmse_union"]
merged["f1_gain"] = merged["f1_union"] - merged["f1_omp"]

print("\nOMP_LearnedUnion gain over OMP:")
print(merged[["coherence_strength", "nrmse_gain", "f1_gain"]])

plt.figure(figsize=(6, 4))
plt.plot(merged["coherence_strength"], merged["f1_gain"], marker="o", label="F1 gain")
plt.xlabel("Coherence strength")
plt.ylabel("F1 gain over OMP")
plt.title("OMP_LearnedUnion F1 Gain Increases with Coherence")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "coherence_sweep_f1_gain.png", dpi=250)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(merged["coherence_strength"], merged["nrmse_gain"], marker="o", label="NRMSE gain")
plt.xlabel("Coherence strength")
plt.ylabel("NRMSE reduction over OMP")
plt.title("OMP_LearnedUnion Error Reduction Increases with Coherence")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "coherence_sweep_nrmse_gain.png", dpi=250)
plt.close()

# Full method ranking by coherence
methods = [
    "CorrelationTopK",
    "LearnedSupportMLP",
    "IHT",
    "CoSaMP",
    "SubspacePursuit",
    "OMP",
    "OMP_LearnedUnion",
]

for metric in ["f1", "nrmse"]:
    plt.figure(figsize=(7, 4))
    for method in methods:
        sub = summary[summary["method"] == method].sort_values("coherence_strength")
        plt.plot(
            sub["coherence_strength"],
            sub[metric],
            marker="o",
            label=method,
        )

    plt.xlabel("Coherence strength")
    plt.ylabel(metric.upper() if metric == "f1" else "NRMSE")
    plt.title(f"Method Comparison Across Coherence Strengths ({metric})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / f"coherence_sweep_all_methods_{metric}.png", dpi=250)
    plt.close()

print(f"Saved coherence sweep figures to {figures_dir}")
