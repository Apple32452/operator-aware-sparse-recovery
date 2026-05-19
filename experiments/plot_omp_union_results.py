import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = {
    "Gaussian": ROOT / "results" / "omp_union_gaussian_k55_200trials.csv",
    "Coherent": ROOT / "results" / "omp_union_coherent_k55_200trials.csv",
}

dfs = []
for setting, path in files.items():
    df = pd.read_csv(path)
    df["setting"] = setting
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

summary = (
    df.groupby(["setting", "method"])[["nrmse", "f1"]]
    .agg(["mean", "sem"])
    .reset_index()
)

print(summary)

figures_dir = ROOT / "figures"
figures_dir.mkdir(exist_ok=True)

methods = [
    "CorrelationTopK",
    "LearnedSupportMLP",
    "CoSaMP",
    "OMP",
    "OMP_LearnedUnion",
]
settings = ["Gaussian", "Coherent"]

x = np.arange(len(methods))
width = 0.35

def get_stat(setting, method, metric, stat):
    row = summary[(summary["setting"] == setting) & (summary["method"] == method)]
    return float(row[(metric, stat)].iloc[0])

# Support F1 with error bars
plt.figure(figsize=(9, 4))
for j, setting in enumerate(settings):
    vals = [get_stat(setting, method, "f1", "mean") for method in methods]
    errs = [get_stat(setting, method, "f1", "sem") for method in methods]
    plt.bar(
        x + (j - 0.5) * width,
        vals,
        width,
        yerr=errs,
        capsize=4,
        label=setting,
    )

plt.ylabel("Support F1")
plt.title("OMP + Learned Union Improves Support Recovery, n=256, m=128, k=55")
plt.xticks(x, methods, rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "omp_union_support_f1_errorbars.png", dpi=250)
plt.close()

# NRMSE with error bars
plt.figure(figsize=(9, 4))
for j, setting in enumerate(settings):
    vals = [get_stat(setting, method, "nrmse", "mean") for method in methods]
    errs = [get_stat(setting, method, "nrmse", "sem") for method in methods]
    plt.bar(
        x + (j - 0.5) * width,
        vals,
        width,
        yerr=errs,
        capsize=4,
        label=setting,
    )

plt.ylabel("NRMSE")
plt.title("OMP + Learned Union Reduces Recovery Error, n=256, m=128, k=55")
plt.xticks(x, methods, rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "omp_union_nrmse_errorbars.png", dpi=250)
plt.close()

print(f"Saved error-bar figures to {figures_dir}")