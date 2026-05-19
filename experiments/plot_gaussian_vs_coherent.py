import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = {
    "Gaussian": ROOT / "results" / "learned_support_gaussian_k55.csv",
    "Coherent": ROOT / "results" / "learned_support_coherent_k55.csv",
}

dfs = []
for setting, path in files.items():
    df = pd.read_csv(path)
    df["setting"] = setting
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

summary = (
    df.groupby(["setting", "method"])[["nrmse", "f1"]]
    .mean()
    .reset_index()
)

print(summary)

figures_dir = ROOT / "figures"
figures_dir.mkdir(exist_ok=True)

methods = ["CorrelationTopK", "LearnedSupportMLP", "CoSaMP", "OMP"]
settings = ["Gaussian", "Coherent"]

x = np.arange(len(methods))
width = 0.35

# F1 plot
plt.figure(figsize=(8, 4))
for j, setting in enumerate(settings):
    vals = []
    for method in methods:
        row = summary[(summary["setting"] == setting) & (summary["method"] == method)]
        vals.append(float(row["f1"].iloc[0]))
    plt.bar(x + (j - 0.5) * width, vals, width, label=setting)

plt.ylabel("Support F1")
plt.title("Support Recovery: Gaussian vs Coherent, n=256, m=128, k=55")
plt.xticks(x, methods, rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "gaussian_vs_coherent_support_f1.png", dpi=200)
plt.close()

# NRMSE plot
plt.figure(figsize=(8, 4))
for j, setting in enumerate(settings):
    vals = []
    for method in methods:
        row = summary[(summary["setting"] == setting) & (summary["method"] == method)]
        vals.append(float(row["nrmse"].iloc[0]))
    plt.bar(x + (j - 0.5) * width, vals, width, label=setting)

plt.ylabel("NRMSE")
plt.title("Recovery Error: Gaussian vs Coherent, n=256, m=128, k=55")
plt.xticks(x, methods, rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "gaussian_vs_coherent_nrmse.png", dpi=200)
plt.close()

print(f"Saved combined figures to {figures_dir}")
