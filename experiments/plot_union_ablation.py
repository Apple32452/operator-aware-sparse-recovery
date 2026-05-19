from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(".")
df = pd.read_csv(ROOT / "results" / "union_ablation_results.csv")

figures_dir = ROOT / "figures"
figures_dir.mkdir(exist_ok=True)

summary = (
    df.groupby("method")[["nrmse", "f1"]]
    .agg(["mean", "sem"])
    .reset_index()
)

methods = [
    "OMP",
    "OMP_RandomUnion",
    "OMP_CorrelationUnion",
    "OMP_LearnedUnion",
]

def get_stat(method, metric, stat):
    row = summary[summary["method"] == method]
    return float(row[(metric, stat)].iloc[0])

x = np.arange(len(methods))

plt.figure(figsize=(7, 4))
vals = [get_stat(method, "f1", "mean") for method in methods]
errs = [get_stat(method, "f1", "sem") for method in methods]
plt.bar(x, vals, yerr=errs, capsize=4)
plt.ylabel("Support F1")
plt.title("Union Ablation: Learned Candidates Improve OMP")
plt.xticks(x, methods, rotation=20, ha="right")
plt.tight_layout()
plt.savefig(figures_dir / "union_ablation_f1.png", dpi=250)
plt.close()

plt.figure(figsize=(7, 4))
vals = [get_stat(method, "nrmse", "mean") for method in methods]
errs = [get_stat(method, "nrmse", "sem") for method in methods]
plt.bar(x, vals, yerr=errs, capsize=4)
plt.ylabel("NRMSE")
plt.title("Union Ablation: Learned Candidates Reduce Error")
plt.xticks(x, methods, rotation=20, ha="right")
plt.tight_layout()
plt.savefig(figures_dir / "union_ablation_nrmse.png", dpi=250)
plt.close()

print(summary)
print(f"Saved ablation figures to {figures_dir}")
