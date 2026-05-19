import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

results_path = ROOT / "results" / "learned_support_results.csv"
df = pd.read_csv(results_path)

summary = df.groupby("method")[["nrmse", "f1"]].mean().reset_index()

print(summary)

figures_dir = ROOT / "figures"
figures_dir.mkdir(exist_ok=True)

plt.figure(figsize=(7, 4))
plt.bar(summary["method"], summary["f1"])
plt.ylabel("Support F1")
plt.title("Support Recovery Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(figures_dir / "support_f1_comparison.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 4))
plt.bar(summary["method"], summary["nrmse"])
plt.ylabel("NRMSE")
plt.title("Sparse Recovery Error Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(figures_dir / "nrmse_comparison.png", dpi=200)
plt.close()

print(f"Saved figures to {figures_dir}")
