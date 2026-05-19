from pathlib import Path
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(".")
df = pd.read_csv(ROOT / "results" / "union_ablation_results.csv")

baseline = df[df["method"] == "OMP"].sort_values("trial")
learned = df[df["method"] == "OMP_LearnedUnion"].sort_values("trial")
random_union = df[df["method"] == "OMP_RandomUnion"].sort_values("trial")
corr_union = df[df["method"] == "OMP_CorrelationUnion"].sort_values("trial")


def compare(a, b, name_a, name_b):
    merged = a.merge(b, on="trial", suffixes=(f"_{name_a}", f"_{name_b}"))

    nrmse_gain = merged[f"nrmse_{name_a}"] - merged[f"nrmse_{name_b}"]
    f1_gain = merged[f"f1_{name_b}"] - merged[f"f1_{name_a}"]

    print("=" * 80)
    print(f"{name_b} vs {name_a}")

    print("\nNRMSE gain positive means second method is better")
    print(f"Mean gain: {nrmse_gain.mean():.6f}")
    print(f"Median gain: {nrmse_gain.median():.6f}")
    print(f"Improved trials: {(nrmse_gain > 0).sum()} / {len(nrmse_gain)}")
    print("Paired t-test:", ttest_rel(merged[f"nrmse_{name_a}"], merged[f"nrmse_{name_b}"]))
    print("Wilcoxon:", wilcoxon(nrmse_gain))

    print("\nF1 gain positive means second method is better")
    print(f"Mean gain: {f1_gain.mean():.6f}")
    print(f"Median gain: {f1_gain.median():.6f}")
    print(f"Improved trials: {(f1_gain > 0).sum()} / {len(f1_gain)}")
    print("Paired t-test:", ttest_rel(merged[f"f1_{name_b}"], merged[f"f1_{name_a}"]))
    print("Wilcoxon:", wilcoxon(f1_gain))


compare(baseline, random_union, "OMP", "OMP_RandomUnion")
compare(baseline, corr_union, "OMP", "OMP_CorrelationUnion")
compare(baseline, learned, "OMP", "OMP_LearnedUnion")
compare(corr_union, learned, "OMP_CorrelationUnion", "OMP_LearnedUnion")
compare(random_union, learned, "OMP_RandomUnion", "OMP_LearnedUnion")
