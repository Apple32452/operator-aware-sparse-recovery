from pathlib import Path
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]

files = {
    "Gaussian": ROOT / "results" / "omp_union_gaussian_k55_200trials.csv",
    "Coherent": ROOT / "results" / "omp_union_coherent_k55_200trials.csv",
}

for setting, path in files.items():
    df = pd.read_csv(path)

    omp = df[df["method"] == "OMP"].sort_values("trial")
    union = df[df["method"] == "OMP_LearnedUnion"].sort_values("trial")

    merged = omp.merge(
        union,
        on="trial",
        suffixes=("_omp", "_union"),
    )

    nrmse_diff = merged["nrmse_omp"] - merged["nrmse_union"]
    f1_diff = merged["f1_union"] - merged["f1_omp"]

    print("=" * 80)
    print(setting)

    print("\nNRMSE: positive means OMP_LearnedUnion is better")
    print(f"Mean gain: {nrmse_diff.mean():.6f}")
    print(f"Median gain: {nrmse_diff.median():.6f}")
    print(f"Improved trials: {(nrmse_diff > 0).sum()} / {len(nrmse_diff)}")
    print("Paired t-test:", ttest_rel(merged["nrmse_omp"], merged["nrmse_union"]))
    print("Wilcoxon:", wilcoxon(nrmse_diff))

    print("\nF1: positive means OMP_LearnedUnion is better")
    print(f"Mean gain: {f1_diff.mean():.6f}")
    print(f"Median gain: {f1_diff.median():.6f}")
    print(f"Improved trials: {(f1_diff > 0).sum()} / {len(f1_diff)}")
    print("Paired t-test:", ttest_rel(merged["f1_union"], merged["f1_omp"]))
    print("Wilcoxon:", wilcoxon(f1_diff))
