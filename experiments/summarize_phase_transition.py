from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "results" / "phase_transition_results.csv")

summary = (
    df.groupby(["matrix_type", "m_over_n", "k_over_m", "method"])[["nrmse", "f1"]]
    .mean()
    .reset_index()
)

omp = summary[summary["method"] == "OMP"].copy()
union = summary[summary["method"] == "OMP_LearnedUnion"].copy()

merged = omp.merge(
    union,
    on=["matrix_type", "m_over_n", "k_over_m"],
    suffixes=("_omp", "_union"),
)

merged["nrmse_gain"] = merged["nrmse_omp"] - merged["nrmse_union"]
merged["f1_gain"] = merged["f1_union"] - merged["f1_omp"]

print("\nOverall phase-transition summary")
print("--------------------------------")
print(f"Average NRMSE gain: {merged['nrmse_gain'].mean():.6f}")
print(f"Average F1 gain:    {merged['f1_gain'].mean():.6f}")
print(f"NRMSE improved in:  {(merged['nrmse_gain'] > 0).sum()} / {len(merged)} settings")
print(f"F1 improved in:     {(merged['f1_gain'] > 0).sum()} / {len(merged)} settings")

print("\nBy matrix type")
print("--------------")
print(
    merged.groupby("matrix_type")[["nrmse_gain", "f1_gain"]]
    .agg(["mean", "min", "max"])
    .round(6)
)

out_path = ROOT / "results" / "phase_transition_gain_summary.csv"
merged.to_csv(out_path, index=False)
print(f"\nSaved detailed gain summary to {out_path}")

