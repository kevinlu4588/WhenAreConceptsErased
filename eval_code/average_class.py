import pandas as pd

# Load the CSV file
df = pd.read_csv("/path/to/your/dataframes/tv.csv")

# Columns to compute
agg_dict = {
    "in_top1": ["mean", "std"],
    "top1_score": ["mean", "std"],
    "in_top5": ["mean", "std"],
    "top5_score": ["mean", "std"]
}

# === 1. Per-Concept Summary (Mean & Std) ===
concept_summary = df.groupby("concept").agg(agg_dict)
concept_summary.columns = ["_".join(col) for col in concept_summary.columns]  # flatten MultiIndex
concept_summary = concept_summary.reset_index()

# Convert to percentage + round
concept_summary["in_top1_mean"] = (concept_summary["in_top1_mean"] * 100).round(2)
concept_summary["in_top1_std"] = (concept_summary["in_top1_std"] * 100).round(2)
concept_summary["in_top5_mean"] = (concept_summary["in_top5_mean"] * 100).round(2)
concept_summary["in_top5_std"] = (concept_summary["in_top5_std"] * 100).round(2)

concept_summary["top1_score_mean"] = concept_summary["top1_score_mean"].round(2)
concept_summary["top1_score_std"] = concept_summary["top1_score_std"].round(2)
concept_summary["top5_score_mean"] = concept_summary["top5_score_mean"].round(2)
concept_summary["top5_score_std"] = concept_summary["top5_score_std"].round(2)

print("=== Per-Concept Classification Metrics (Mean ± Std) ===")
print(concept_summary)

# === 2. Per-Eval Summary (Mean & Std) ===
eval_summary = df.groupby("eval").agg(agg_dict)
eval_summary.columns = ["_".join(col) for col in eval_summary.columns]
eval_summary = eval_summary.reset_index()

# Convert to percentage + round
eval_summary["in_top1_mean"] = (eval_summary["in_top1_mean"] * 100).round(2)
eval_summary["in_top1_std"] = (eval_summary["in_top1_std"] * 100).round(2)
eval_summary["in_top5_mean"] = (eval_summary["in_top5_mean"] * 100).round(2)
eval_summary["in_top5_std"] = (eval_summary["in_top5_std"] * 100).round(2)

eval_summary["top1_score_mean"] = eval_summary["top1_score_mean"].round(2)
eval_summary["top1_score_std"] = eval_summary["top1_score_std"].round(2)
eval_summary["top5_score_mean"] = eval_summary["top5_score_mean"].round(2)
eval_summary["top5_score_std"] = eval_summary["top5_score_std"].round(2)

print("\n=== Per-Eval Classification Metrics (Mean ± Std) ===")
print(eval_summary)

# === 3. Overall Summary (Mean & Std) ===
overall_summary = df.agg({
    "in_top1": ["mean", "std"],
    "top1_score": ["mean", "std"],
    "in_top5": ["mean", "std"],
    "top5_score": ["mean", "std"]
})

overall_summary.loc["mean", ["in_top1", "in_top5"]] *= 100
overall_summary.loc["std", ["in_top1", "in_top5"]] *= 100
overall_summary = overall_summary.round(2)

print("\n=== Overall Classification Metrics (Mean ± Std) ===")
print(f"Top-1 Accuracy: {overall_summary.loc['mean', 'in_top1']}% ± {overall_summary.loc['std', 'in_top1']}%")
print(f"Avg Top-1 Score: {overall_summary.loc['mean', 'top1_score']} ± {overall_summary.loc['std', 'top1_score']}")
print(f"Top-5 Accuracy: {overall_summary.loc['mean', 'in_top5']}% ± {overall_summary.loc['std', 'in_top5']}%")
print(f"Avg Top-5 Score: {overall_summary.loc['mean', 'top5_score']} ± {overall_summary.loc['std', 'top5_score']}")
