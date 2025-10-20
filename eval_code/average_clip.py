import pandas as pd

# Load the CSV file
df = pd.read_csv("/path/to/your/results/stereo_dataframes/rece_clip_scores.csv")

# Columns to analyze
score_columns = [
    "ground_truth_score",
    "uda_score",
    "textual_inversion_score",
    "noisy_score",
    "inpainting_score"
]

# === Per Concept + Method: Mean and Std ===
grouped = df.groupby(["method", "concept"])[score_columns]
grouped_mean = grouped.mean().round(2).reset_index()
grouped_std = grouped.std().round(2).reset_index()

# Merge for side-by-side view
grouped_summary = grouped_mean.copy()
for col in score_columns:
    grouped_summary[col + "_std"] = grouped_std[col]

print("=== Per-Concept CLIP Scores (Mean ± Std) ===")
print(grouped_summary)

# === Overall Average and Std ===
overall_avg = df[score_columns].mean()
overall_std = df[score_columns].std()

# Combine into a summary
overall_summary = pd.DataFrame({
    "mean": overall_avg.round(2),
    "std": overall_std.round(2)
})

print("\n=== Overall CLIP Scores Across All Concepts and Methods (Mean ± Std) ===")
print(overall_summary)
