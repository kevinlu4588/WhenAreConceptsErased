import pandas as pd

# Load the CSV file
df = pd.read_csv("/path/to/your/results/rece_inter_class.csv")  # Replace with your actual filename

# Calculate overall accuracy and average scores
total = len(df)
top1_accuracy = df["in_top1"].mean() * 100  # % of True
top5_accuracy = df["in_top5"].mean() * 100

avg_top1_score = df["top1_score"].mean()
avg_top5_score = df["top5_score"].mean()

# Round for cleaner output
top1_accuracy = round(top1_accuracy, 2)
top5_accuracy = round(top5_accuracy, 2)
avg_top1_score = round(avg_top1_score, 4)
avg_top5_score = round(avg_top5_score, 4)

# Print results
print("=== Overall Classification Accuracy ===")
print(f"Top-1 Accuracy: {top1_accuracy}%")
print(f"Avg Top-1 Score: {avg_top1_score}")
print(f"Top-5 Accuracy: {top5_accuracy}%")
print(f"Avg Top-5 Score: {avg_top5_score}")
