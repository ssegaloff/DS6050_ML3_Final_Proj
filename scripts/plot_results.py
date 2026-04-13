"""
plot_results.py

Reads test_metrics.csv files from validate.py and plots a comparison.

Usage:
    python plot_results.py

Output:
    results_comparison.png
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# --- configure: add/remove runs ---
RUNS = {
    "baseline": Path("runs/detect/baseline_yolo26l/validation/test_metrics.csv"),
    "freeze23":  Path("runs/detect/sharks_v5_freeze23/validation/test_metrics.csv"),
    "freeze10":  Path("runs/detect/sharks_v5_freeze10/validation/test_metrics.csv"),
}

def load_metrics(csv_path):
    """Load a test_metrics.csv into a {metric: value} dict."""
    metrics = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            metrics[row["metric"]] = float(row["value"])
    return metrics


run_names = list(RUNS.keys())
colors = ["#888780", "#378ADD", "#1D9E75"]  # baseline, freeze23, freeze10

# validate all paths exist
for name, path in RUNS.items():
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found for '{name}': {path}")

assert len(colors) >= len(run_names), "Add a color for each run"

# load all runs
data = {name: load_metrics(path) for name, path in RUNS.items()}

# --- plot 1: overall metrics ---
overall_metrics = ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]
class_metrics   = ["boat_AP50", "human_AP50", "other_AP50", "shark_AP50"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison — Shark Detection", fontsize=14)

x = np.arange(len(overall_metrics)) # gives evenly spaced positions for the groups of bars on the x-axis
width = 0.8 / len(run_names)  # bar width

for i, (name, color) in enumerate(zip(run_names, colors)):
    values = [data[name].get(m, 0) for m in overall_metrics]
    ax1.bar(x + i * width, values, width, label=name, color=color)

ax1.set_title("Overall metrics")
ax1.set_xticks(x + width * (len(data) - 1) / 2) # centers tick labels in middle of each group
ax1.set_xticklabels(overall_metrics, rotation=15)
ax1.set_ylim(0, 1)
ax1.legend()

# --- plot 2: per-class AP50 ---
x2 = np.arange(len(class_metrics)) # gives evenly spaced positions for the groups
class_labels = [m.replace("_AP50", "") for m in class_metrics]

for i, (name, color) in enumerate(zip(run_names, colors)):
    values = [data[name].get(m, 0) for m in class_metrics]
    ax2.bar(x2 + i * width, values, width, label=name, color=color)

ax2.set_title("Per-class AP50")
ax2.set_xticks(x2 + width * (len(data) - 1) / 2) # centers tick labels in middle of each group
ax2.set_xticklabels(class_labels)
ax2.set_ylim(0, 1)
ax2.legend()

plt.tight_layout()
plt.savefig("results_comparison.png", dpi=150)
print("Saved: results_comparison.png")
plt.show()