'''
plot_training_curves_multi.py

Plots training and validation curves for multiple YOLO runs overlaid on
the same axes for direct comparison. Draws a horizontal baseline reference
line on the mAP50 panel.

Usage:
    python plot_training_curves_multi.py
    > Enter run names, comma-separated: sharks_v5_freeze23, sharks_v6_freeze11

Results are saved to:
    runs/detect/training_curves_multi.png
'''

import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# --- configure ---
BASELINE_CSV = Path("runs/detect/baseline_yolo26l/validation/test_metrics.csv")
OUT_PATH     = Path("runs/detect/training_curves_multi.png")

raw = input("Enter run names, comma-separated (e.g. sharks_v5_freeze23, sharks_v6_freeze11): ").strip()
RUN_NAMES = [r.strip() for r in raw.split(',') if r.strip()]

if not RUN_NAMES:
    raise ValueError("No run names provided.")

PALETTES = [
    '#378ADD',  # blue
    '#D85A30',  # red-orange
    '#1D9E75',  # green
    '#9B6DD6',  # purple
]

if len(RUN_NAMES) > len(PALETTES):
    raise ValueError(f"Max {len(PALETTES)} runs supported. Add more entries to PALETTES to extend.")

# --- load baseline mAP50 ---
baseline_map50 = None
if not BASELINE_CSV.exists():
    print(f"[warn] Baseline CSV not found at {BASELINE_CSV} — skipping reference line.")
else:
    with open(BASELINE_CSV, newline='') as f:
        for row in csv.DictReader(f):
            if row['metric'] == 'mAP50':
                baseline_map50 = float(row['value'])
                break
    print(f"Baseline mAP50 loaded: {baseline_map50:.4f}")

# --- data loading ---
dataframes = {}
for run_name in RUN_NAMES:
    csv_path = Path(f"runs/detect/{run_name}/results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Did you type the run name correctly?")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    dataframes[run_name] = df

def smooth(series):
    """Exponential moving average."""
    return series.ewm(alpha=0.4, adjust=False).mean()

# --- plotting ---
metric_pairs = [
    ('box_loss', 'Bounding box loss'),
    ('cls_loss', 'Classification loss'),
    ('dfl_loss', 'Distribution focal loss'),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Training curves — " + "  vs.  ".join(RUN_NAMES), fontsize=13)

for run_name, color in zip(RUN_NAMES, PALETTES):
    df     = dataframes[run_name]
    epochs = df['epoch']

    for i, (metric, title) in enumerate(metric_pairs):
        ax = axes[i]
        ax.plot(epochs, smooth(df[f'train/{metric}']), color=color, linewidth=2,
                label=f'{run_name} train')
        ax.plot(epochs, smooth(df[f'val/{metric}']),   color=color, linewidth=2,
                linestyle='--', label=f'{run_name} val')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.spines[['top', 'right']].set_visible(False)

    ax = axes[3]
    ax.plot(epochs, smooth(df['metrics/mAP50(B)']), color=color, linewidth=2,
            label=f'{run_name} val mAP50')
    ax.set_title('Val mAP50', fontsize=11)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.spines[['top', 'right']].set_visible(False)

if baseline_map50 is not None:
    axes[3].axhline(
        y=baseline_map50,
        color='#888780',
        linewidth=1.5,
        linestyle=':',
        label=f'baseline (test) {baseline_map50:.3f}'
    )

for ax in axes:
    ax.legend(fontsize=7, frameon=False)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Training curves saved to: {OUT_PATH}")