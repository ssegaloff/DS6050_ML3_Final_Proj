'''
plot_training_curves_multi.py

Plots training and validation curves for multiple YOLO runs overlaid on
the same axes for direct comparison. Raw lines are omitted to keep the
chart readable — use plot_training_curves.py for single-run inspection.

Usage:
    python plot_training_curves_multi.py
    > Enter run names, comma-separated: sharks_v5_freeze23, sharks_v5_freeze10

Results are saved to:
    runs/detect/training_curves_multi.png
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# --- configure ---
raw = input("Enter run names, comma-separated (e.g. sharks_v5_freeze23, sharks_v5_freeze10): ").strip()
RUN_NAMES = [r.strip() for r in raw.split(',') if r.strip()]

OUT_PATH = Path("runs/detect/training_curves_multi.png")

if not RUN_NAMES:
    raise ValueError("No run names provided.")

# one colour per run — train is solid, val is dashed, same color
# add more entries here if comparing more than 4 runs
PALETTES = [
    '#378ADD',  # blue
    '#D85A30',  # red-orange
    '#1D9E75',  # green
    '#9B6DD6',  # purple
]

if len(RUN_NAMES) > len(PALETTES):
    raise ValueError(f"Max {len(PALETTES)} runs supported. Add more color entries to PALETTES to extend.")

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
    """Exponential moving average. alpha=0.4 matches weight=0.6 in the manual loop."""
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
        train_col = f'train/{metric}'
        val_col   = f'val/{metric}'

        ax.plot(epochs, smooth(df[train_col]), color=color, linewidth=2,                 label=f'{run_name} train')
        ax.plot(epochs, smooth(df[val_col]),   color=color, linewidth=2, linestyle='--', label=f'{run_name} val')

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.spines[['top', 'right']].set_visible(False)

    # 4th panel: mAP50 (val only — no train equivalent)
    ax = axes[3]
    map_col = 'metrics/mAP50(B)'
    ax.plot(epochs, smooth(df[map_col]), color=color, linewidth=2, label=f'{run_name} val mAP50')

    ax.set_title('Val mAP50', fontsize=11)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.spines[['top', 'right']].set_visible(False)

# draw legends after all runs are plotted so entries group naturally
for ax in axes:
    ax.legend(fontsize=7, frameon=False)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Training curves saved to: {OUT_PATH}")