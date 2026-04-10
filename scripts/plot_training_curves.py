'''
plot_training_curves.py

Plots training and validation curves for YOLO26l.
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# --- configure ---
RUN_NAME = input("Enter run name (e.g., sharks_v5_freeze23): ").strip()

RUN_DIR  = Path(f"runs/detect/{RUN_NAME}")
CSV_PATH = RUN_DIR / "results.csv"
SAVE_DIR = RUN_DIR / "train_val_curves"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = SAVE_DIR / "training_curves.png"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Did you type the run name correctly?")

# --- data processing ---
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
epochs = df['epoch']

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
fig.suptitle(f"Training curves — {RUN_NAME}", fontsize=13)

TRAIN_COLOR = '#378ADD'
VAL_COLOR   = '#D4537E'
MAP_COLOR   = '#1D9E75'

for i, (metric, title) in enumerate(metric_pairs):
    ax = axes[i]
    train_col = f'train/{metric}'
    val_col   = f'val/{metric}'

    ax.plot(epochs, df[train_col],         color=TRAIN_COLOR, alpha=0.2, linewidth=1, label='train (raw)')
    ax.plot(epochs, smooth(df[train_col]), color=TRAIN_COLOR, alpha=1.0, linewidth=2, label='train (smoothed)')
    ax.plot(epochs, df[val_col],           color=VAL_COLOR,   alpha=0.2, linewidth=1, label='val (raw)')
    ax.plot(epochs, smooth(df[val_col]),   color=VAL_COLOR,   alpha=1.0, linewidth=2, label='val (smoothed)')

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7, frameon=False)

# 4th panel: mAP50 (val only — no train equivalent)
ax = axes[3]
map_col = 'metrics/mAP50(B)'
ax.plot(epochs, df[map_col],         color=MAP_COLOR, alpha=0.2, linewidth=1, label='val mAP50 (raw)')
ax.plot(epochs, smooth(df[map_col]), color=MAP_COLOR, alpha=1.0, linewidth=2, label='val mAP50 (smoothed)')
ax.set_title('Val mAP50', fontsize=11)
ax.set_xlabel('Epoch', fontsize=9)
ax.tick_params(labelsize=8)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=7, frameon=False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Training curves saved to: {OUT_PATH}")