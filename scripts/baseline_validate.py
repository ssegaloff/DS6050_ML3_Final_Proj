'''
baseline_validate.py

Evaluates the off-the-shelf pretrained yolo26l.pt on our held-out test set,
with no fine-tuning. Provides a performance floor to compare trained models against.

Usage:
    python baseline_validate.py

Results are saved to:
    runs/detect/baseline_yolo26l/validation/
        test_metrics.txt   — human-readable summary
        test_metrics.csv   — machine-readable summary
'''

import os
import csv
import torch
from pathlib import Path
from ultralytics import YOLO

# Hardware-Agnostic Device Selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    NUM_WORKERS = min(12, max(1, os.cpu_count() - 2))
    IS_CUDA = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    NUM_WORKERS = 0 # Multiprocessing on Mac MPS can hang; set to 0 for safety
    IS_CUDA = False
else:
    device = torch.device("cpu")
    NUM_WORKERS = 0
    IS_CUDA = False

print(f"Device: {device}, Number of workers: {NUM_WORKERS}, CUDA: {IS_CUDA}")

if IS_CUDA:
    device_arg = 0          # first GPU
elif str(device) == "mps":
    device_arg = "mps"
else:
    device_arg = "cpu"

# --- configure variables ---
MODEL_PATH = Path("yolo26l.pt")  # pretrained weights, no fine-tuning
DATA_YAML  = Path("../DS6050_ML3_Final_Proj/data/raw/data.yaml")
SPLIT      = "test"   # evaluate on the held-out test set, not the validation set
BATCH_SIZE = 16       # doesn't affect results, only speed/memory usage during validation

# verify files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not DATA_YAML.exists():
    raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")

# --- initialize model ---
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(str(MODEL_PATH))

# --- execute validation ---
print(f"Running validation on '{SPLIT}' split...")

metrics = model.val(
    data      = str(DATA_YAML),
    split     = SPLIT,
    device    = device_arg,
    workers   = NUM_WORKERS,
    batch     = BATCH_SIZE,
    save_json = True,
    plots     = True,
    # Force an absolute path to bypass the framework's global settings
    project  = str(Path("runs/detect/baseline_yolo26l").resolve()),
    name     = "validation",
    conf      = 0.15,  # match predict.py operating threshold
    # TODO: add iou= to adjust NMS overlap threshold if needed
)

SAVE_DIR = Path(metrics.save_dir)
print(f"Results will be saved to: {SAVE_DIR}\n")

# only zip over classes that have actual results (pretrained model has 80 COCO
# classes but metrics.box.ap50 only has entries for our 4 dataset classes)
result_classes = list(metrics.names.values())[:len(metrics.box.ap50)]

# --- build results summary ---
summary_lines = [
    f"Model:     {MODEL_PATH}  (pretrained, no fine-tuning)",
    f"Data:      {DATA_YAML}",
    f"Split:     {SPLIT}",
    "",
    "--- Evaluation Results ---",
    f"mAP50:          {metrics.box.map50:.4f}",
    f"mAP50-95:       {metrics.box.map:.4f}",
    f"Precision:      {metrics.box.mp:.4f}",
    f"Recall:         {metrics.box.mr:.4f}",
    f"F1:             {metrics.box.f1.mean():.4f}",
    "",
    "--- Per-Class Results ---",
]
for i, name in enumerate(result_classes):
    summary_lines.append(
        f"  {name:<20} AP50: {metrics.box.ap50[i]:.4f}  AP50-95: {metrics.box.ap[i]:.4f}"
    )

# --- print to console ---
print("\n" + "\n".join(summary_lines))

# --- save human-readable .txt ---
txt_path = SAVE_DIR / "test_metrics.txt"
txt_path.write_text("\n".join(summary_lines))
print(f"\nSaved: {txt_path}")

# --- save machine-readable .csv ---
csv_path = SAVE_DIR / "test_metrics.csv"
csv_rows = [
    {"metric": "mAP50",     "value": f"{metrics.box.map50:.4f}"},
    {"metric": "mAP50-95",  "value": f"{metrics.box.map:.4f}"},
    {"metric": "Precision", "value": f"{metrics.box.mp:.4f}"},
    {"metric": "Recall",    "value": f"{metrics.box.mr:.4f}"},
    {"metric": "F1",        "value": f"{metrics.box.f1.mean():.4f}"},
]
for i, name in enumerate(result_classes):
    csv_rows.append({"metric": f"{name}_AP50",    "value": f"{metrics.box.ap50[i]:.4f}"})
    csv_rows.append({"metric": f"{name}_AP50-95", "value": f"{metrics.box.ap[i]:.4f}"})

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "value"])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"Saved: {csv_path}")