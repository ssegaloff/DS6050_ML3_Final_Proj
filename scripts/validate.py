'''
validate.py
 
Evaluates a trained YOLO model on the held-out test set and saves results
to a `validation/` subdirectory alongside the model weights.
 
Usage:
    Run the script and enter the run name when prompted:
        python validate.py
        > Enter run name (e.g. sharks_v4_frozen): sharks_v4_frozen
    
    Results are saved to:
        runs/detect/<run_name>/validation/
            test_metrics.txt   — human-readable summary
            test_metrics.csv   — machine-readable summary
            (+ Ultralytics plots: confusion matrix, PR curve, etc.)
'''
# TODO: update for shared configs if desired

import os
import csv
import torch
from pathlib import Path
from ultralytics import YOLO

# Hardware-Agnostic Device Selection
if torch.cuda.is_available():
    device = torch.device("cuda") # Lenovo / WSL2 (NVIDIA GPU)
    
    # Dynamic Worker Allocation Formula
    # Reserve 2 cores for Windows/WSL; cap at 12 to avoid over-allocation (pytorch performance tends to plateau beyond 8 workers)
    NUM_WORKERS = min(12, max(1, os.cpu_count() - 2))
    IS_CUDA = True

elif torch.backends.mps.is_available():
    device = torch.device("mps") # Mac Apple Silicon
    NUM_WORKERS = 0 # Multiprocessing on Mac MPS can hang; set to 0 for safety
    IS_CUDA = False

else:
    device = torch.device("cpu") # Fallback to CPU
    NUM_WORKERS = 0 # Multiprocessing on CPU can be less efficient; set to 0 for simplicity
    IS_CUDA = False

print(f"Device: {device}, Number of workers: {NUM_WORKERS}, CUDA: {IS_CUDA}")

# Convert torch.device to the string Ultralytics expects
if IS_CUDA:
    device_arg = 0          # first GPU
elif str(device) == "mps":
    device_arg = "mps"
else:
    device_arg = "cpu"

# --- configure variables ---
RUN_NAME   = input("Enter run name (e.g. sharks_v4_frozen): ").strip()
MODEL_PATH = Path(f"runs/detect/{RUN_NAME}/weights/best.pt")
DATA_YAML = Path("../DS6050_ML3_Final_Proj/data/raw/data.yaml") 
SPLIT = "test"  # evaluate on the held-out test set, not the validation set
BATCH_SIZE = 16  # doesn't affect results, only speed/memory usage during validation

# Save results to a validation/ folder inside the run directory
# (i.e. alongside weights/, so everything for a run stays together)
SAVE_DIR = MODEL_PATH.parent.parent / "validation"

# verify files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not DATA_YAML.exists():
    raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")

# --- initialize model ---
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(str(MODEL_PATH)) # ultralytics expects a string path rather than a `Path` object

# --- execute validation ---
print(f"Running validation on '{SPLIT}' split...")
print(f"Results will be saved to: {SAVE_DIR}\n")

metrics = model.val(
    data = str(DATA_YAML), # tell ultralytics where to find the test images and ground truth labels
    split = SPLIT,
    device = device_arg,
    workers = NUM_WORKERS,
    batch = BATCH_SIZE,
    save_json = True,   # saves COCO-format results (useful for future analysis)
    plots   = True,     # saves confusion matrix, PR curve, etc.
    project = str(SAVE_DIR.parent),
    name    = SAVE_DIR.name
    # TODO: add conf= and iou= if we want metrics at a specific operating threshold
#       (e.g. match the conf=0.25 used in predict.py). Omitting uses Ultralytics defaults,
#       which is fine for standard benchmarking but won't match real-world predict.py behavior exactly
)

# --- build results summary ---
summary_lines = [
    f"Model:     {MODEL_PATH}",
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
for i, name in enumerate(metrics.names.values()):
    summary_lines.append(
        f"  {name:<20} AP50: {metrics.box.ap50[i]:.4f}  AP50-95: {metrics.box.ap[i]:.4f}"
    )

# --- print to console ---
print("\n" + "\n".join(summary_lines))
 
# --- save human-readable .txt ---
SAVE_DIR.mkdir(parents=True, exist_ok=True)
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
for i, name in enumerate(metrics.names.values()):
    csv_rows.append({"metric": f"{name}_AP50",    "value": f"{metrics.box.ap50[i]:.4f}"})
    csv_rows.append({"metric": f"{name}_AP50-95", "value": f"{metrics.box.ap[i]:.4f}"})
 
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "value"])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"Saved: {csv_path}")