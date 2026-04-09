'''
This script evaluates the performance of a trained model on the held-out test set.
'''

import os
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
MODEL_PATH = Path("runs/detect/shark_v1/weights/best.pt") # The model to evaluate
DATA_YAML = Path("../DS6050_ML3_Final_Proj/data/raw/data.yaml") 
SPLIT = "test"  # evaluate on the held-out test set, not the validation set
BATCH_SIZE = 16

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
metrics = model.val(
    data = str(DATA_YAML), # tell ultralytics where to find the test images and ground truth labels
    split = SPLIT,
    device = device_arg,
    workers = NUM_WORKERS,
    batch = BATCH_SIZE
    # TODO: Do we want/need to add conf and iou
)

# --- output results ---
# The metrics object contains your precise numbers. 
print("\n--- Evaluation Results ---")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
# TODO: Extract and print Precision and Recall as well.