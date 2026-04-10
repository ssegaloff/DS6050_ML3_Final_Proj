'''
predict.py

Runs inference on the test set using a trained YOLO model and saves
annotated images and label files.

Usage:
    Run the script and enter the run name when prompted:
        python predict.py
        > Enter run name (e.g. sharks_v4_frozen): sharks_v4_frozen

    Results are saved to:
        runs/detect/<run_name>/predict/
            images/    — annotated images with bounding boxes
            labels/    — YOLO-format label files
        
'''

# TODO: update for shared configs if desired

import torch
import os
from pathlib import Path
from ultralytics import YOLO


# Hardware-Agnostic Device Selection
if torch.cuda.is_available():
    device = torch.device("cuda") # Lenovo / WSL2 (NVIDIA GPU)
    
    # Dynamic Worker Allocation Formula
    # Reserve 2 cores for Windows/WSL; cap at 16 to avoid over-allocation (pytorch performance tends to plateau beyond 8 workers)
    NUM_WORKERS = min(16, max(1, os.cpu_count() - 2))
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

# --- configure ---
RUN_NAME   = input("Enter run name (e.g. sharks_v4_frozen): ").strip()
MODEL_PATH = Path(f"runs/detect/{RUN_NAME}/weights/best.pt")
SOURCE = Path("../DS6050_ML3_Final_Proj/data/raw/test/images")
CONF_THRESHOLD = 0.15

# verify model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# verify source exists
if not SOURCE.exists():
    raise FileNotFoundError(f"Test images not found at {SOURCE}")

# --- initialize model ---
model = YOLO(MODEL_PATH)

results = model.predict(
    source   = str(SOURCE),
    conf     = CONF_THRESHOLD,
    save     = True,
    save_txt = True,
    project  = str(Path(f"runs/detect/{RUN_NAME}").resolve()),
    name     = f"predict",
    device   = device_arg,
    workers  = NUM_WORKERS,
)


print(f"Results saved to: runs/detect/{RUN_NAME}/predict")