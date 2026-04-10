# TODO: update for shared configs

import torch
import os
from datetime import datetime
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
MODEL_PATH = Path("runs/detect/shark_v1/weights/best.pt")
SOURCE = Path("data/raw/test/images")
CONF_THRESHOLD = 0.25
default = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = input("Enter run name (e.g. shark_v2_unfrozen): ").strip() or default

# verify model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# --- initialize model ---
model = YOLO(MODEL_PATH)

results = model.predict(
    source = str(SOURCE),
    conf = CONF_THRESHOLD,
    save=True,                 # saves annotated images with bounding boxes
    save_txt=True,             # saves label files
    name = RUN_NAME,
    device = device_arg
)

print(f"Results saved to: runs/detect/{RUN_NAME}")