# TODO: update / fix as needed

import os
from pathlib import Path
from ultralytics import YOLO
import torch

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

# --- Configuration ---
DATA_YAML = Path("../DS6050_ML3_Final_Proj/data/raw/data.yaml")

if not DATA_YAML.exists():
    raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")

# For tuning, we may want to use a smaller model (like 'n' or 's') 
# to save compute, assuming the hyperparameters will roughly transfer to 'l'.
# If we have massive GPU power, we can tune directly on 'yolov8l.pt'.
model = YOLO("yolo26l.pt") 

# --- Execute the Evolution ---
print("Starting Hyperparameter Tuning...")

# The .tune() method wraps the standard .train() loop but runs it multiple times.
model.tune(
    data = str(DATA_YAML),
    # TODO: Decide on tuning boundaries
    epochs = 10,        # How long each mutant model lives (keep this relatively short, e.g., 10-15)
    # note that short epochs may give fast but noisy signal so may want to increase the number of iterations to compensate
    iterations = 30,    # How many generations/mutations to try (at least 30 is recommended)
    optimizer = 'MuSGD',
    device = device_arg,
    # Flags to save compute during the tuning sweep
    plots = False,      
    save = False,      
    val = False
)

# --- The Output ---
# Ultralytics will automatically save a file called `best_hyperparameters.yaml` in runs/detect/tune directory.