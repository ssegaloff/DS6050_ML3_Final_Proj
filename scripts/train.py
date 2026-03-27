# TODO: COMPLETE THIS SCRIPT AND MAKE UPDATES NEEDED

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

# --- configure variables ---
# TODO: CHANGE / DECIDE ON THESE PARAMETERS (everything is currently a placeholder)
DATA_ROOT = Path("../DS6050_ML3_Final_Proj/data/raw")
YAML_PATH = DATA_ROOT / "data.yaml"
MODEL_SIZE = "l"        # n, s, m, l, x
EPOCHS = 50
IMG_SIZE = 640          
BATCH_SIZE = 16
FREEZE = 10             # TODO: Decide if we are freezing the backbone or fine tuning the whole network; freeze backbone layers for transfer learning
RUN_NAME = "shark_v1"   # name of the run

# verify data exists
if not YAML_PATH.exists():
    raise FileNotFoundError(
        f"data.yaml not found at {YAML_PATH}. Run `python scripts/download_data.py` first."
    )

# --- initialize model ---
# load a pretrained YOLO large model
model = YOLO(f"yolo26{MODEL_SIZE}.pt")

# --- training loop ---
results = model.train(
    data = str(YAML_PATH),
    epochs = EPOCHS,
    imgsz = IMG_SIZE,
    batch = BATCH_SIZE,
    lr0 = 0.001,            # want a lower LR for fine tuning
    freeze = FREEZE,        
    patience = 10,          # early stopping if validation loss plateaus
    augment = True,         # since we did not augment during roboflow export, may want to do
    name = RUN_NAME,
    exist_ok = True,        # overwrite existing runs (won't crash if run name already exists)
    device = device,        # explicitly pass GPU/CPU
    workers = NUM_WORKERS,  # from hardware acceleration above
    cache = True,           # cache images in RAM for faster epoch times
    amp = True              # automatic mixed precision
)

print(f"Best weights saved to: {model.trainer.best}")