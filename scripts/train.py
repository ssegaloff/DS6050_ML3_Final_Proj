# TODO: update for shared configs if we want

import os
import torch
from datetime import datetime
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
# TODO: Confirm these
DATA_ROOT = Path("../DS6050_ML3_Final_Proj/data/raw")
YAML_PATH = DATA_ROOT / "data.yaml"
SEED = 26
MODEL_SIZE = "l"        # n, s, m, l, x
EPOCHS = 100            # use patience for early stopping
IMG_SIZE = 640          
BATCH_SIZE = 16         # can increase depending on GPU
FREEZE = 0             # full fine-tuning; TODO: Freezing or no?
default = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = input("Enter run name (e.g. shark_v2_unfrozen): ").strip() or default
OPTIMIZER = "MuSGD" # SGD with Muon-style orthagonalized updates

# --- tuned hyperparameters ---
# These values were selected by hyperparameter_search.py on YYYY-MM-DD.
# To update: run `python hyperparameter_search.py --recommend <csv_path>`
# and paste the recommended values below.
LR0 = # TODO
LRF = # TODO
MOMENTUM = # TODO
WEIGHT_DECAY = # TODO
DEGREES = # TODO
MOSAIC = # TODO
FLIPUD = # TODO # data augmentation param: vertical flip (up-down)
FLIPLR = # TODO # data augmentation param: horizontal flip (left-right)
# TODO: OTHERS?

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
    seed = SEED,
    epochs = EPOCHS,
    imgsz = IMG_SIZE,
    batch = BATCH_SIZE,
    optimizer = OPTIMIZER,
    lr0 = LR0,
    lrf = LRF, # learning rate final
    momentum = MOMENTUM, # momentum for MuSGD (higher means past gradints have more influence on current step; helps smooth noisy gradients but may overshoot)
    weight_decay = WEIGHT_DECAY, #      
    freeze = FREEZE,        
    patience = 20,          # TODO: choose this (I felt 10 was too harsh) # early stopping if validation loss plateaus
    augment = True,         # since we did not augment during roboflow export, may want to do
    degrees = DEGREES,      # randomly rotates training images by up to this may degrees
    mosaic = MOSAIC,        # YOLO specific augmentation that tiles 4 diff training images into one composite (force model to detect at diff scales and cluttered contexts)
    flipud = FLIPUD,        
    fliplr = FLIPLR,
    name = RUN_NAME,
    exist_ok = True,        # overwrite existing runs (won't crash if run name already exists)
    device = device_arg,        # explicitly pass GPU/CPU
    workers = NUM_WORKERS,  # from hardware acceleration above
    verbose = True,         # print training progress to console
    cache = False,         # # disk is fast enough; saves 136GB
)

print(f"Best weights saved to: {model.trainer.best}")