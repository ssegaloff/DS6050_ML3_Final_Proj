'''
train.py

Fine-tunes a pretrained YOLO26 model of specified size on the shark detection
dataset using tuned MuSGD hyperparameters and mosaic/augmentation settings
selected by hyperparameter_search.py.

The backbone is frozen for the first phase of training (FREEZE = 23).
To unfreeze C2PSA, neck, and head layers for a second-phase run, set FREEZE = 10 
and drop LR0 by 10x (see commented values in the config section).

Usage:
    python train.py
    > Enter run name (e.g. shark_v2_unfrozen): sharks_v1_frozen

    If no name is entered, the run is timestamped automatically.

Configuration:
    Edit the constants under "configure variables" and "tuned hyperparameters"
    to change model size, batch size, freeze depth, or optimizer settings.
    To refresh hyperparameters after a new sweep, run:
        python hyperparameter_search.py --recommend <csv_path>
    and paste the recommended values into the LR0/LRF/etc. constants below.

Results are saved to:
    runs/detect/<run_name>/
        weights/best.pt        — best checkpoint (use this for validate.py / predict.py)
        weights/last.pt        — final epoch checkpoint
        results.csv            — per-epoch training and validation metrics
        (plots, confusion matrix, etc.)
'''



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

# --- configure variables ---
# TODO: Confirm these
DATA_ROOT = Path("../DS6050_ML3_Final_Proj/data/raw")
YAML_PATH = DATA_ROOT / "data.yaml"
SEED = 26
MODEL_SIZE = "l"        # n, s, m, l, x
EPOCHS = 300            # use patience for early stopping
IMG_SIZE = 640          

# BATCH_SIZE = 32         # can increase depending on GPU
BATCH_SIZE = 16       # standardizing to 16 to avoid crashing with unfrozen runs

# FREEZE = 23             # head only trainable
# FREEZE = 10             # Unfreeze C2PSA and neck and head
FREEZE = 11             # head and neck only trainable

default = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = input("Enter run name (e.g. shark_v2_unfrozen): ").strip() or default
OPTIMIZER = "MuSGD" # SGD with Muon-style orthagonalized updates

# --- tuned hyperparameters ---
# These values were selected by hyperparameter_search.py on 2026-04-08.
# To update: run `python hyperparameter_search.py --recommend <csv_path>`
# and paste the recommended values below.

# LR0 = 0.01329
LR0 = 0.001329 # drop 10x for unfreezing

LRF = 0.01337
MOMENTUM = 0.7198
WEIGHT_DECAY = 0.0005
DEGREES = 0.00062
SCALE = 0.32768
FLIPUD = 0.01632 # data augmentation param: vertical flip (up-down)
FLIPLR = 0.41356 # data augmentation param: horizontal flip (left-right)
MOSAIC = 0.7285
HSV_S = 0.68914 # data augmentation param: HSV saturation (color shift)
HSV_V = 0.32452 # data augmentation param: HSV brightness

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
    freeze = FREEZE,        
    patience = 20,          # standardized across all runs after sharks_v4_frozen
    lr0 = LR0,
    lrf = LRF, # learning rate final
    momentum = MOMENTUM, # momentum for MuSGD (higher means past gradients have more influence on current step; helps smooth noisy gradients but may overshoot)
    weight_decay = WEIGHT_DECAY, # essentailly L2 regularization (when used with SGD based optimizers)
    degrees = DEGREES,      # randomly rotates training images by up to this many degrees
    scale = SCALE,          # randomly scale training images by this factor (e.g. 0.5 means scale by 50-150%)
    flipud = FLIPUD,        # randomly flip training images vertically (up-down) with this probability
    fliplr = FLIPLR,        # randomly flip training images horizontally (left-right) with this probability
    mosaic = MOSAIC,        # YOLO specific augmentation that tiles 4 diff training images into one composite (force model to detect at diff scales and cluttered contexts)
    hsv_s = HSV_S,         # randomly shift saturation by up to this factor (e.g. 0.5 means shift by up to 50%)
    hsv_v = HSV_V,         # randomly shift brightness by up to this factor
    name = RUN_NAME,
    exist_ok = True,        # overwrite existing runs (won't crash if run name already exists)
    device = device_arg,       # explicitly pass GPU/CPU
    workers = NUM_WORKERS,  # from hardware acceleration above
    verbose = True,         # print training progress to console
    cache = False,         # disk is fast enough; saves 136GB
)

try:
    print(f"Best weights saved to: {model.trainer.best}")
except AttributeError:
    print("Training did not complete — best weights path unavailable.")