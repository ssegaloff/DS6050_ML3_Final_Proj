# TODO: COMPLETE THIS SCRIPT AND MAKE UPDATES NEEDED

import os
from pathlib import Path
from ultralytics import YOLO

# --- configure variables ---
# TODO: CHANGE / DECIDE ON THESE PARAMETERS (everything is currently a placeholder)
DATA_ROOT = Path("../DS6050_ML3_Final_Proj/data/raw")
YAML_PATH = DATA_ROOT / "data.yaml"
MODEL_SIZE = "l"        # n, s, m, l, x
EPOCHS = 100
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
)

print(f"Best weights saved to: {model.trainer.best}")