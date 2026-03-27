import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


if torch.cuda.is_available():
    device_arg = 0
elif torch.backends.mps.is_available():
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
model = YOLO("runs/detect/shark_v1/weights/best.pt")

results = model.predict(
    source = str(SOURCE),
    conf = CONF_THRESHOLD,
    save=True,                 # saves annotated images with bounding boxes
    save_txt=True,             # saves label files
    name = RUN_NAME,
    device = device_arg
)

print(f"Results saved to: runs/detect/{RUN_NAME}")