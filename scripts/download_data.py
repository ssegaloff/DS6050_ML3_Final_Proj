import os
from pathlib import Path
from roboflow import Roboflow

DATA_ROOT = Path("../DS6050_ML3_Final_Proj/data/raw")

if not DATA_ROOT.exists():
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("chloe-space").project("shark-detection-and-classification")
    dataset = project.version(2).download("yolo26", location=str(DATA_ROOT))
else:
    print("Dataset already present, skipping download.")
