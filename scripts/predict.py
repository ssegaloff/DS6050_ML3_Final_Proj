
from ultralytics import YOLO

model = YOLO("runs/detect/shark_v1/weights/best.pt")

results = model.predict(
    source="data/raw/test/images/502_jpg.rf.b5b40891ad2e884c13e5be8377844adb.jpg",   # folder, image, or video
    conf = 0.85,
    save=True,                 # saves annotated images with bounding boxes
    save_txt=True,             # saves label files
)