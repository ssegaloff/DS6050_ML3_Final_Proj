from ultralytics import YOLO

model = YOLO("runs/detect/sharks_v5_freeze10/weights/best.pt")

# ultralytics will handle resizing of the image automatically

results = model.predict(
    source="https://cdn.images.express.co.uk/img/dynamic/128/590x/UK-707706.jpg",  # any public image URL
    conf=0.15,
    save=True,
)