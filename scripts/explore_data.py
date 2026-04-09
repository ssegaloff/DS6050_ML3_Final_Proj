'''
explore_data.py

Exploratory data analysis for our shark detection dataset.

Reads the Roboflow-generated data.yaml to discover split paths and class names,
then reports:
  - Image and annotation counts per split (train / val / test)
  - Class distribution (annotations per class, per split)
  - Image dimension and aspect ratio statistics
  - A saved sample grid of images with bounding boxes drawn

Usage:
    python explore_data.py

Output:
    Prints a summary to the console.
    Saves a sample image grid to eda_outputs/sample_grid.jpg
'''

# imports
import random
import yaml
from pathlib import Path

from PIL import Image, ImageDraw

# configure paths and and constants
DATA_YAML = Path('../DS6050_ML3_Final_Proj/data/raw/data.yaml')
OUTPUT_DIR = Path("eda_outputs")
SAMPLE_SIZE = 16    # number of images to show in the grid
GRID_COLS = 4       # grid will be GRID_COLS x (SAMPLE_SIZE // GRID_COLS)
SEED = 26

# load data.yaml
if not DATA_YAML.exists():
    raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")

with open(DATA_YAML, "r") as f:
    cfg = yaml.safe_load(f) # safe_load() won't execute arbitrary python embedded in the YAML (good practice)

class_names = cfg["names"] # using len(class_names) instead of nc
yaml_dir = DATA_YAML.parent

# resolve split paths
splits = {
    "train": (yaml_dir / cfg["train"]).resolve(),
    "val":   (yaml_dir / cfg["val"]).resolve(),
    "test":  (yaml_dir / cfg["test"]).resolve(),
}


def draw_boxes(image_file: Path, class_names: list) -> Image.Image:
    """Open an image, draw its YOLO bounding boxes, and return the annotated image."""
    labels_file = image_file.parent.parent / "labels" / image_file.with_suffix(".txt").name

    with Image.open(image_file) as img:
        img = img.convert("RGB")
        W, H = img.size
        draw = ImageDraw.Draw(img)

        if labels_file.exists():
            for line in labels_file.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])

                x0 = (x_center - w / 2) * W
                y0 = (y_center - h / 2) * H
                x1 = (x_center + w / 2) * W
                y1 = (y_center + h / 2) * H

                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                draw.text((x0, y0), class_names[class_id], fill="red")

        return img
    
def make_grid(samples: list, class_names: list, cols: int) -> Image.Image:
    """
    Draw bounding boxes on sampled images and stitch them into a grid.
    Note: Assumes all images are the same size since Roboflow exports at a consistent resolution.
    """
    annotated = [draw_boxes(f, class_names) for f in samples]

    cell_w, cell_h = annotated[0].size # reads size from first image and assumes consistency across all images
    rows = len(annotated) // cols

    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(0, 0, 0)) # create a black canvas sized to fit the full grid

    for idx, img in enumerate(annotated):
        row = idx // cols  # get row index
        col = idx % cols   # get column index
        grid.paste(img, (col * cell_w, row * cell_h)) # place each annotated image at correct pixel offset on the canvas

    return grid

# per-split summary
print("=" * 50)
print("SPLIT SUMMARY")
print("=" * 50)

for split_name, images_dir in splits.items():
    labels_dir = images_dir.parent / "labels"

    image_files = sorted(images_dir.glob("*.jpg")) # sort for reproducibility
    label_files = sorted(labels_dir.glob("*.txt"))

    # Count annotations per class
    class_counts = {}
    for label_file in label_files:
        for line in label_file.read_text().splitlines(): # splitlines reads whole label file and splits on newlines
            if not line.strip(): # skip empty lines
                continue
            class_id = int(line.split()[0]) # grabs first whitespace delimited token (always class ID in yolo format)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1  # tally annotations per class

    total_annotations = sum(class_counts.values())

    print(f"\n{split_name}")
    print(f"  Images      : {len(image_files)}")
    print(f"  Label files : {len(label_files)}")
    print(f"  Annotations : {total_annotations}")
    for class_id, count in sorted(class_counts.items()): # ensure classes always print in order (regardless of appearance order in the files)
        name = class_names[class_id]
        print(f"    {name:<20} : {count}")

    # Image dimension stats
    widths = []
    heights = []
    for image_file in image_files:
        with Image.open(image_file) as img:
            widths.append(img.width)
            heights.append(img.height)

    aspect_ratios = [w / h for w, h in zip(widths, heights)]

    print(f"  Image dimensions:")
    print(f"    Width  : min={min(widths)}, max={max(widths)}, avg={sum(widths)//len(widths)}") # // for average width and height gives clean integer since pixels are whole numbers
    print(f"    Height : min={min(heights)}, max={max(heights)}, avg={sum(heights)//len(heights)}")
    print(f"    Aspect : min={min(aspect_ratios):.2f}, max={max(aspect_ratios):.2f}, avg={sum(aspect_ratios)/len(aspect_ratios):.2f}") # / for average aspect ratio so we can have decimals


# sample image grid
random.seed(SEED)
train_images = sorted(splits["train"].glob("*.jpg"))
sample = random.sample(train_images, min(SAMPLE_SIZE, len(train_images))) # guard to prevent crashing if train set is smaller than SAMPLE_SIZE


# build and save grid
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # wont crash if dir already exists
grid = make_grid(sample, class_names, GRID_COLS)
output_path = OUTPUT_DIR / "sample_grid.jpg"
grid.save(output_path)
print(f"\nSample grid saved to: {output_path}")