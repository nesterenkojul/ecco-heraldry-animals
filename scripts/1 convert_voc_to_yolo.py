import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "img_dataset_scaled"
OUTPUT_DIR = PROJECT_ROOT / "dataset"

TRAIN_RATIO = 0.8
CLASS_ID = 0
CLASS_NAME = "coat_of_arms"
RANDOM_SEED = 42


def create_folders():
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def create_yaml():
    yaml_path = OUTPUT_DIR / "final_dataset.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            f"path: {OUTPUT_DIR}\n\n"
            "train: images/train\n"
            "val: images/val\n\n"
            "names:\n"
            f"  0: {CLASS_NAME}\n"
        )

    print(f"Created YAML file: {yaml_path}")


def voc_box_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    box_width = xmax - xmin
    box_height = ymax - ymin

    x_center = (xmin + box_width / 2) / image_width
    y_center = (ymin + box_height / 2) / image_height
    box_width = box_width / image_width
    box_height = box_height / image_height

    return x_center, y_center, box_width, box_height


def convert_xml(xml_file: Path, split: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename_tag = root.find("filename")
    size_tag = root.find("size")

    if filename_tag is None or size_tag is None:
        print(f"Skipping {xml_file.name}: missing filename or size")
        return

    image_name = filename_tag.text
    image_path = SOURCE_DIR / image_name

    if not image_path.exists():
        print(f"Skipping {xml_file.name}: image not found: {image_path}")
        return

    image_width = int(size_tag.find("width").text)
    image_height = int(size_tag.find("height").text)

    lines = []

    for obj in root.findall("object"):
        name_tag = obj.find("name")

        if name_tag is not None and name_tag.text != CLASS_NAME:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = int(float(box.find("xmin").text))
        ymin = int(float(box.find("ymin").text))
        xmax = int(float(box.find("xmax").text))
        ymax = int(float(box.find("ymax").text))

        x_center, y_center, box_width, box_height = voc_box_to_yolo(
            xmin, ymin, xmax, ymax, image_width, image_height
        )

        lines.append(
            f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    label_path = OUTPUT_DIR / "labels" / split / f"{xml_file.stem}.txt"
    image_output_path = OUTPUT_DIR / "images" / split / image_name

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    shutil.copy2(image_path, image_output_path)


def main():
    random.seed(RANDOM_SEED)

    create_folders()
    create_yaml()

    xml_files = sorted(SOURCE_DIR.glob("*.xml"))
    random.shuffle(xml_files)

    split_index = int(len(xml_files) * TRAIN_RATIO)

    train_files = xml_files[:split_index]
    val_files = xml_files[split_index:]

    print(f"Found XML files: {len(xml_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    for xml_file in train_files:
        convert_xml(xml_file, "train")

    for xml_file in val_files:
        convert_xml(xml_file, "val")

    print("YOLO dataset created.")


if __name__ == "__main__":
    main()