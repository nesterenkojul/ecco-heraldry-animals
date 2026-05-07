import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "img_dataset_scaled"
OUTPUT_DIR = PROJECT_ROOT / "cropped_coats_of_arms"

CLASS_NAME = "coat_of_arms"


def crop_from_xml(xml_file: Path):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename_tag = root.find("filename")

    if filename_tag is None or not filename_tag.text:
        print(f"Skipping {xml_file.name}: no filename found")
        return

    image_name = filename_tag.text
    image_path = SOURCE_DIR / image_name

    if not image_path.exists():
        print(f"Skipping {xml_file.name}: image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    coat_counter = 1

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

        cropped = image.crop((xmin, ymin, xmax, ymax))

        output_name = f"{image_path.stem}_{coat_counter}.png"
        output_path = OUTPUT_DIR / output_name

        cropped.save(output_path)
        print(f"Saved: {output_path}")

        coat_counter += 1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(SOURCE_DIR.glob("*.xml"))

    if not xml_files:
        print("No XML files found.")
        return

    print(f"Found XML files: {len(xml_files)}")

    for xml_file in xml_files:
        crop_from_xml(xml_file)

    print("Done.")


if __name__ == "__main__":
    main()