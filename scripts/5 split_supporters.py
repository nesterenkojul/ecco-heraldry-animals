from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "cropped_coats_of_arms"
OUTPUT_DIR = PROJECT_ROOT / "split_supporters"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def split_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")

    width, height = image.size
    middle = width // 2

    left = image.crop((0, 0, middle, height))
    right = image.crop((middle, 0, width, height))

    left_path = OUTPUT_DIR / f"{image_path.stem}_left.png"
    right_path = OUTPUT_DIR / f"{image_path.stem}_right.png"

    left.save(left_path)
    right.save(right_path)

    print(f"Saved: {left_path}")
    print(f"Saved: {right_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_files:
        print("No images found.")
        return

    print(f"Found images: {len(image_files)}")

    for image_path in image_files:
        split_image(image_path)

    print("Done.")


if __name__ == "__main__":
    main()