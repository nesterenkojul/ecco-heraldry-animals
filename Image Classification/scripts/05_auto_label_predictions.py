from pathlib import Path
import shutil

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "supporter_cls_yolo26l_run1"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    BASE_DIR.parent
    / "Image Extraction"
    / "cropped_coats_of_arms"
)

OUTPUT_DIR = BASE_DIR / "auto_labeled"

CONF_THRESHOLD = 0.90

ALLOWED_CLASSES = {
    "dragon",
    "eagle",
    "griffin",
    "lion",
    "human",
    "unicorn",
}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    model = YOLO(MODEL_PATH)
    class_names = model.names

    total = 0
    accepted = 0
    review = 0

    image_paths = sorted(
        list(IMAGE_DIR.glob("*.png"))
        + list(IMAGE_DIR.glob("*.jpg"))
        + list(IMAGE_DIR.glob("*.jpeg"))
    )

    for img_path in image_paths:
        total += 1

        results = model(img_path)
        probs = results[0].probs

        top1 = probs.top1
        confidence = float(probs.top1conf)
        label = class_names[top1]

        if label in ALLOWED_CLASSES and confidence >= CONF_THRESHOLD:
            save_dir = OUTPUT_DIR / label
            accepted += 1
        else:
            save_dir = OUTPUT_DIR / "review"
            review += 1

        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, save_dir / img_path.name)

    print(f"Processed images: {total}")
    print(f"Auto-labeled:      {accepted}")
    print(f"Needs review:      {review}")
    print(f"Output directory:  {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()