from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "coat_of_arms_final" / "weights" / "best.pt"
SOURCE_DIR = PROJECT_ROOT / "test_pages"

IMG_SIZE = 1280
CONFIDENCE = 0.25

def main():
    model = YOLO(MODEL_PATH)

    model.predict(
        source=SOURCE_DIR,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        device=0,
        save=True,
        save_crop=True,
    )


if __name__ == "__main__":
    main()