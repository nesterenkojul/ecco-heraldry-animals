from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_YAML = PROJECT_ROOT / "dataset" / "final_dataset.yaml"

BASE_MODEL = "yolov8n.pt"

PROJECT = PROJECT_ROOT / "models"
RUN_NAME = "coat_of_arms_final"


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATASET_YAML,
        epochs=20,
        imgsz=1280,
        batch=8,
        device=0,
        workers=2,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()