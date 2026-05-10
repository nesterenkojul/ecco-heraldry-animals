from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "supporter_cls_ml"
BASE_MODEL = "yolo26l-cls.pt"

EPOCHS = 25
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = 0
WORKERS = 2
RUN_NAME = "supporter_cls_yolo26l_run1"

def main():
    model = YOLO(BASE_MODEL)

    model.train(
    data=str(DATASET_PATH),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=WORKERS,
    pretrained=True,
    name=RUN_NAME,
    project=str(BASE_DIR),
)


if __name__ == "__main__":
    main()