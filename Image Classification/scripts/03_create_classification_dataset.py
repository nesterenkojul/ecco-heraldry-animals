import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "annotations_ml_local.csv"
DATASET_ROOT = BASE_DIR / "supporter_cls_ml"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 1e-9

def copy_subset(df: pd.DataFrame, subset_name: str, dataset_root: Path):
    copied = 0
    missing = 0

    for _, row in df.iterrows():
        src = Path(row["image"])
        label = row["label"]

        if not src.exists():
            print(f"[WARN] File not found: {src}")
            missing += 1
            continue

        dst_dir = dataset_root / subset_name / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied += 1

    print(f"{subset_name}: copied={copied}, missing={missing}")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if not {"image", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: image,label")

    df["label"] = df["label"].astype(str).str.strip().str.lower()

    print("Total samples:", len(df))
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_SIZE),
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Val:  ", len(val_df))
    print("Test: ", len(test_df))

    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    copy_subset(train_df, "train", DATASET_ROOT)
    copy_subset(val_df, "val", DATASET_ROOT)
    copy_subset(test_df, "test", DATASET_ROOT)

    train_df.to_csv(DATASET_ROOT / "train_split.csv", index=False)
    val_df.to_csv(DATASET_ROOT / "val_split.csv", index=False)
    test_df.to_csv(DATASET_ROOT / "test_split.csv", index=False)

    print(f"\nCreated YOLO classification dataset at: {DATASET_ROOT.resolve()}")


if __name__ == "__main__":
    main()