import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "annotations_ml.csv"
OUTPUT = BASE_DIR / "annotations_ml_local.csv"

EXPECTED_COLUMNS = [
    "annotation_id",
    "annotator",
    "created_at",
    "id",
    "image",
    "label",
    "lead_time",
    "updated_at",
]


def to_local_path(path):
    path = str(path).replace("\\", "/").strip().strip('"')
    filename = os.path.basename(path)
    return str(BASE_DIR.parent / "Image Extraction" / "split_supporters" / filename)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    df.columns = [c.strip().strip('"') for c in df.columns]

    if "image" not in df.columns or "label" not in df.columns:
        # Fallback for CSV files without headers
        df = pd.read_csv(INPUT, header=None, names=EXPECTED_COLUMNS)

    df["image"] = df["image"].apply(to_local_path)

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.lower()
    )

    df = df[["image", "label"]].copy()

    df = df[
        df["image"].notna()
        & df["label"].notna()
        & (df["image"] != "")
        & (df["label"] != "")
        & (~df["image"].str.endswith("/image"))
        & (df["label"] != "label")
        & (df["label"] != "nan")
    ].copy()

    df.to_csv(OUTPUT, index=False)

    print(f"Saved local ML CSV to: {OUTPUT.resolve()}")
    print(df.head())
    print("\nRows:", len(df))


if __name__ == "__main__":
    main()