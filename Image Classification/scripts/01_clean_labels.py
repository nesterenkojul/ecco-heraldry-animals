import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "annotations_detailed.csv"
OUTPUT = BASE_DIR / "annotations_ml.csv"

LABEL_MAPPING = {
    # Birds
    "ostrich": "bird_other",
    "swan": "bird_other",
    "stork": "bird_other",
    "chicken": "bird_other",

    # Mammals
    "boar": "mammal_other",
    "bear": "mammal_other",
    "sheep": "mammal_other",
    "goat": "mammal_other",
    "hare": "mammal_other",
    "beaver": "mammal_other",
    "elephant": "mammal_other",
    "monkey": "mammal_other",

    # Hybrids
    "lion_fish": "hybrid_other",
    "water_horse": "hybrid_other",
    "hybrid_other": "hybrid_other",

    # Deer
    "doe_deer": "stag_deer",
}

DROP_LABELS = {"unclear", "nothing", "", "nan"}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    df.columns = [c.strip().strip('"') for c in df.columns]

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.lower()
    )

    df = df[~df["label"].isin(DROP_LABELS)].copy()
    df["label"] = df["label"].replace(LABEL_MAPPING)

    df.to_csv(OUTPUT, index=False)

    print(f"Saved cleaned annotation file to: {OUTPUT.resolve()}")
    print("\nML label distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()