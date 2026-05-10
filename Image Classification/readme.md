# Heraldic Supporter Classification Pipeline

This repository contains the code used to prepare, train, and apply an image classification model for heraldic supporters.

The workflow consists of:

1. Cleaning detailed manual annotation labels.
2. Mapping rare labels into broader machine-learning classes.
3. Converting image paths to local project paths.
4. Creating train/validation/test splits.
5. Training a YOLO image classification model.
6. Applying the trained model to automatically sort high-confidence predictions.

## Scripts

### 01_clean_labels.py
Reads `annotations_detailed.csv`, removes unusable labels such as `unclear` and `nothing`, and maps rare supporter classes into broader labels such as `bird_other`, `mammal_other`, and `hybrid_other`.

Output: `annotations_ml.csv`

### 02_make_local_csv.py
Converts exported annotation paths into local image paths and keeps only the columns needed for classification.

Output: `annotations_ml_local.csv`

### 03_create_classification_dataset.py
Creates a YOLO-compatible image classification dataset with train, validation, and test folders.

Output: `supporter_cls_ml/`

### 04_train_classifier.py
Trains a YOLO image classification model on the prepared supporter dataset.

### 05_auto_label_predictions.py
Uses a trained classifier to sort new images into predicted class folders. Only predictions above a confidence threshold are accepted automatically; uncertain cases are placed in a review folder.

## Data

The repository does not include the full image dataset or trained model weights because these files are too large and/or derived from copyrighted source material.

Expected local files:

- `annotations_detailed.csv`
- `images/`
- `cropped_coats_of_arms/`
- trained YOLO weights, e.g. `runs/classify/.../weights/best.pt`