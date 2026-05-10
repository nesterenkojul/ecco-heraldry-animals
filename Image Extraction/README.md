# Image Extraction

This folder contains the code for detecting, cropping, and splitting coats of arms from scanned heraldry pages.

The pipeline has five steps:

1. Convert manually annotated XML files to YOLO format.
2. Crop manually annotated coats of arms from the source pages.
3. Train a YOLO object detection model.
4. Predict coats of arms on unseen pages.
5. Split each cropped coat of arms into left and right supporter images.


## Folder structure

    Image Extraction/
    ├── scripts/
    │   ├── 1_convert_voc_to_yolo.py
    │   ├── 2_crop_coats_from_xml.py
    │   ├── 3_train_yolo.py
    │   ├── 4_predict_coats.py
    │   └── 5_split_supporters.py
    ├── img_dataset_scaled/
    ├── dataset/
    ├── models/
    ├── cropped_coats_of_arms/
    ├── split_supporters/
    └── test_pages/
    
## Input data

The pipeline expects manually annotated page images and XML files in:

img_dataset_scaled/


The annotations must be in Pascal VOC XML format and use the class name:

coat_of_arms


## Step 1: Convert XML annotations to YOLO format


Image Extraction/scripts/1_convert_voc_to_yolo.py

This creates:

dataset/images/train/
dataset/images/val/
dataset/labels/train/
dataset/labels/val/
dataset/final_dataset.yaml


## Step 2: Crop annotated coats of arms

Image Extraction/scripts/2_crop_coats_from_xml.py

This crops all manually annotated coats of arms and saves them to:

cropped_coats_of_arms/


## Step 3: Train YOLO detector

Image Extraction/scripts/3_train_yolo.py

This trains a YOLO object detection model using:

dataset/final_dataset.yaml

The model output is saved to:

models/coat_of_arms_final/

## Step 4: Predict coats of arms on test pages

Image Extraction/scripts/4_predict_coats.py

This applies the trained model to images in:

test_pages/


Predictions and cropped detections are saved by Ultralytics.

## Step 5: Split supporter images

Image Extraction/scripts/5_split_supporters.py

This splits each cropped coat of arms into a left and right half.

The output is saved to:

split_supporters/

These images can then be used for supporter classification.

## Notes

Large generated folders such as `dataset/`, `models/`, `cropped_coats_of_arms/`, and `split_supporters/` are not meant to be committed to GitHub.

The repository should contain the scripts and documentation, but not the full image dataset or trained model weights.