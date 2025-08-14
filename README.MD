# Traffic Sign Detector

A computer-vision project for detecting and classifying traffic signs (bounding boxes + class labels). The repository includes dataset preparation, training and evaluation scripts and (possibly) a small demo for running inference on uploaded images.

## Dataset

This project uses the **GTSRB (German Traffic Sign Recognition Benchmark)** for training.
Official dataset page: [https://benchmark.ini.rub.de/gtsrb\_dataset.html](https://benchmark.ini.rub.de/gtsrb_dataset.html)

## How to prepare the dataset

> Note: The repository already contains a script `src/prepare_dataset.py` that reads CSV annotation files and copies images into a structured folder hierarchy under `data/processed/gtsrb/{train,val,test}/{class_id}/...`.

### Steps to run the existing preparation script

1. Place the raw dataset under `data/raw/gtsrb/` (for example, as downloaded from the official benchmark).
2. Make sure `Train.csv` and `Test.csv` are available in `data/raw/gtsrb/`.
3. Run the script:

   ```bash
   python src/prepare_dataset.py
   ```
4. Example console output produced by the script (local run):

   ```text
   Verarbeite Trainingsdaten...
   Trainingsdaten verarbeitet: 31367 Bilder in 'train', 7842 in 'val'
   Verarbeite Testdaten...
   Testdaten verarbeitet: 12630 Bilder in 'test'

   Datenverarbeitung abgeschlossen
   ```
5. Result: `data/processed/gtsrb/train|val|test/<class_id>/...` with the copied images. The script checks whether destination files already exist and will not copy files twice.

### Missing steps for a YOLOv8-ready dataset
