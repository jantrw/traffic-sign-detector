# YOLOv8 Traffic Sign Detector

---

A computer-vision project for detecting and classifying German traffic signs (bounding boxes + class labels). This repository provides conversion tools, validators and training scripts to convert the SynsetSignsetGermany raw data into a YOLOv8-compatible dataset, train a YOLOv8 model, and run inference.

<!-- Preview: small thumbnails (centered, clean, no borders) -->
<div align="center" style="margin: 16px 0;">
  <a href="images/predict/steileKurve.jpg">
    <img src="images/predict/steileKurve.jpg" alt="steileKurve" width="260px" height="350px">
  </a>
  <a href="images/predict/vorfahrt_blauerHimmel.jpg">
    <img src="images/predict/vorfahrt_blauerHimmel.jpg" alt="vorfahrt_blauerHimmel" width="260px" height="350px">
  </a>
  <a href="images/predict/wild.jpg">
    <img src="images/predict/wild.jpg" alt="wild" width="260px" height="350px">
  </a>
</div>

---

## Table of contents
1. [Reproduce](#reproduce)  
   1.1 [Detected images](#detected-images)  
2. [Training pictures & data](#training-pictures--data)  
3. [What this repo does (summary)](#what-this-repo-does-summary)  
4. [Dataset](#dataset)  
5. [Files & scripts](#files--scripts)  
6. [Example: convert & prepare dataset](#example-convert--prepare-dataset)  
7. [Validate dataset](#validate-dataset)  
8. [Train (yolov8m) — example](#train-yolov8m-example)  
9. [Inference example](#inference-example)  

---
## Reproduce & Quick Start

### 1) Clone the repository
```bash
git clone https://github.com/jantrw/traffic-sign-detector.git
cd traffic-sign-detector
```

### 2) Create and activate a virtual environment (recommended)
**Linux / macOS**
```bash
python3 -m venv yolov8traffictracking
source yolov8traffictracking/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv yolov8traffictracking
.\yolov8traffictracking\Scripts\Activate.ps1
# (or for cmd): yolov8traffictracking\Scripts\activate.bat
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4) Install Model & Git LFS (important)
The model `training_results/yolov8m_run_013/weights/best.pt` is stored with **Git LFS** because it exceeds GitHub's 100 MB file size limit. After cloning the repo you **must** pull LFS objects to get the real `.pt` file (otherwise you only get a pointer file).

**Linux / macOS**
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs         # or use your distro package manager
git lfs install
```

**Windows (PowerShell / admin)**
```powershell
choco install git-lfs # Or Download installer from https://git-lfs.github.com/
git lfs install
```
**Download LFS objects (this fetches the real model and not just the pointer)**
```bash
git lfs pull        # fetches LFS objects for the current refs
```

## 5) Python wrapper scripts (recommended)
The repo includes `run_detect.py` for convenient and reproducible runs.

**Example: single-image**
```bash
python run_detect.py images/steileKurve.jpg
```
**Example: with your own Image**
```bash
python run_detect.py C:/Users/deinUserName/Downloads/image.png
```
**Example: multi-image detection**
```bash
python run_detect.py images/
```
### 6) Troubleshooting quick guide
- **No detections:** try `--conf 0.01` or `conf=0.05` and `imgsz=1024`.  
- **Detections but many false positives:** raise `conf` to `0.3–0.5`, use temporal smoothing in video (require detection in 2 consecutive frames).  
- **Model checkpoint missing or wrong:** ensure `training_results/.../best.pt` exists and is a detection checkpoint . Use `yolo val model=... data=data2.yaml` to verify performance on the validation set.  

### 8) Detected Images

<div style="display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start;">
  <img src="images/predict/steileKurve.jpg" alt="steileKurve" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
  <img src="images/predict/steinschlag.jpg" alt="steinschlag" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
  <img src="images/predict/vorfahrt_blauerHimmel.jpg" alt="vorfahrt_blauerHimmel" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
  <img src="images/predict/wild.jpg" alt="wild" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
  <img src="images/predict/Zeichen261.jpg" alt="Zeichen261" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
</div>

---

## Training Pictures & Data


### Training Batches with Border
![Training_Batches_Image](images/helper/val_batch1_labels.jpg)
### Training Data in resumed Training (=14 Epochs)
![Trainign_Data](images/helper/results.png)

### Confusion Matrix Normalized: Recognition Rate in Percentages
![ConfusionMatrixNormalized](images/helper/confusion_matrix_normalized.png)


## What this repo does (summary)

1. Convert SynsetSignsetGermany → YOLO format (images + `.txt` labels with normalized bboxes).
2. Produce `data/yolo_signs` containing:
   - `train/images`, `train/labels`
   - `val/images`,   `val/labels`
   - `classes.txt` (one class name per line)
   - `data.yaml` (YOLO/Ultralytics dataset file with relative paths)
3. Validate dataset integrity with `scripts/validate_yolo_dataset.py`.
4. Train YOLOv8 (we use `yolov8m.pt` by default) using `scripts/train_model/start_train.py` with W&B logging and TensorBoard.

## Dataset
 **Current dataset:** 
`SynsetSignsetGermany` (replaces GTSRB for this project).

Download location (example / internal):  
https://owncloud.fraunhofer.de/index.php/s/OLQ6E5BVN4pRGu8?path=%2FSynsetSignsetGermany

Place the extracted SynsetSignsetGermany archive under `data/raw/` so the converter scripts can find the expected folders.

Typical extracted structure (important folders):

```
data/raw/SynsetSignsetGermany/
  Ogre/               # images organized in class subfolders (0..210)
  Masks/              # masks per-class (0..210) used to compute bboxes
  Labels/             # optional semantic labels per-class (0..210)
  CsvFiles/           # various csv splits and metadata (optional)
  Cycles/             # JSON metadata per-image (optional)
```

Notes:
- Each Ogre class folder contains the original image files + .json with metadata (e.g. `0_ogre.png`, `0_ogre.json`, `1_ogre.png`,`1_ogre.json` …).
- Masks and Labels folders are used to extract bounding boxes (masks -> contours -> bbox).
- Many helper scripts expect these folders to exist under `data/raw/SynsetSignsetGermany`.

---

## Example: Convert & prepare dataset

From project root, example conversion command:
```
python scripts/convert_synset_to_yolo.py `
   --raw data/raw/SynsetSignsetGermany `
   --masks data/raw/SynsetSignsetGermany/Masks `
   --semantic data/raw/SynsetSignsetGermany/Labels `
   --out data/yolo_signs `
   --split 0.8 0.2 0.0
```
**Note: Paths in `scripts/convert_synset_to_yolo.py` + the conversion command maybe have to be adjusted.**

This will:
- index masks and labels,
- compute bounding boxes from masks,
- copy images into `data/yolo_signs/train|val/images` and write corresponding `.txt` label files,
- write `classes.txt`.

---
## Validate dataset (quick)

Run:
```
python scripts/validate_yolo_dataset.py
```
The script prints:
- image/label counts for `train` and `val`,
- up to 10 sample images without labels,
- up to 10 labels without images,
- basic format issues found in the first 200 labels.

---

## Troubleshooting & tips

- **`val/images` missing**: Check `data/yolo_signs/data.yaml` for relative paths and that `val/images` actually exists. If you have only `train/`, use `create_val_split.py` to carve out a validation set.
- **Slow conversion**: The mask-indexing step can be I/O heavy. If you see very long times while scanning many files, ensure masks are on a local disk (not slow network) and the script is using the optimized index variant (`--index` style`).
- **Windows multiprocessing errors**: Use `if __name__ == '__main__':` guard and `multiprocessing.freeze_support()` in start scripts (already included in `start_train.py`). If still failing, use `workers=0` in training.
- **Extra small model downloads (e.g., `yolo11n.pt`)**: Usually caused by tracker or helper modules. Keep `tracker=None` if you don't require tracking features.

---

## Example: Generate data.yaml (relative paths)

Run:
```
python scripts/create_data_yaml.py 
```
`data/yolo_signs/data.yaml` will use relative paths (so the trainer resolves `train: train/images`, `val: val/images` when `path: .`).

Example `data.yaml` contents:

path: data/yolo_signs
train: train/images
val:   val/images
nc:    211
names:
  0: "0_Geschwindigkeit20"
  1: "100_VerbotReiter"
  ...

Important: Ultralytics accepts either `names` list or mapping; `classes.txt` can be used by many visualization tools to map id->name.

---

## Train (YOLOv8, example)

This example shows how the model was trained using Ultralytics YOLOv8. 

I trained the model on Kaggle (https://www.kaggle.com) because 17 epochs took about 16 hours on my hardware. For longer training runs or larger datasets, Kaggle or another GPU instance is recommended.

Kaggle training command (edit ur paths into it):
```
!python -u -c "from ultralytics import YOLO; YOLO('yolov8m.pt').train(data='/kaggle/working/data2.yaml', epochs=25, imgsz=640, batch=16, name='yolov8m_run_01', project='/kaggle/working/synset_signset_germany', tracker=None)" | tee /kaggle/working/train_log.txt
```
Start training (OWN GPU), (from project root):
```
python scripts/train_model/start_train.py
```
Key training notes:
- Uses `ultralytics` `YOLO("yolov8m.pt")` and `model.train(...)`.
- The script integrates with **Weights & Biases** (W&B). Login once with `wandb login` if you want online monitoring.
- Trainer prints a TensorBoard command like:
  TensorBoard: Start with 'tensorboard --logdir <run_dir>', view at http://localhost:6006/

- If you run on Windows and encounter the multiprocessing spawn error, either:
  - ensure `start_train.py` is the guarded main module (it is in the repo), or
  - use `workers=0` in `model.train(...)` (slower I/O, but avoids spawn issues).
- If you don't want tracker downloads (small helper weights like `yolo11n.pt`), the trainer script in repo sets `tracker=None` to avoid the extra download by default.

Example training parameters you can tweak in `start_train.py`:
```python
model.train(
  data="data/yolo_signs/data.yaml",
  epochs=50,
  imgsz=640,
  batch=16,               # adjust by GPU memory
  name="synset_signs_run",
  project="synset_signset_germany",
  tracker=None,           # avoid extra downloads
  # workers=0             # use if spawn issues occur
)
```

W&B run is available at `https://wandb.ai/<your-account>/<project>/runs/<run-id>` (the script prints the run URL at start).

---

## Inference example (draw boxes with class names)

Below is a minimal snippet that runs inference and draws bounding boxes labeled by class names (labels come from `classes.txt`):

```python
from ultralytics import YOLO
import cv2, json, pathlib

model = YOLO("runs/train/your_run/weights/best.pt")  # or yolov8m.pt (finetune)
res = model("some_test_image.jpg")                   # returns results

# load class names
names = [l.strip() for l in open("data/yolo_signs/classes.txt", encoding="utf8").read().splitlines()]

# draw (first result)
boxes = res[0].boxes  # ultralytics result
img = cv2.imread("some_test_image.jpg")
for box in boxes:
    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
    cls = int(box.cls[0].item())
    label = names[cls] if cls < len(names) else str(cls)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
cv2.imwrite("inference_with_labels.jpg", img)
```

This draws borders with the **folder/class names** (as requested) by mapping class IDs to names using `classes.txt`.

---