<div align="center">
  <h1>YOLOv8 Traffic Sign Detector</h1>
  <p><strong>Focused computer-vision workflow for German road-sign detection, from raw dataset conversion to YOLOv8 training and inference.</strong></p>
  <p>
    <a href="https://www.python.org/" target="_blank" rel="noreferrer"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://docs.ultralytics.com/" target="_blank" rel="noreferrer"><img src="https://img.shields.io/badge/Ultralytics-YOLOv8-111827" alt="Ultralytics YOLOv8"></a>
    <a href="https://numpy.org/" target="_blank" rel="noreferrer"><img src="https://img.shields.io/badge/NumPy-2.2.6-013243?logo=numpy&logoColor=white" alt="NumPy"></a>
    <a href="https://pytorch.org/" target="_blank" rel="noreferrer"><img src="https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer"><img src="https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
    <a href="https://owncloud.fraunhofer.de/index.php/s/OLQ6E5BVN4pRGu8?path=%2FSynsetSignsetGermany" target="_blank" rel="noreferrer"><img src="https://img.shields.io/badge/Dataset-SynsetSignsetGermany-0F766E" alt="Dataset"></a>
  </p>
  <p>
    <a href="#1-key-features">Key Features</a> •
    <a href="#2-overview">Overview</a> •
    <a href="#3-setup">Setup</a> •
    <a href="#4-dataset">Dataset</a> •
    <a href="#5-training">Training</a> •
    <a href="#6-inference">Inference</a>
  </p>
</div>

This repository is built around a narrow goal: detect and classify German traffic signs with a dedicated pipeline instead of relying on broad or mixed-country detection projects. It also includes the full glue code for dataset conversion, validation, training, and local inference, based on a `SynsetSignsetGermany` dataset that I manually reviewed and validated to ensure a clean YOLO-compatible structure and annotation workflow.

<!-- Preview: GitHub-friendly inline images -->
<p align="center">
  <a href="images/predict/steileKurve.jpg"><img src="images/predict/steileKurve.jpg" alt="steileKurve" width="280" height="370"></a>
  <a href="images/predict/vorfahrt_blauerHimmel.jpg"><img src="images/predict/vorfahrt_blauerHimmel.jpg" alt="vorfahrt_blauerHimmel" width="280" height="370"></a>
  <a href="images/predict/wild.jpg"><img src="images/predict/wild.jpg" alt="wild" width="280" height="370"></a>
</p>

---

## 1. Key Features

- Focused on German road signs instead of a mixed-country detector.
- Converts `SynsetSignsetGermany` into a YOLOv8-ready dataset.
- Includes validation scripts, training entrypoints, and inference helpers.
- Keeps example predictions and training outputs in the repository.

---

## 2. Overview

1. Convert `SynsetSignsetGermany` to YOLO format with normalized bounding boxes.
2. Produce `data/yolo_signs` with `train/`, `val/`, `classes.txt`, and `data.yaml`.
3. Validate dataset integrity with `scripts/validate_yolo_dataset.py`.
4. Train YOLOv8 with `scripts/train_model/start_train.py`.
5. Run local inference with `run_detect.py` and your trained `.pt` weights.

---

## 3. Setup

### 3.1 Clone the repository

```bash
git clone https://github.com/jantrw/traffic-sign-detector.git
cd traffic-sign-detector
```

### 3.2 Create and activate a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 Model files

Cloning this repository is not enough for end-to-end inference. You need your own local model file.

- `yolov8m.pt` is not stored in this repository.
- `training_results/.../best.pt` is not stored in this repository.
- Ultralytics downloads `yolov8m.pt` automatically during training if needed.
- Inference requires a valid local `.pt` path passed via `--model`.

---

## 4. Dataset

**Current dataset:** `SynsetSignsetGermany`

Download location:
https://owncloud.fraunhofer.de/index.php/s/OLQ6E5BVN4pRGu8?path=%2FSynsetSignsetGermany

Place the extracted archive under `data/raw/` so the scripts can find the expected folders.

Typical structure:

```text
data/raw/SynsetSignsetGermany/
  Ogre/
  Masks/
  Labels/
  CsvFiles/
  Cycles/
```

Notes:

- `Ogre/` contains the source images and per-image metadata JSON files.
- `Masks/` and `Labels/` are used to derive bounding boxes and semantic labels.
- Helper scripts expect the dataset under `data/raw/SynsetSignsetGermany`.

### 4.1 Convert and prepare the dataset

```powershell
python scripts/convert_synset_to_yolo.py `
  --raw data/raw/SynsetSignsetGermany `
  --masks data/raw/SynsetSignsetGermany/Masks `
  --semantic data/raw/SynsetSignsetGermany/Labels `
  --out data/yolo_signs `
  --split 0.8 0.2 0.0
```

This computes bounding boxes from masks and writes images plus `.txt` labels into `data/yolo_signs/train` and `data/yolo_signs/val`.

### 4.2 Validate the dataset

```bash
python scripts/validate_yolo_dataset.py
```

What it checks:

- counts for `train` and `val` images and labels
- sample images missing labels, or labels missing images
- basic YOLO label format issues

### 4.3 Generate `data.yaml`

```bash
python scripts/create_data_yaml.py
```

Expected structure in `data/yolo_signs/data.yaml`:

```yaml
path: data/yolo_signs
train: train/images
val: val/images
nc: 211
names:
  0: "0_Geschwindigkeit20"
  1: "100_VerbotReiter"
  ...
```

Important: Ultralytics accepts either a `names` list or mapping. `classes.txt` is useful for manual label lookup and visualization tooling. If your checked-in `data.yaml` differs, regenerate it with `python scripts/create_data_yaml.py`.

---

## 5. Training

### 5.1 Example outputs

![Validation batches](images/helper/val_batch1_labels.jpg)
![Training results](images/helper/results.png)
![Normalized confusion matrix](images/helper/confusion_matrix_normalized.png)

### 5.2 Start training

```bash
python scripts/train_model/start_train.py
```

Notes:

- Adjust `batch` and `imgsz` to your GPU memory.
- The trainer initializes Weights & Biases logging.
- Ultralytics downloads the base `yolov8m.pt` weights if they are not present locally.

---

## 6. Inference

### 6.1 Run the wrapper

```bash
python run_detect.py images/steileKurve.jpg --model path/to/best.pt
```

More examples:

```bash
python run_detect.py C:/Users/deinUserName/Downloads/image.png --model path/to/best.pt
python run_detect.py images/ --model path/to/best.pt
```

### 6.2 Troubleshooting

- No detections: try `--conf 0.01` or a larger `--imgsz`.
- Too many false positives: raise `--conf` to around `0.3` to `0.5`.
- Missing checkpoint: pass a valid local `.pt` path via `--model`.

### 6.3 Example detections

<div style="display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start;">
  <img src="images/predict/steileKurve.jpg" alt="steileKurve" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
  <img src="images/predict/steinschlag.jpg" alt="steinschlag" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
  <img src="images/predict/vorfahrt_blauerHimmel.jpg" alt="vorfahrt_blauerHimmel" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
  <img src="images/predict/wild.jpg" alt="wild" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
  <img src="images/predict/Zeichen261.jpg" alt="Zeichen261" style="width:32%; max-height:180px; object-fit:cover; border:1px solid #ddd; padding:2px;">
</div>

### 6.4 Minimal inference snippet

```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/train/your_run/weights/best.pt")
results = model("some_test_image.jpg")

names = [line.strip() for line in open("data/yolo_signs/classes.txt", encoding="utf8")]
image = cv2.imread("some_test_image.jpg")

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cls = int(box.cls[0].item())
    label = names[cls] if cls < len(names) else str(cls)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

cv2.imwrite("inference_with_labels.jpg", image)
```
