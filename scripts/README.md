# Traffic Sign Detector – Training & Conversion Scripts

This README describes the scripts found in `scripts/` (and the `scripts/train_model/` subfolder) used to convert the SynsetSignsetGermany raw data into a YOLOv8-compatible dataset and to train a YOLOv8 model.

---

## Layout (relevant files)

Top-level `scripts/` folder contains:

- `convert_synset_to_yolo.py` — Convert the original SynsetSignsetGermany raw archive (Ogre/Cycles/Labels/Masks) into a YOLOv8 dataset (images + labels). This is the main conversion script that builds masks → bounding boxes and writes YOLO `.txt` labels.

- `convert_traindata_to_image.py` — Reads `resultset.csv` (or converted label outputs) and generates a visualization image (training graph). Output: `traffic-sign-detector/trainingGraph.jpg`.

- `create_data_yaml.py` — Generates `data/yolo_signs/data.yaml` (with relative paths) and optionally writes `classes.txt` from the current mapping.

- `create_val_split.py` — Moves a percentage (default \~10%) of `train/images`+`train/labels` to `val/images` + `val/labels` to create a validation split.

- `validate_yolo_dataset.py` — Quick validator that checks image↔label correspondence and basic YOLO label format. (See file `scripts/validate_yolo_dataset.py`.)

- `test_gpu_support.py` — Small helper to check if PyTorch detects a CUDA GPU and prints device info.

Subfolder `scripts/train_model/` contains training utilities and weights:

- `start_train.py` — Entrypoint to start YOLOv8 training (wrapped for Windows multiprocessing support and integrated with Weights & Biases). Uses `yolov8m.pt` by default. Make sure `start_train.py` is guarded with `if __name__ == '__main__':`.

- `yolov8m.pt` — Pretrained YOLOv8m weights (used as backbone).

- `yolo11n.pt` — Small auxiliary weights sometimes used by helper components (optional tho).

- `README.md` — (this file) — lives at `scripts/README.md`.

---

## ️ Setup (quick)

Run from project root (where `data/` and `scripts/` live).

1. Create & activate virtual environment (Windows Powershell):

```powershell
python -m venv .venv1
.venv1\Scripts\activate
```

2. Install required packages:

```powershell
pip install -U pip
pip install ultralytics wandb pandas opencv-python tqdm scikit-learn
```

3. (Optional) login to W&B once:

```powershell
wandb login
```

---

## Typical workflow

1. **Convert the raw Synset archive to YOLO**

```
python scripts/convert_synset_to_yolo.py `
   --raw data/raw/SynsetSignsetGermany `
   --masks data/raw/SynsetSignsetGermany/Masks `
   --semantic data/raw/SynsetSignsetGermany/Labels `
   --out data/yolo_signs `
   --split 0.8 0.2 0.0
```

This will:

- Index masks/labels and ogre images
- Compute bounding boxes from masks
- Write `data/yolo_signs/train/` and `data/yolo_signs/val/` (images + labels)
- Create `data/yolo_signs/classes.txt`
- run `create_data_yaml.py` next

If you already converted and need to re-run only the dataset formatting, use the same command and a different `--out` if needed.

2. **(Optional) Create a visual training graph image (during training or after)**

```powershell
python scripts/convert_traindata_to_image.py
```

This script reads the result CSV and saves a visualization to `trainingGraph.jpg` (path configurable in the script).

3. **Create or regenerate**

If you prefer `create_data_yaml.py` to ensure relative paths and names are correct:

```powershell
python scripts/create_data_yaml.py
```

`data.yaml` will set `path: .` and `train: train/images` / `val: val/images`.

4. **(Optional) Split train → val**

If you already have all images in `train/` and want to carve out a validation set:

```powershell
python scripts/create_val_split.py
```

This moves files (images + corresponding labels) into `val/` and keeps names synced.

5. **Validate the YOLO dataset**

Run the quick validator to check for missing labels or format issues:

```powershell
python scripts/validate_yolo_dataset.py
```

It prints counts and examples of mismatches and does a basic format validation for the first 200 label files.

6. **Test GPU support (optional)**

```powershell
python scripts/test_gpu_support.py
```

This prints whether PyTorch sees a CUDA device.

7. **Start training**

Run the trainer (from project root):

```powershell
python scripts/train_model/start_train.py
```

Notes:

- `start_train.py` uses `YOLO('yolov8m.pt')` and calls `model.train(...)` with sensible defaults. It initializes a W&B run (if logged in) and is Windows-safe (`multiprocessing.freeze_support()` + `if __name__ == '__main__'`).
- If you experience Windows multiprocessing errors, open `start_train.py` and set `workers=0` in the `model.train()` call.
- TensorBoard is available; the trainer prints the `tensorboard --logdir ...` command in the console. W&B links to the run are printed as well.

---

## Quick troubleshooting

- **Missing **``** error**: Make sure `data/yolo_signs/data.yaml` uses relative paths and is placed inside `data/yolo_signs` (use `path: .`).
- **No labels found**: Run `validate_yolo_dataset.py` to see missing labels → re-run `convert_synset_to_yolo.py` or inspect mask indexing.
- **Windows spawn/multiprocessing error**: Ensure `start_train.py` contains the `if __name__ == '__main__':` guard and `multiprocessing.freeze_support()`, or set `workers=0`.
- **Extra small model downloads (e.g. **``**)**: The trainer may download helper weights for optional modules (tracking). `start_train.py` sets `tracker=None` by default to avoid this.

---

## validate\_yolo\_dataset.py (what it does)

A short summary of the validator included in `scripts/validate_yolo_dataset.py`:

- Scans `data/yolo_signs/train/images` and `data/yolo_signs/train/labels` (and the `val/` counterparts)
- Prints total counts of images and labels
- Lists up to 10 image basenames that are missing labels and up to 10 labels without images
- Performs a quick sanity check of the YOLO label format for the first 200 `.txt` files (5 tokens per line; normalized floats between 0 and 1)

You can open and run the validator directly to get a quick dataset health check.

---
