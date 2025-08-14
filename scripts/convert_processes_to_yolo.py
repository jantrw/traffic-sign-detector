#!/usr/bin/env python3
"""
ChatGPT 5 created

convert_processed_to_yolo.py

Reads Train.csv and Test.csv from data/raw/gtsrb ,
walks through data/processed/{train,val,test}/{class_id}/... images,
matches each image to the CSV row (by basename), converts bbox to YOLO format,
copies image to data/{train,val,test}/images and writes corresponding .txt in
data/{train,val,test}/labels.

Creates data/traffic.yaml with generic class names (class_0..class_42).

Run from project root:
    python scripts/convert_processed_to_yolo.py
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd

# --- CONFIG (anpassen falls nötig) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # assumes script in scripts/ or src/
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gtsrb"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
YOLO_BASE = PROJECT_ROOT / "data"   # results will be in data/train, data/val, data/test
NUM_CLASSES = 43
# --------------------------------------

TRAIN_CSV_FILENAME = "Train.csv"
TEST_CSV_FILENAME = "Test.csv"

SPLITS = ["train", "val", "test"]

def find_csv_file(base_path: Path, filename: str) -> Path:
    """Search recursively for filename under base_path. Return Path or raise"""
    for p in base_path.rglob(filename):
        return p
    raise FileNotFoundError(f"{filename} not found under {base_path}")

def load_annotations():
    """Load Train.csv and Test.csv into a single DataFrame (adds a 'split' column)."""
    # Try to find files
    try:
        train_csv = find_csv_file(RAW_DATA_PATH, TRAIN_CSV_FILENAME)
    except FileNotFoundError:
        print(f"[WARN] {TRAIN_CSV_FILENAME} not found under {RAW_DATA_PATH}. Continuing without it.")
        train_csv = None

    try:
        test_csv = find_csv_file(RAW_DATA_PATH, TEST_CSV_FILENAME)
    except FileNotFoundError:
        print(f"[WARN] {TEST_CSV_FILENAME} not found under {RAW_DATA_PATH}. Continuing without it.")
        test_csv = None

    dfs = []
    if train_csv is not None:
        df_train = pd.read_csv(train_csv)
        df_train["__source_csv"] = str(train_csv)
        dfs.append(df_train)
    if test_csv is not None:
        df_test = pd.read_csv(test_csv)
        df_test["__source_csv"] = str(test_csv)
        dfs.append(df_test)

    if not dfs:
        raise FileNotFoundError("No Train.csv or Test.csv found. Place them under data/raw/gtsrb/")

    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    # Normalize column names if there are spaces or different cases
    df_all.columns = [c.strip() for c in df_all.columns]
    required = {"Width","Height","Roi.X1","Roi.Y1","Roi.X2","Roi.Y2","ClassId","Path"}
    if not required.issubset(set(df_all.columns)):
        print("[ERROR] CSV missing required columns. Found columns:", df_all.columns.tolist())
        raise SystemExit(1)

    # Create a lookup map basename -> list of rows (as dicts)
    lookup = {}
    for _, row in df_all.iterrows():
        path_val = str(row["Path"])
        basename = os.path.basename(path_val)
        if basename not in lookup:
            lookup[basename] = []
        lookup[basename].append(row.to_dict())
    return lookup

def ensure_yolo_dirs():
    for split in SPLITS:
        (YOLO_BASE / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_BASE / split / "labels").mkdir(parents=True, exist_ok=True)

def convert_and_copy(lookup):
    summary = {s: 0 for s in SPLITS}
    unmatched = []
    multiple_matches = []

    for split in SPLITS:
        split_processed = PROCESSED_PATH / split
        if not split_processed.exists():
            print(f"[WARN] processed split folder not found: {split_processed} (skipping)")
            continue

        # Walk class_id subfolders
        for class_dir in sorted(split_processed.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.iterdir():
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}:
                    continue

                img_basename = img_path.name  # e.g., 000005_000003_000010.png
                # attempt direct basename match in CSV lookup
                rows = lookup.get(img_basename, None)

                if not rows:
                    # No match by basename: try matching by basename without extension
                    name_no_ext = img_path.stem
                    possible = []
                    for k in lookup.keys():
                        if k.startswith(name_no_ext) or name_no_ext.startswith(Path(k).stem):
                            possible.extend(lookup[k])
                    if possible:
                        rows = possible

                if not rows:
                    unmatched.append(str(img_path))
                    continue

                # If multiple rows match, we will write multiple bbox lines
                if len(rows) > 1:
                    multiple_matches.append((str(img_path), len(rows)))

                # prepare target paths
                dst_img = YOLO_BASE / split / "images" / img_path.name
                dst_lab = YOLO_BASE / split / "labels" / (img_path.stem + ".txt")
                # copy image (overwrite if exists)
                shutil.copy2(img_path, dst_img)

                # write label file (one line per matching row)
                lines = []
                for r in rows:
                    try:
                        img_w = float(r["Width"])
                        img_h = float(r["Height"])
                        x1 = float(r["Roi.X1"])
                        y1 = float(r["Roi.Y1"])
                        x2 = float(r["Roi.X2"])
                        y2 = float(r["Roi.Y2"])
                        cls = int(r["ClassId"])
                    except Exception as e:
                        print(f"[ERROR] malformed row for {img_path}: {e}")
                        continue

                    # compute YOLO format (normalized)
                    x_center = (x1 + x2) / 2.0 / img_w
                    y_center = (y1 + y2) / 2.0 / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h

                    # clamp values to [0,1]
                    def clamp(v):
                        return max(0.0, min(1.0, float(v)))
                    x_center = clamp(x_center)
                    y_center = clamp(y_center)
                    bw = clamp(bw)
                    bh = clamp(bh)

                    lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

                with open(dst_lab, "w", encoding="utf8") as f:
                    f.write("\n".join(lines))

                summary[split] += 1

    return summary, unmatched, multiple_matches

def write_traffic_yaml():
    yaml_path = PROJECT_ROOT / "data" / "traffic.yaml"
    names = {i: f"class_{i}" for i in range(NUM_CLASSES)}
    content_lines = [
        f"train: { (Path('data') / 'train' / 'images').as_posix() }",
        f"val:   { (Path('data') / 'val' / 'images').as_posix() }",
        "names:"
    ]
    for k in range(NUM_CLASSES):
        content_lines.append(f"  {k}: {names[k]}")
    yaml_text = "\n".join(content_lines) + "\n"
    with open(yaml_path, "w", encoding="utf8") as f:
        f.write(yaml_text)
    print(f"[INFO] Wrote {yaml_path}")

def main():
    print("Starting conversion: processed -> YOLO format")
    print(f"RAW_DATA_PATH = {RAW_DATA_PATH}")
    print(f"PROCESSED_PATH = {PROCESSED_PATH}")
    print(f"Output YOLO base = {YOLO_BASE}")

    # load CSV annotations
    lookup = load_annotations()

    # prepare YOLO directories
    ensure_yolo_dirs()

    # run conversion
    summary, unmatched, multiple_matches = convert_and_copy(lookup)

    # write yaml
    write_traffic_yaml()

    print("\nSummary:")
    for s in SPLITS:
        print(f"  {s}: {summary.get(s,0)} images processed")

    if multiple_matches:
        print("\n[WARN] Some images had multiple matching annotation rows (wrote multiple bbox lines):")
        for im, cnt in multiple_matches[:20]:
            print(f"  {im} -> {cnt} rows")
        if len(multiple_matches) > 20:
            print(f"  ... and {len(multiple_matches)-20} more")

    if unmatched:
        print("\n[WARN] Could not find CSV annotation for the following images (showing up to 50):")
        for im in unmatched[:50]:
            print("  " + im)
        print(f"\nTotal unmatched: {len(unmatched)}")
        print("If many are unmatched, check that the CSV 'Path' basename matches the image filenames in processed folder.")
        print("Possible fixes:")
        print(" - Ensure CSV filenames and processed image basenames are identical.")
        print(" - If CSV uses different naming, tell me and I will adapt the matching logic.")
    else:
        print("\nAll processed images matched to CSV rows successfully.")

if __name__ == "__main__":
    main()
