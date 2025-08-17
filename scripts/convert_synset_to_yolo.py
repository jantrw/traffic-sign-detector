#!/usr/bin/env python3
"""
Robuster & schneller Konverter SynsetSignsetGermany -> YOLO (Masken-basiert).
Speichert dataset in data/yolo_signs/{train,val}/{images,labels}.
"""
import os
import random
import shutil
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
import cv2
import numpy as np

# ---------- konfig ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "SynsetSignsetGermany"
OGRE_DIR = RAW_ROOT / "Ogre"
MASKS_DIR = RAW_ROOT / "Masks"
LABELS_DIR = RAW_ROOT / "Labels"
CSV_DIR = RAW_ROOT / "CsvFiles"
OUT_ROOT = PROJECT_ROOT / "data" / "yolo_signs"
OUT_TRAIN_IMG = OUT_ROOT / "train" / "images"
OUT_TRAIN_LBL = OUT_ROOT / "train" / "labels"
OUT_VAL_IMG = OUT_ROOT / "val" / "images"
OUT_VAL_LBL = OUT_ROOT / "val" / "labels"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MIN_CONTOUR_AREA = 50
TRAIN_RATIO_IF_NO_CSV = 0.8
SEED = 42
# ----------------------------

random.seed(SEED)
np.random.seed(SEED)

for p in (OUT_TRAIN_IMG, OUT_TRAIN_LBL, OUT_VAL_IMG, OUT_VAL_LBL):
    p.mkdir(parents=True, exist_ok=True)

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def discover_class_folders(ogre_dir: Path) -> List[str]:
    if not ogre_dir.exists():
        return []
    folders = [p.name for p in ogre_dir.iterdir() if p.is_dir()]
    return sorted(folders, key=lambda x: int(x) if x.isdigit() else x)

def build_mask_index(mask_roots: List[Path]) -> Dict[str, List[Path]]:
    """
    Scannt alle Mask-Roots und baut ein index:
      key -> list(paths)
    Keys: stem, stem without _msk/_mask/_seg, also with leading zeros stripped.
    """
    idx: Dict[str, List[Path]] = {}
    suffix_variants = ["", "_msk", "_mask", "_seg", "-msk", "-mask"]
    for root in mask_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                continue
            stem = p.stem
            # add original stem
            idx.setdefault(stem, []).append(p)
            # add stripped variants
            for suf in suffix_variants:
                if suf and stem.endswith(suf):
                    base = stem[: -len(suf)]
                    idx.setdefault(base, []).append(p)
            # add zero-stripped variant (e.g. '0001' -> '1')
            try:
                if stem.isdigit():
                    idx.setdefault(str(int(stem)), []).append(p)
            except Exception:
                pass
    return idx

def find_masks_for_image_fast(class_folder: str, img_basename: str, mask_roots: List[Path], mask_index: Dict[str, List[Path]]) -> List[Path]:
    """
    Robust match: aus '0_ogre' -> token '0' erzeugen, dann candidate stems
      ['0', '0_msk', '0_mask', '0_ogre', '0_ogre_msk', ..]
    Prüft zuerst Masks/<class_folder>/*, dann global in mask_roots, dannn mask_index fallback.
    Liefert Listte von Path (kann leer sein)
    """
    candidates: List[Path] = []

    # token extraction: leading digits preferred, sonst first token before '_'
    m = re.match(r"^(\d+)", img_basename)
    num_token = m.group(1) if m else img_basename.split("_")[0]

    # build ordered stems to try
    stems = []
    stems.append(num_token)
    stems.append(f"{num_token}_msk")
    stems.append(f"{num_token}_mask")
    stems.append(img_basename)
    stems.append(f"{img_basename}_msk")
    stems.append(f"{img_basename}_mask")
    # remove duplicates while preserving order
    seen = set()
    stems = [s for s in stems if not (s in seen or seen.add(s))]

    # check each mask root -> prefer class folder inside it
    for root in mask_roots:
        if not root or not Path(root).exists():
            continue
        class_dir = Path(root) / class_folder
        # 1) try class folder
        for stem in stems:
            for ext in IMG_EXTS:
                p = class_dir / (stem + ext)
                if p.exists():
                    candidates.append(p)
        # 2) try root top-level (global)
        if not candidates:
            for stem in stems:
                for ext in IMG_EXTS:
                    p = Path(root) / (stem + ext)
                    if p.exists():
                        candidates.append(p)

    # fallback to mask_index (if available) for stems
    if not candidates and mask_index:
        for stem in stems:
            if stem in mask_index:
                candidates.extend(mask_index[stem])
        # also try numeric token key (e.g. '0')
        if num_token in mask_index:
            candidates.extend(mask_index.get(num_token, []))

    # deduplicate preserving order
    uniq = []
    seen_fp = set()
    for p in candidates:
        try:
            rp = str(Path(p).resolve())
        except Exception:
            rp = str(p)
        if rp not in seen_fp:
            uniq.append(Path(p))
            seen_fp.add(rp)
    return uniq

def mask_to_bboxes(mask_path: Path, min_area: int = MIN_CONTOUR_AREA):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return []
    _, th = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))
    return boxes

def read_csv_splits(csv_dir: Path):
    if not csv_dir.exists():
        return None, None
    train_set, val_set = set(), set()
    found = False
    import pandas as pd
    for f in csv_dir.glob("*.csv"):
        n = f.name.lower()
        try:
            df = pd.read_csv(f, engine="python", header=None)
            vals = df.values.flatten()
            basenames = [os.path.splitext(os.path.basename(str(v)))[0] for v in vals if str(v).strip()]
            if "train" in n:
                train_set.update(basenames); found = True
            if "val" in n:
                val_set.update(basenames); found = True
        except Exception:
            continue
    return (train_set, val_set) if found else (None, None)

def main():
    # Build mask roots
    mask_roots: List[Path] = []
    if MASKS_DIR.exists():
        mask_roots.append(MASKS_DIR)
    if LABELS_DIR.exists():
        mask_roots.append(LABELS_DIR)
    # also check common alternate location
    alternate = RAW_ROOT / "MasksOnly" / "Masks"
    if alternate.exists():
        mask_roots.append(alternate)
    print(f"[INFO] mask roots: {mask_roots}")

    mask_index = build_mask_index(mask_roots)
    print(f"[INFO] Indexed ~{sum(len(v) for v in mask_index.values())} mask files under {len(mask_index)} keys")

    # class folders from OGRE
    class_folders = discover_class_folders(OGRE_DIR)
    class_map = {folder: idx for idx, folder in enumerate(class_folders)}
    print(f"[INFO] Found {len(class_folders)} classes (OGRE). Example: {list(class_map.items())[:8]}")

    # read csv split if present
    train_csv, val_csv = read_csv_splits(CSV_DIR)
    if train_csv is not None:
        print(f"[INFO] Using CSV split: {len(train_csv)} train entries, {len(val_csv)} val entries")

    labels_acc: Dict[str, Tuple[Path, List[str], int]] = {}
    skipped_info = []  # collect example skips (reason, img_path)
    # iterate images
    total_images = 0
    for cls in tqdm(class_folders, desc="Scanning classes"):
        cls_dir = OGRE_DIR / cls
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.iterdir():
            if not img_path.is_file() or not is_image(img_path):
                continue
            total_images += 1
            basename = img_path.stem
            # UNIQUE key to avoid collisions across classes
            out_key = f"{int(class_map.get(cls,0)):03d}_{cls}_{basename}"

            # robust mask-finding using find_masks_for_image_fast()
            candidates = find_masks_for_image_fast(cls, basename, mask_roots, mask_index)

            # deduplicate candidate list (safety)
            cand_unique = []
            seen_paths = set()
            for c in candidates:
                try:
                    rp = str(Path(c).resolve())
                except Exception:
                    rp = str(c)
                if rp not in seen_paths:
                    cand_unique.append(c)
                    seen_paths.add(rp)

            if not cand_unique:
                # no mask found
                if len(skipped_info) < 20:
                    skipped_info.append(("no_mask_found", str(img_path)))
                continue

            # load image once
            img = cv2.imread(str(img_path))
            if img is None:
                if len(skipped_info) < 20:
                    skipped_info.append(("img_read_failed", str(img_path)))
                continue
            h, w = img.shape[:2]

            all_lines = []
            any_box = False
            for mpath in cand_unique:
                boxes = mask_to_bboxes(mpath, MIN_CONTOUR_AREA)
                if not boxes:
                    # mask exists but no contours (maybe very small)
                    if len(skipped_info) < 20:
                        skipped_info.append(("mask_no_contours", f"{img_path} <- {mpath}"))
                    continue
                # create label lines
                for (x1,y1,x2,y2) in boxes:
                    x_c = (x1 + x2) / 2.0
                    y_c = (y1 + y2) / 2.0
                    bw = (x2 - x1)
                    bh = (y2 - y1)
                    xcn = x_c / w
                    ycn = y_c / h
                    bwn = bw / w
                    bhn = bh / h
                    class_id = class_map.get(cls, 0)
                    line = f"{class_id} {xcn:.6f} {ycn:.6f} {bwn:.6f} {bhn:.6f}"
                    all_lines.append(line)
                    any_box = True

            if any_box:
                labels_acc[out_key] = (img_path, all_lines, class_map.get(cls, 0))

    print(f"[INFO] Scanned {total_images} images. Collected labels for {len(labels_acc)} images.")
    if skipped_info:
        print("[INFO] Beispiele für übersprungene Items (erst 20):")
        for reason, p in skipped_info:
            print(f" - {reason}: {p}")

    # Build train/val split
    all_keys = sorted(list(labels_acc.keys()))
    if train_csv is not None:
        # map original basenames to out_keys
        orig_to_keys = {}
        for k, (imgp, lines, cid) in labels_acc.items():
            orig = imgp.stem
            orig_to_keys.setdefault(orig, []).append(k)
        train_keys, val_keys = set(), set()
        for b in train_csv:
            train_keys.update(orig_to_keys.get(b, []))
        for b in val_csv:
            val_keys.update(orig_to_keys.get(b, []))
        remaining = set(all_keys) - train_keys - val_keys
        rem = list(remaining)
        random.shuffle(rem)
        n_train = int(len(rem) * TRAIN_RATIO_IF_NO_CSV)
        train_keys.update(rem[:n_train])
        val_keys.update(rem[n_train:])
    else:
        random.shuffle(all_keys)
        split_idx = int(len(all_keys) * TRAIN_RATIO_IF_NO_CSV)
        train_keys = set(all_keys[:split_idx])
        val_keys = set(all_keys[split_idx:])

    # write files
    for k in tqdm(all_keys, desc="Writing files"):
        imgp, lines, cid = labels_acc[k]
        ext = imgp.suffix.lower()
        if k in train_keys:
            dst_img = OUT_TRAIN_IMG / (k + ext)
            dst_lbl = OUT_TRAIN_LBL / (k + ".txt")
        else:
            dst_img = OUT_VAL_IMG / (k + ext)
            dst_lbl = OUT_VAL_LBL / (k + ".txt")
        # copy image (fast)
        shutil.copy2(imgp, dst_img)
        dst_lbl.write_text("\n".join(lines), encoding="utf8")

    # write classes.txt (simple: class map by folder order)
    classes_txt = OUT_ROOT / "classes.txt"
    id_to_name = {v:k for k,v in class_map.items()}
    with open(classes_txt, "w", encoding="utf8") as f:
        for i in range(len(class_map)):
            name = id_to_name.get(i, f"class_{i:03d}")
            f.write(name + "\n")
    print("[DONE] Conversion finished.")

if __name__ == "__main__":
    main()
