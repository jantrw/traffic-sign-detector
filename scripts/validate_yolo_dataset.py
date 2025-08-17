# scripts/validate_yolo_dataset.py
from pathlib import Path
import sys

ROOT = Path("data/yolo_signs")
train_img = ROOT / "train" / "images"
train_lbl = ROOT / "train" / "labels"
val_img = ROOT / "val" / "images"
val_lbl = ROOT / "val" / "labels"

def stats(img_dir, lbl_dir):
    imgs = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in (".png",".jpg",".jpeg")])
    lbls = sorted([p for p in lbl_dir.rglob("*.txt")])
    print(f"{img_dir} -> images: {len(imgs)}, labels: {len(lbls)}")
    # find images without label
    img_basenames = {p.stem for p in imgs}
    lbl_basenames = {p.stem for p in lbls}
    no_label = sorted(list(img_basenames - lbl_basenames))[:10]
    no_image = sorted(list(lbl_basenames - img_basenames))[:10]
    print(" samples without label (first 10):", no_label)
    print(" labels without image (first 10):", no_image)
    # quick label format check (first 200 labels)
    bad = 0
    for i,p in enumerate(lbls):
        if i>200: break
        try:
            for line in p.read_text(encoding="utf8").splitlines():
                parts = line.strip().split()
                if not parts: continue
                if len(parts)!=5:
                    bad += 1; break
                # class_id = int(parts[0])  # optionally check range later
                nums = list(map(float, parts[1:]))
                if not all(0.0 <= x <= 1.0 for x in nums):
                    bad += 1; break
        except Exception:
            bad += 1
    print(" label format issues in first 200 files:", bad)

stats(train_img, train_lbl)
stats(val_img, val_lbl)
