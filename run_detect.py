#!/usr/bin/env python3
# run_detect.py
"""
Usage:
  python run_detect.py path/to/image.jpg --model path/to/model.pt
  python run_detect.py path/to/image.jpg --model path/to/model.pt --copy-dest C:/tmp/repo_copy --conf 0.05 --iou 0.4 --imgsz 1024 --save-txt
"""

import argparse
import os
import shutil
import sys
import glob
import time

def newest_run_detect_dir(base="runs/detect"):
    # find newest runs/detect* folder
    candidates = glob.glob(base + "*")
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def copy_repo(dest):
    src = os.getcwd()
    if os.path.exists(dest):
        raise FileExistsError(f"Destination '{dest}' already exists. Choose a non-existing folder")
    print(f"Copying repository from\n  {src}\n to\n  {dest}\nThis can take a while")
    shutil.copytree(src, dest)
    print("Copy finished.")

def main():
    p = argparse.ArgumentParser(description="Run YOLO detect predict from Python (wrapper).")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--model", required=True, help="Path to model .pt")
    p.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (e.g. 0.3)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (e.g. 0.45)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (px)")
    p.add_argument("--copy-dest", default=None, help="Optional: copy current repo to DEST before running")
    p.add_argument("--save-txt", action="store_true", help="Also save detection .txt files")
    args = p.parse_args()

    # basic checks
    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}", file=sys.stderr); sys.exit(2)
    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr); sys.exit(2)

    if args.copy_dest:
        try:
            copy_repo(args.copy_dest)
        except Exception as e:
            print("Failed to copy repo:", e, file=sys.stderr)
            sys.exit(3)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run 'pip install -r requirements.txt'.", file=sys.stderr)
        sys.exit(4)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Running inference on {args.image}  (conf={args.conf}, iou={args.iou}, imgsz={args.imgsz})")
    t0 = time.time()
    results = model.predict(source=args.image,
                            conf=args.conf,
                            iou=args.iou,
                            imgsz=args.imgsz,
                            save=True,
                            save_txt=args.save_txt)
    t = time.time() - t0
    print(f"Done. Inference time: {t:.2f}s")

    out_dir = newest_run_detect_dir()
    if out_dir:
        print("Results saved to:", out_dir)
        # list files written
        for root, _, files in os.walk(out_dir):
            for f in files:
                path = os.path.join(root, f)
                print("  ", os.path.relpath(path))
    else:
        print("No runs/detect output found. Check ultralytics version / permissions")

    # optional: print raw boxes for quick debug (first result)
    try:
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            print("Model returned NO boxes for this image")
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            print("Detections:")
            for i,(b,c,cl) in enumerate(zip(xyxy, confs, clss)):
                name = model.names.get(int(cl), str(int(cl)))
                print(f"  #{i}: {name}  conf={c:.3f}  box={b}")
    except Exception as e:
        print("Could not print raw boxes:", e)

if __name__ == "__main__":
    main()
