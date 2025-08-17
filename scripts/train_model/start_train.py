# scripts/train_model/start_train.py
import multiprocessing
from pathlib import Path
from ultralytics import YOLO
import wandb
import os

def main():

    # working dir: project root (wo 'data/' liegt)
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)

    run = wandb.init(
        project="synset_signset_germany",
        name="yolov8m_run_01",
        config={
            "model": "yolov8m.pt",
            "data": "data/yolo_signs/data.yaml",
            "epochs": 50,
            "imgsz": 640,
            "batch": 16
        },
        reinit=True
    )
    try:
        print("W&B Run URL:", wandb.run.url)

        model = YOLO("yolov8m.pt")

        # Train starten  tracker=None verhindert gewöhnlich zusätzliche Downloads (z. B. yolo11n)
        model.train(
            data="data/yolo_signs/data.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            name="yolov8m_run_01",
            project="synset_signset_germany",
            tracker=None,    # deaktiviert tracker
        )

    finally:
        wandb.finish()
        print("W&B finished. Check run at:", run.url)


if __name__ == "__main__":
    # wichtig auf Windows vor spawn-basierten Prozessen
    multiprocessing.freeze_support()
    main()
