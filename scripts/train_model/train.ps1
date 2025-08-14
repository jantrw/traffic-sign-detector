# train.ps1
# training scrript for YOLOv8 on GTSRB dataset
# my gpu NVIDIA RTX 3070 8GB VRAM

Write-Host "Start YOLOv8 training" -ForegroundColor Green

# Parameter
# model: Start mit YOLOv8m (mittelgroß), gute Balance Speed/Accuracy
# imgsz: 640 (Standard für YOLO) using 512, GPU was at 100%
# batch: auto -> passt Batchgröße automatisch an VRAM an
# epochs: 100 (lang genug für gute Genauigkeit, anpassbar) using 50 GPU was at 100%
# device: 0 -> nutzt GPU
# workers: 8 -> nutzt 8 CPU-Threads fürs Laden der Daten

yolo detect train `
    model=yolov8m.pt `
    data=data/traffic.yaml `
    imgsz=416 `
    epochs=50 `
    batch=8 `
    #nutzt gpu
    device=0 `
    workers=8 `
    optimizer=AdamW `
    patience=20 `
    save=True `
    plots=True `
    tensorboard=True `
    project=runs

Write-Host "Training done" -ForegroundColor Green
