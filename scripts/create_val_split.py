import os
import random
import shutil
from pathlib import Path

# Basisverzeichnis
base_dir = Path("traffic-sign-detector/data/yolo_signs")

# Ordner definieren
train_img_dir = base_dir / "train/images"
train_lbl_dir = base_dir / "train/labels"
val_img_dir = base_dir / "val/images"
val_lbl_dir = base_dir / "val/labels"

# Val-Ordner anlegen
val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# Alle Trainingsbilder sammeln
images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
print(f"Gefundene Trainingsbilder: {len(images)}")

# Anteil für Validation
val_ratio = 0.1   # 10%
val_count = int(len(images) * val_ratio)
print(f"-> Verschiebe {val_count} Bilder nach val/")

# Zufällige Auswahl
random.seed(42)  # für Reproduzierbarkeit
val_images = random.sample(images, val_count)

for img_path in val_images:
    # Bild verschieben
    shutil.move(str(img_path), str(val_img_dir / img_path.name))

    # Label verschieben
    lbl_path = train_lbl_dir / (img_path.stem + ".txt")
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(val_lbl_dir / lbl_path.name))

print(f"{val_count} Bilder + Labels nach val/ verschoben")
print(f"Train übrig: {len(list(train_img_dir.glob('*')))} Bilder")
print(f"Val jetzt: {len(list(val_img_dir.glob('*')))} Bilder")
