import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_csv_path = os.path.join(script_dir, "..",  "synset_signset_germany", "yolov8m_run_013", "results.csv")
csv_path = os.path.abspath(relative_csv_path)

# CSV einlesen
df = pd.read_csv(csv_path)
epochs = df["epoch"]

#Bild mit zwei nebeneinander liegenden Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Trainings- und Validierungsverluste
ax1.plot(epochs, df["train/box_loss"], label="Train Box Loss")
ax1.plot(epochs, df["train/cls_loss"], label="Train Class Loss")
ax1.plot(epochs, df["train/dfl_loss"], label="Train DFL Loss")
ax1.plot(epochs, df["val/box_loss"], label="Val Box Loss", linestyle="--")
ax1.plot(epochs, df["val/cls_loss"], label="Val Class Loss", linestyle="--")
ax1.plot(epochs, df["val/dfl_loss"], label="Val DFL Loss", linestyle="--")
ax1.set_title("Losses pro Epoche")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

# Subplot 2: Metriken
ax2.plot(epochs, df["metrics/precision(B)"], label="Precision")
ax2.plot(epochs, df["metrics/recall(B)"], label="Recall")
ax2.plot(epochs, df["metrics/mAP50(B)"], label="mAP@50")
ax2.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@50-95")
ax2.set_title("Metriken pro Epoche")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Wert")
ax2.legend()
ax2.grid(True)

# Layout & Speichern
plt.tight_layout()
output_image_path = os.path.join(script_dir, "..", "TrainingResults.jpg")
plt.savefig(output_image_path)
plt.show()
