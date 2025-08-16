# src/infer.py

import argparse
import os
from ultralytics import YOLO
from PIL import Image

def predict(model_path, image_path, device="cpu", output_dir="output"):
    model = YOLO(model_path)
    results = model(image_path, device=device)

    # Output-Ordner erstellen
    os.makedirs(output_dir, exist_ok=True)

    # Annotiertes Bild speichern
    for i, result in enumerate(results):
        result_path = os.path.join(output_dir, f"prediction_{i}.jpg")
        annotated_frame = result.plot()  # returns np.ndarray
        im = Image.fromarray(annotated_frame)
        im.save(result_path)
        print(f"[INFO] Saved: {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on an image")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model (.pt file)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--output", type=str, default="output", help="Directory to save predictions")

    args = parser.parse_args()
    predict(args.model, args.image, args.device, args.output)
