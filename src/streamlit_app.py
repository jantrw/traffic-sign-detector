# src/streamlit_app.py

import streamlit as st
import tempfile
import os
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
import yaml
import cv2

# CONFIG
MODEL_PATH = "runs/detect/train/weights/best.pt"  # oder yolov8m.pt
DEVICE = "cuda" # oder cpu
CLASS_NAMES_PATH = "data/traffic.yaml"

# SIDEBAR CONFIG
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

# Klassennamen aus traffic.yaml laden
@st.cache_data
def load_class_names(yaml_path=CLASS_NAMES_PATH):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["names"].items()}

# YOLO-Modell laden
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

# Inferenz durchführen
def predict_and_plot(image_path, model, device, conf_thresh, class_names):
    results = model(image_path, device=device, conf=conf_thresh)
    if not results:
        return None, []

    result = results[0]
    annotated = result.plot()

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(conf, 3),
            "bbox": [round(x, 1) for x in xyxy]
        })

    return annotated, detections

# UI
st.title("Traffic Sign Detector")
st.markdown("Upload an image and get traffic signs detected using YOLOv8")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Zeige Originalbild
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Speichere temporär
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Lade Modell & Klassennamen
    with st.spinner("Running inference"):
        model = load_model()
        class_names = load_class_names()
        annotated_img, detections = predict_and_plot(tmp_path, model, DEVICE, confidence_threshold, class_names)

    if annotated_img is not None:
        # BGR (OpenCV) → RGB (für Streamlit)
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.success(f"Prediction complete. Found {len(detections)} object(s).")
        st.image(annotated_rgb, caption="Detected Traffic Signs", width=400)  # Anzeige in RGB

        # JSON-Ausgabe der Detektionen anzeigen
        st.json(detections)

        # Annotiertes Bild zum Download anbieten
        img_pil = Image.fromarray(annotated_rgb)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as out_tmp:
            img_pil.save(out_tmp.name)
            with open(out_tmp.name, "rb") as file:
                st.download_button(
                    label="Download Annotated Image",
                    data=file,
                    file_name="prediction.jpg",
                    mime="image/jpeg"
                )
    else:
        st.warning("No predictions above confidence threshold.")

