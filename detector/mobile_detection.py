"""
mobile_detection.py
--------------------
Runs YOLOv8 inference to detect mobile phones in a video frame.

The model is loaded once at import time (lazy on first call) to avoid
re-loading on every frame.  Detection results include bounding boxes,
confidence scores, and a convenience flag indicating whether a phone was found.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# Path to the custom-trained YOLOv8 weights
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "best_yolov8.pt"
)

# YOLO class index for 'cell phone' in the COCO dataset (used as fallback)
COCO_PHONE_CLASS_ID = 67

# Minimum confidence threshold to count as a detection
CONFIDENCE_THRESHOLD = 0.40

# Module-level model holder (loaded once)
_model = None


def _load_model() -> YOLO:
    """
    Load and return the YOLOv8 model.  Uses the custom weights when available,
    falls back to the pretrained YOLOv8n COCO model otherwise.

    Returns
    -------
    YOLO – Loaded Ultralytics YOLO instance.
    """
    global _model
    if _model is not None:
        return _model

    if os.path.exists(MODEL_PATH):
        _model = YOLO(MODEL_PATH)
        print(f"[MobileDetector] Loaded custom weights: {MODEL_PATH}")
    else:
        # Fallback: use the pretrained nano model from Ultralytics hub
        _model = YOLO("yolov8n.pt")
        print(
            "[MobileDetector] Custom weights not found – using pretrained YOLOv8n "
            "(only COCO 'cell phone' class will be used)."
        )

    return _model


def detect_phones(frame: np.ndarray):
    """
    Run phone detection on a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray – BGR video frame from OpenCV.

    Returns
    -------
    tuple[bool, list[dict]]
        phone_detected : bool – True if at least one phone was found.
        detections     : list[dict] – Each entry has keys:
                          'bbox'       – (x1, y1, x2, y2) in pixels
                          'confidence' – float in [0, 1]
                          'label'      – str class name
    """
    model = _load_model()

    results = model(frame, verbose=False)[0]

    detections = []
    phone_detected = False

    for box in results.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        # Accept if custom model (any class) or COCO phone class
        is_custom_model = os.path.exists(MODEL_PATH)
        is_phone_class = is_custom_model or (class_id == COCO_PHONE_CLASS_ID)

        if is_phone_class and confidence >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = results.names.get(class_id, "phone")
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "label": label,
                }
            )
            phone_detected = True

    return phone_detected, detections


def draw_phone_boxes(frame: np.ndarray, detections: list) -> None:
    """
    Draw bounding boxes and confidence labels for detected phones on the frame
    in-place.

    Parameters
    ----------
    frame      : np.ndarray – BGR video frame.
    detections : list[dict] – Output from detect_phones().
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        confidence = det["confidence"]
        label = det["label"]

        # Red bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Label background
        label_text = f"{label} {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), (0, 0, 255), -1
        )
        cv2.putText(
            frame,
            label_text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
