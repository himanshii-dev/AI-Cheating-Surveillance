"""
app.py
------
AI Cheating Surveillance – Flask Application Entry Point

Starts the webcam capture loop, runs every detector module each frame,
manages suspicion scoring, screenshot saving, logging, and database writes.
Streams the annotated video to the browser via MJPEG and exposes a JSON
state endpoint consumed by the dashboard JavaScript.
"""

import os
import time
import threading
import json
from datetime import datetime

import cv2
import dlib
import numpy as np
from flask import Flask, Response, render_template, jsonify

from detector.database import init_db, insert_violation, fetch_recent_violations, fetch_all_violations
from detector.head_pose import get_head_pose_direction, draw_pose_annotation
from detector.eye_movement import get_eye_status
from detector.mobile_detection import detect_phones, draw_phone_boxes
from detector.suspicion_score import compute_score, is_suspicious, ALERT_THRESHOLD

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "static", "screenshots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "violations.txt")
LANDMARK_MODEL = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# dlib setup
# ---------------------------------------------------------------------------
face_detector = dlib.get_frontal_face_detector()

landmark_predictor = None
if os.path.exists(LANDMARK_MODEL):
    landmark_predictor = dlib.shape_predictor(LANDMARK_MODEL)
    print(f"[App] dlib landmark predictor loaded: {LANDMARK_MODEL}")
else:
    print(
        "[App] WARNING: shape_predictor_68_face_landmarks.dat not found. "
        "Head pose and eye movement detection will be disabled."
    )

# ---------------------------------------------------------------------------
# Shared state (updated by the background capture thread, read by routes)
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
shared_state = {
    "head_direction": "N/A",
    "eye_status": "N/A",
    "phone_detected": False,
    "face_count": 0,
    "suspicion_score": 0,
    "alert": False,
    "active_violations": [],
    "recent_violations": [],
}

# Frame buffer for MJPEG streaming
frame_lock = threading.Lock()
output_frame: np.ndarray = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_violation(violation_type: str, score: int, screenshot_path: str = None):
    """
    Write a violation entry to violations.txt and insert it into SQLite.

    Parameters
    ----------
    violation_type  : str  – Label for the violation.
    score           : int  – Suspicion score at detection time.
    screenshot_path : str  – Relative web path of the screenshot or None.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] | Type: {violation_type} | Score: {score} | Screenshot: {screenshot_path or 'None'}\n"

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

    insert_violation(violation_type, score, screenshot_path)


def save_screenshot(frame: np.ndarray, prefix: str) -> str:
    """
    Save a frame to the screenshots directory.

    Parameters
    ----------
    frame  : np.ndarray – BGR frame to save.
    prefix : str        – Filename prefix (e.g. "phone", "no_face").

    Returns
    -------
    str – Relative web path: "screenshots/<filename>".
    """
    filename = f"{prefix}_{int(time.time())}.jpg"
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    cv2.imwrite(filepath, frame)
    return f"screenshots/{filename}"


# ---------------------------------------------------------------------------
# Per-violation cooldown tracker (avoids spamming the DB on every frame)
# ---------------------------------------------------------------------------
_last_logged: dict[str, float] = {}
VIOLATION_COOLDOWN_SECONDS = 5  # only re-log the same type every N seconds


def should_log(violation_type: str) -> bool:
    """Return True if enough time has elapsed since this violation was last logged."""
    now = time.time()
    last = _last_logged.get(violation_type, 0)
    if now - last >= VIOLATION_COOLDOWN_SECONDS:
        _last_logged[violation_type] = now
        return True
    return False


# ---------------------------------------------------------------------------
# Main capture and processing loop
# ---------------------------------------------------------------------------

def capture_and_process():
    """
    Background thread: opens the webcam, runs all detectors every frame,
    updates shared_state, and populates output_frame for MJPEG streaming.
    """
    global output_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[App] ERROR: Cannot open webcam. Make sure a camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    suspicion_score = 0
    screenshot_cooldown: dict[str, float] = {}
    SCREENSHOT_COOLDOWN_SECONDS = 10

    def _screenshot_ready(key: str) -> bool:
        now = time.time()
        if now - screenshot_cooldown.get(key, 0) >= SCREENSHOT_COOLDOWN_SECONDS:
            screenshot_cooldown[key] = now
            return True
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)  # mirror for natural webcam feel
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Face detection with dlib ------------------------------------
        faces = face_detector(gray, 1)
        print(f"[DEBUG] faces found: {len(faces)}, frame shape: {gray.shape}, predictor loaded: {landmark_predictor is not None}")
        face_count = len(faces)

        head_direction = "N/A"
        eye_status = "N/A"

        if landmark_predictor is not None and face_count == 1:
            landmarks = landmark_predictor(gray, faces[0])
            head_direction = get_head_pose_direction(landmarks, w, h)
            eye_status = get_eye_status(landmarks, gray)
            draw_pose_annotation(frame, head_direction)
        elif face_count == 0:
            cv2.putText(frame, "No Face Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif face_count > 1:
            cv2.putText(frame, f"Multiple Faces: {face_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw dlib face rectangles
        for face_rect in faces:
            x1, y1, x2, y2 = (face_rect.left(), face_rect.top(),
                               face_rect.right(), face_rect.bottom())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---- Phone detection with YOLOv8 ---------------------------------
        phone_detected, phone_detections = detect_phones(frame)
        draw_phone_boxes(frame, phone_detections)

        # ---- Suspicion scoring -------------------------------------------
        suspicion_score, active_violations = compute_score(
            suspicion_score, phone_detected, head_direction, eye_status, face_count
        )
        alert = is_suspicious(suspicion_score)

        # ---- Overlay: score and alert ------------------------------------
        score_color = (0, 255, 0) if not alert else (0, 0, 255)
        cv2.putText(frame, f"Score: {suspicion_score}/{ALERT_THRESHOLD}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

        if alert:
            cv2.putText(frame, "!! SUSPICIOUS ACTIVITY DETECTED !!",
                        (w // 2 - 220, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # ---- Logging and screenshots ------------------------------------
        for vtype in active_violations:
            if should_log(vtype):
                screenshot_path = None

                # Decide whether to take a screenshot for this violation
                take_shot = False
                if vtype == "Phone Detected":
                    take_shot = _screenshot_ready("phone")
                elif vtype == "No Face Detected":
                    take_shot = _screenshot_ready("no_face")
                elif vtype.startswith("Multiple Faces"):
                    take_shot = _screenshot_ready("multi_face")
                elif alert:
                    take_shot = _screenshot_ready("high_score")

                if take_shot:
                    prefix = vtype.lower().replace(" ", "_")[:15]
                    screenshot_path = save_screenshot(frame, prefix)

                log_violation(vtype, suspicion_score, screenshot_path)

        # ---- Update shared state ----------------------------------------
        recent = fetch_recent_violations(10)
        recent_list = [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "violation_type": r["violation_type"],
                "score": r["score"],
                "screenshot_path": r["screenshot_path"],
            }
            for r in recent
        ]
        
    

        with state_lock:
            shared_state.update(
                {
                    "head_direction": head_direction,
                    "eye_status": eye_status,
                    "phone_detected": phone_detected,
                    "face_count": face_count,
                    "suspicion_score": suspicion_score,
                    "alert": alert,
                    "active_violations": active_violations,
                    "recent_violations": recent_list,
                }
            )

        # ---- Write to frame buffer for streaming ------------------------
        with frame_lock:
            output_frame = frame.copy()


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

def generate_mjpeg():
    """
    Generator yielding MJPEG frames for the /video_feed endpoint.
    Encodes output_frame to JPEG and wraps it in the multipart boundary.
    """
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            _, jpeg = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = jpeg.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.033)  # ~30 fps cap


@app.route("/")
def index():
    """Render the main surveillance dashboard."""
    return render_template("index.html", alert_threshold=ALERT_THRESHOLD)


@app.route("/video_feed")
def video_feed():
    """Live MJPEG stream of the annotated webcam feed."""
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/state")
def state():
    """Return current detector state as JSON for dashboard polling."""
    with state_lock:
        return jsonify(shared_state)


@app.route("/history")
def history():
    """Return the full violation history as JSON."""
    rows = fetch_all_violations()
    data = [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "violation_type": r["violation_type"],
            "score": r["score"],
            "screenshot_path": r["screenshot_path"],
        }
        for r in rows
    ]
    return jsonify(data)


# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialise database
    init_db()

    # Start the background capture thread (daemon so it exits with the main process)
    capture_thread = threading.Thread(target=capture_and_process, daemon=True)
    capture_thread.start()

    print("\n" + "=" * 60)
    print("  AI Cheating Surveillance – Running")
    print("  Dashboard : http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
