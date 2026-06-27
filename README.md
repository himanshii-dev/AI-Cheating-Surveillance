# AI Cheating Surveillance System

> Real-time AI-powered cheating detection for online examinations and interviews using a webcam, Computer Vision, and Deep Learning.

---

## Project Overview

The **AI Cheating Surveillance System** is a production-quality web application that monitors candidates during online tests or interviews. It analyses a live webcam feed and automatically flags suspicious behaviours such as looking away, using a mobile phone, multiple people in frame, or eyes being closed — all in real time.

The dashboard displays the live annotated video feed alongside detector readouts, a running suspicion score, and a timestamped violation history stored in an SQLite database.

---

## Features

| Feature | Description |
|---|---|
| **Mobile Phone Detection** | YOLOv8 detects phones with bounding boxes and confidence scores |
| **Head Pose Estimation** | Tracks gaze direction: Left, Right, Up, Down, Straight |
| **Eye Movement Detection** | Detects eye direction and closed eyes using dlib EAR |
| **Face Count Detection** | Identifies No Face / One Face / Multiple Faces |
| **Suspicion Score** | Rule-based score with real-time gauge; alert at threshold |
| **Auto Screenshots** | Saved on phone detection, no-face, multi-face, high score |
| **Violation Logging** | Timestamped entries in `logs/violations.txt` and SQLite |
| **Live Dashboard** | Dark, modern web UI with camera feed and all detector cards |
| **Violation History** | Browsable table of past violations with score badges |

---

## Folder Structure

```
AI-Cheating-Surveillance/
│
├── app.py                          # Flask app, capture loop, routes
├── requirements.txt                # Python dependencies
├── README.md
├── database.db                     # SQLite database (auto-created)
│
├── detector/
│   ├── __init__.py
│   ├── head_pose.py                # PnP-based head pose estimation
│   ├── eye_movement.py             # EAR + iris gaze detection
│   ├── mobile_detection.py         # YOLOv8 phone detection
│   ├── suspicion_score.py          # Rule-based scoring engine
│   └── database.py                 # SQLite helpers
│
├── models/
│   ├── best_yolov8.pt              # Custom-trained YOLOv8 weights
│   └── shape_predictor_68_face_landmarks.dat   # dlib model
│
├── templates/
│   └── index.html                  # Dashboard HTML
│
├── static/
│   ├── css/style.css               # Dark surveillance UI
│   ├── js/script.js                # Dashboard polling logic
│   └── screenshots/                # Auto-saved violation screenshots
│
├── logs/
│   └── violations.txt              # Plain-text violation log
│
└── Demo_vid/                       # Place demo recordings here




## Requirements

| Library | Version | Purpose |
|---|---|---|
| flask | 3.0.3 | Web framework |
| opencv-python | 4.10.0.84 | Video capture, drawing |
| dlib | 19.24.6 | Face detection, landmarks |
| ultralytics | 8.2.87 | YOLOv8 inference |
| torch | 2.4.0 | PyTorch (required by YOLO) |
| torchvision | 0.19.0 | Vision utilities |
| numpy | 1.26.4 | Numerical operations |
| Pillow | 10.4.0 | Image I/O |
| imutils | 0.5.4 | OpenCV helpers |


## Technologies Used

- **Python 3.10+** — backend language
- **Flask** — lightweight web framework + MJPEG streaming
- **OpenCV** — video capture, frame annotation, image saving
- **dlib** — frontal face detection, 68-point landmark prediction
- **YOLOv8 (Ultralytics)** — real-time phone object detection
- **PyTorch** — deep learning runtime for YOLO
- **NumPy** — matrix operations for pose estimation
- **SQLite** — embedded violation database (no server needed)
- **HTML / CSS / JavaScript** — dashboard frontend (vanilla, no framework)
*
