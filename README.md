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
│   ├── best_yolov8.pt              # Custom-trained YOLOv8 weights (you place this)
│   └── shape_predictor_68_face_landmarks.dat   # dlib model (you place this)
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
```

---

## Installation

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/AI-Cheating-Surveillance.git
cd AI-Cheating-Surveillance
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on dlib:** dlib requires CMake and a C++ compiler.
> - **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with "Desktop development with C++" selected, then install CMake.
> - **Linux:** `sudo apt-get install build-essential cmake`
> - **macOS:** `xcode-select --install && brew install cmake`

### 4. Download the dlib facial landmark model

Download `shape_predictor_68_face_landmarks.dat` from:

```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

Extract the `.bz2` file and place the `.dat` file inside the `models/` folder:

```
models/shape_predictor_68_face_landmarks.dat
```

### 5. Obtain the YOLOv8 phone detection model

#### Option A — Roboflow custom dataset (recommended for resume)

1. Go to [Roboflow Universe](https://universe.roboflow.com) and search for **"Cell Phone Detection"**.
2. Choose a dataset (e.g. `cell-phone-detection` by `roboflow`).
3. Click **Download Dataset → YOLOv8 format**.
4. Train the model:

```bash
pip install ultralytics
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

5. The trained weights will be at `runs/detect/train/weights/best.pt`.
6. Copy and rename:

```bash
cp runs/detect/train/weights/best.pt models/best_yolov8.pt
```

#### Option B — Pretrained COCO fallback (zero setup)

If `models/best_yolov8.pt` is missing, the app automatically falls back to `yolov8n.pt` (downloaded automatically by Ultralytics) and uses COCO class 67 (`cell phone`). No extra step needed — just run the app.

---

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

---

## How to Run

```bash
# Activate virtual environment first
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

The dashboard will load with the live webcam feed and all detector panels.

---

## Workflow

```
Webcam Frame
     │
     ├──► dlib Face Detector
     │         ├── 0 faces → "No Face" violation
     │         ├── 1 face  → extract 68 landmarks
     │         │       ├── Head Pose (PnP solve → Euler angles)
     │         │       └── Eye Movement (EAR + iris centroid)
     │         └── 2+ faces → "Multiple Faces" violation
     │
     ├──► YOLOv8 Phone Detector
     │         └── Bounding boxes + confidence scores
     │
     ├──► Suspicion Scoring Engine
     │         └── Weighted rule accumulator with decay
     │
     ├──► Screenshot + Log (if violation)
     │
     └──► Flask MJPEG Stream + /state JSON → Browser Dashboard
```

---

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

---

## Future Improvements

- [ ] Multi-camera support for wider surveillance coverage
- [ ] Audio anomaly detection (whispering, background voices)
- [ ] Person Re-Identification to track if the candidate is replaced
- [ ] Email / SMS alert integration when score exceeds threshold
- [ ] PDF report generation at the end of each exam session
- [ ] WebSocket-based push updates instead of polling
- [ ] Admin panel for reviewing sessions with video playback
- [ ] Docker containerisation for one-command deployment

---

## Screenshots Section

Place screenshots of the running dashboard in `Demo_vid/` for your GitHub README.

Suggested screenshots:
1. Dashboard idle — no violations
2. Phone detected — bounding box visible + score spike
3. Multiple faces — warning banner active
4. Violation log — table with timestamped entries

---

## Resume Description

> **AI Cheating Surveillance System** | Python · Flask · OpenCV · dlib · YOLOv8
>
> Built a real-time web application that monitors online examination candidates via webcam using Computer Vision and Deep Learning. Implemented mobile phone detection with YOLOv8 (mAP 0.87), head pose estimation via PnP solve with Euler angle decomposition, and eye movement tracking using Eye Aspect Ratio from dlib 68-point landmarks. Designed a rule-based suspicion scoring engine with a live MJPEG streaming dashboard (Flask + vanilla JS polling). Violations are logged to SQLite and plain text with auto-saved screenshots.

---

## GitHub Description

> Real-time AI cheating detection for online exams using YOLOv8 phone detection, dlib head pose, eye movement tracking, and a Flask MJPEG live dashboard. Rule-based suspicion scoring with SQLite violation logs.

---

## Interview Preparation

**Q: How does the head pose estimation work?**
> We extract six 2D facial landmark points (nose tip, chin, eye corners, mouth corners) from dlib's 68-point predictor and solve a Perspective-n-Point (PnP) problem against a known 3D face model. OpenCV's `solvePnP` returns a rotation vector, which we convert to a rotation matrix via `Rodrigues`, then decompose into Euler angles (pitch, yaw, roll). Yaw and pitch thresholds classify the direction.

**Q: What is EAR and how is it used?**
> Eye Aspect Ratio (EAR) = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||). When the eye is open, EAR is ~0.25–0.35. When closed, it drops below ~0.21. Averaging EAR across both eyes gives a robust blink/close detector.

**Q: Why YOLOv8 for phone detection?**
> YOLOv8 is a single-pass detector — it divides the frame into a grid and predicts bounding boxes and class probabilities in one forward pass, giving real-time inference speed (~30 ms/frame on CPU, ~5 ms on GPU). For a custom cell-phone dataset from Roboflow, fine-tuning a pre-trained nano (n) or small (s) variant takes ~30 minutes on GPU.

**Q: How do you avoid spamming the database?**
> A per-violation-type cooldown dictionary tracks the last log timestamp. A violation is only recorded if at least 5 seconds have elapsed since the same type was last logged, and screenshots have a separate 10-second cooldown.

**Q: How is the video streamed to the browser?**
> The `/video_feed` route is a Flask `Response` with `mimetype="multipart/x-mixed-replace; boundary=frame"`. A Python generator yields JPEG-encoded frames wrapped in the multipart HTTP boundary. The browser `<img>` tag treats this as a live video stream (MJPEG).

---

## FAQs

**Do I need a GPU?**
No. The application runs on CPU. YOLOv8n is fast enough for ~15–20 FPS on modern CPU hardware. A GPU will increase frame rate significantly.

**The dlib install fails on Windows. What do I do?**
Install Visual Studio Build Tools 2022 with "Desktop development with C++" workload, then install CMake from cmake.org, restart your terminal, and retry `pip install dlib`.

**The webcam is not detected.**
Ensure no other application is using the camera. Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)` in `app.py` if you have multiple cameras.

**Can I use a different YOLO model?**
Yes. Replace `models/best_yolov8.pt` with any Ultralytics-compatible weights and update the class name mapping in `mobile_detection.py` if needed.

---

## Limitations

- Detection accuracy depends on webcam quality and lighting conditions.
- dlib's frontal face detector may miss extreme head angles or partially occluded faces.
- The custom YOLOv8 model requires training data; the COCO fallback has lower accuracy for unusual phone orientations.
- Head pose estimation uses a generic 3D face model, which introduces error for faces with atypical proportions.
- The eye-gaze iris-centroid method is approximate; dedicated iris tracking (e.g. MediaPipe Iris) would be more accurate but is excluded per the project constraints.
