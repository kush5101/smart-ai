# 👁️ Smart AI Monitoring Dashboard

A multi-modal, real-time AI computer vision system designed for intelligent surveillance, safety monitoring, and automated attendance management.

---

## 🚀 Project Overview

The **Smart AI Monitoring Dashboard** is a powerful Flask-based application that integrates several state-of-the-art computer vision models into a unified, user-friendly interface. It enables real-time monitoring of multiple camera sources (local and RTSP) while simultaneously running AI analysis for security, fire safety, and operational efficiency.

---

## ✨ Core Features

### 1. 🔥 Fire Detection Surveillance
*   **Intelligent Analysis**: Re-purposed OpenCV heuristics and deep learning to detect incandescent flames.
*   **False Positive Mitigation**: Precise visual criteria to ignore standard red objects, focusing on actual thermal signatures.
*   **Automatic Snapshots**: Captures high-confidence fire alerts for audit and physical review (`static/screenshots/`).

### 2. 👤 Face Recognition & Attendance
*   **Dual Tracking**: Modern facial recognition using **YuNet** and **SFace** (ONNX-optimized) for high-speed indexing.
*   **Automated Logging**: Records check-ins and check-outs automatically to a persistent database.
*   **User Profiles**: Self-service face registration through the dashboard for non-admin users.
*   **Database Integration**: Generates daily attendance reports in `.csv` and `.xlsx` formats.

### 3. 🛡️ Security & Object Detection
*   **Weapon Identification**: Real-time detection of high-risk items (knives, firearms) using YOLO.
*   **Interaction Awareness**: Alerts if suspicious actors are near identified fire hazards or secure areas.
*   **Modular Control**: Enable or disable specific security modules via the Admin Model Control Panel.

### 4. 🏍️ Vehicle & Safety Compliance
*   **Vehicle Identification**: Detects cars, bikes, and trucks in transit.
*   **Helmet Enforcement**: Identifies riders not wearing helmets to ensure site safety compliance.
*   **Number Plate (OCR)**: Extracts license plate information (YOLOv8 + OCR) and logs it for security tracking.

---

## 🏗️ System Architecture

### Backend (Python/Flask)
*   **Asynchronous Detection Loop**: A dedicated background thread processes every active camera frame without blocking the UI.
*   **Camera Supervisor**: Dynamically manages local `cv2.CAP_DSHOW` and network `RTSP/HTTP` capture threads.
*   **MJPEG Streaming**: Specialized low-latency generators that overlay AI results directly onto the video stream at ~20 FPS.
*   **Caching Layer**: Robust memory-based state management (`debug_cache.py`) to prevent flickering detections.

### AI Model Stack
*   **YOLOv8 (`yolov8n.pt`)**: General-purpose object detection and vehicle monitoring.
*   **YuNet + SFace**: High-performance CPU-optimized facial detection and recognition.
*   **OpenCV 4.x**: Low-level image processing, fire heuristics, and visualization.

### Frontend (HTML/CSS/JS)
*   **Real-time Dashboard**: Modern, dark-mode optimized interface with dynamic UI updates.
*   **Model Control Center**: Toggle individual AI modules (Fire, Face, etc.) in real-time without restarting the server.
*   **Multi-Grid View**: Monitor all connected cameras on a single page.

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8 to 3.11 (3.12+ may have compatibility issues with older OpenCV builds).
*   **Windows**: [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) are required for `face_recognition` and `dlib` compilation.

### Step 1: Clone & Configure
```bash
git clone <repository_url>
cd smart_ai_monitoring
```

### Step 2: Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Add Known Faces
Place high-quality portrait images in the `known_faces/` directory. The filename should be the person's name (e.g., `John_Doe.jpg`).

### Step 5: Run the Application
```bash
python app.py
```
Access the dashboard at: `http://localhost:5000`

---

## 📂 Project Structure

```text
smart_ai_monitoring/
├── app.py                     # Main Flask Application & Route logic
├── detection_models/          # Core AI Processing Modules
│   ├── face_attendance.py     # Facial recognition logic
│   ├── fire_detection.py      # Fire heuristic and analysis
│   ├── object_detector.py     # Security threat identification
│   └── vehicle_detector.py    # Vehicle & License plate tracking
├── models/                    # pre-trained AI weights (.onnx, .pt)
├── camera_manager/            # Dynamic CAMERA_SOURCE thread handlers
├── database/                  # Persistent logs (attendance.csv, vehicle_logs)
├── static/                    # Dashboard assets (CSS, JS, saved screenshots)
├── templates/                 # Jinja2 HTML Dashboard templates
├── known_faces/               # Root directory for face training images
└── requirements.txt           # Python dependency list
```

---

## 🔧 Administration & Usage

1.  **Login**: Use the default administrator credentials (if configured) to access full features.
2.  **Switching Modes**: Use the sidebar to toggle between "Security View," "Attendance Summary," and "Vehicle Tracking."
3.  **Adding Cameras**: Navigate to the "Cameras" page to add local USB cameras or remote RTSP IP streams.
4.  **Model Control**: Enable/Disable specific detections (e.g., disable fire detection but keep faces active) to optimize performance.
5.  **Export Logs**: Click the "Export" buttons in the attendance/vehicle pages to download Excel/CSV reports.

---

## 🔒 Safety & Privacy
*   Face data is stored locally in the `face_images` cache and `known_faces` directory; no cloud processing is used.
*   Ensure legal compliance regarding public space recording in your jurisdiction.
