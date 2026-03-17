# Smart AI Monitoring 👁️

A comprehensive, real-time AI computer vision dashboard that combines two powerful systems:
1. **Fire Detection Surveillance**: Uses advanced OpenCV heuristics to accurately detect incandescent flames and ignore false positive red objects.
2. **Face Recognition Attendance**: Uses advanced face encoding to automatically index people and record their presence to a CSV database.

## 🚀 Key Features
- **Dual Mode Camera**: Seamlessly switch between Fire Analysis and Face Attendance from a unified dashboard.
- **Smart Attendance Logging**: Automatically stores recognized faces, dates, and times to `database/attendance.csv`.
- **False Positive Prevention**: Strict visual criteria for fire detection eliminates errors from common red objects.
- **Auto-Screenshots**: Captures instances of highly confident fire alerts for physical review (`static/screenshots/`).

## 🛠️ Local Setup

1. **Prerequisites**:  
   * Python 3.8+
   * Windows Users: You may need **C++ Build Tools** installed via Visual Studio to compile `dlib` during the install step.
   * Add portrait photos of known people to the `known_faces/` directory.

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Server**:
   ```bash
   python app.py
   ```

4. **Access Dashboard**: Open your browser and navigate to `http://localhost:5000`

## 🌐 Live Hosting Deployment

### Render & Railway
These platforms are excellent for hosting Python web applications.
1. Upload this codebase to a GitHub Repository.
2. Connect your repo to Render/Railway as a **Web Service**.
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `gunicorn app:app --workers 1 --threads 2`
5. **Environment Variable**: Make sure you set `PORT=5000` or let the platform inject it dynamically.

*Note for Render/Railway*: Because they do not have webcams, the video feed will fail if you run the app precisely as written in a remote environment without modifying `cv2.VideoCapture(0)` to process an IP Camera stream (RTSP/HTTP). Ensure you change index 0 to your appropriate video URL in production!

### HuggingFace Spaces
1. Create a new **Docker** Space.
2. Upload these files.
3. Your `Dockerfile` should install `libgl1-mesa-glx` to satisfy OpenCV requirements:
   ```dockerfile
   FROM python:3.9
   RUN apt-get update && apt-get install -y libgl1-mesa-glx cmake
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
   ```

## 📂 Project Structure
```
smart_ai_monitoring/
├── app.py                     # Flask Server & Logic
├── fire_detection.py          # Ported OpenCV Fire Heuristic
├── face_attendance.py         # Face Recognition Module
├── templates/
│   └── index.html             # Multi-mode Dashboard UI
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── screenshots/           # Auto-saved Fire Evidence
├── database/
│   └── attendance.csv         # Generated log of seen faces
├── known_faces/               # ADD PORTRAIT IMAGES HERE
└── requirements.txt
```
