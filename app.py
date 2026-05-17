from flask import Flask, render_template, Response, jsonify, request, send_file, session, redirect, url_for, flash
import cv2
import time
import os
import io
import threading
import pandas as pd
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import json
import logging
from datetime import datetime
import traceback

# ── Detection module imports ───────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'detection_models'))

from fire_detection import FireDetector
from face_attendance import FaceAttendanceSystem
from object_detector import ObjectDetector
from vehicle_detector import VehicleDetector

app = Flask(__name__)
app.secret_key = 'super-secret-smart-ai-key-change-in-production'

def normalize_name(name):
    if not name: return ""
    return "".join(name.split()).lower()

# ── Authentication Decorators ───────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('user') != 'admin':
            flash('Access denied. Administrator only.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# ── Detectors (Lazy Loading) ──────────────────────────────────────────────────
import gc
fire_sys   = None
fire_detectors = {} # Camera-specific FireDetectors to prevent cross-camera history/size mismatch
face_sys   = None
obj_sys    = None
veh_sys    = None
is_initializing = False

# ── Model Toggles (enable/disable individual AI modules) ──
model_toggles = {"fire": True, "face": True, "object": True, "vehicle": True,
    "helmet": True,
    "plate": True}

def initialize_detectors():
    global fire_sys, face_sys, obj_sys, veh_sys, is_initializing
    if is_initializing: return
    is_initializing = True
    try:
        print("[INFO] Initializing detectors sequentially...")
        
        if fire_sys is None:
            print("[INFO] Loading Fire Detector...")
            fire_sys = FireDetector(confidence_threshold=0.6)
            gc.collect()
            
        if face_sys is None:
            print("[INFO] Loading Face Attendance System...")
            face_sys = FaceAttendanceSystem()
            gc.collect()
            
        if obj_sys is None:
            print("[INFO] Loading Object Detector...")
            obj_sys = ObjectDetector()
            gc.collect()
            
        if veh_sys is None:
            print("[INFO] Loading Vehicle Detector...")
            # Use local import and explicit class reference to bypass any namespace ghosting
            from vehicle_detector import VehicleDetector as VDet
            veh_sys = VDet()
            gc.collect()
            
        print("[INFO] All detectors loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load detectors: {e}")
        traceback.print_exc()
    finally:
        is_initializing = False

# Per-camera state
camera_frames   = {} # { cam_id: { "raw": frame, "processed": frame } }
camera_caps     = {} # { cam_id: cv2.VideoCapture }
camera_statuses = {} # { cam_id: { "fire": ..., "attendance": ..., "object": ... } }

def _init_cam_status(cid):
    if cid not in camera_statuses:
        camera_statuses[cid] = {
            "fire":       {"detected": False, "confidence": 0.0, "timestamp": None},
            "attendance": {"recent_faces": []},
            "object":     {"weapon": False, "weapon_labels": [], "nearby": [], "timestamp": None},
            "vehicle":    {"detected": []}
        }

_frame_lock = threading.Lock()
last_screenshot_time = 0

# ── Multi-Camera State ────────────────────────────────────────────────────────
default_source = os.environ.get("CAMERA_SOURCE", 0)
try:
    if str(default_source).isdigit():
        default_source = int(default_source)
except:
    pass

# ── RTSP Relay: auto-swap RTSP → public MJPEG when relay tunnel is active ──
def _load_relay_overrides():
    """
    Reads rtsp_relay/tunnel_url.txt written by relay_server.py --ngrok.
    Returns a dict: { cam_id_int: "https://<ngrok>/stream/<id>" }
    Example file content:
        https://abc123.ngrok.io
        cam1=https://abc123.ngrok.io/stream/1
        cam2=https://abc123.ngrok.io/stream/2
    """
    overrides = {}
    relay_file = os.path.join(os.path.dirname(__file__), "rtsp_relay", "tunnel_url.txt")
    if not os.path.exists(relay_file):
        return overrides
    try:
        with open(relay_file, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, url = line.split("=", 1)
                    cam_id = int(key.replace("cam", "").strip())
                    overrides[cam_id] = url.strip()
        if overrides:
            print(f"[RelayLoader] Loaded {len(overrides)} relay overrides from tunnel_url.txt")
    except Exception as e:
        print(f"[RelayLoader] Could not parse tunnel_url.txt: {e}")
    return overrides

_relay_overrides = _load_relay_overrides()

camera_sources = [
    {"id": 0, "name": "Main Entry - Cam 01", "source": default_source},
    {
        "id": 1,
        "name": "Testing RTSP",
        # Use relay MJPEG URL if available (production), else fall back to direct RTSP (local)
        "source": _relay_overrides.get(1, "rtsp://admin:admin@192.168.1.20:1935")
    }
]
active_camera_id = 0

if _relay_overrides:
    print(f"[RelayLoader] Camera 1 source → {camera_sources[1]['source']}")

# ── Camera Thread ─────────────────────────────────────────────────────────────
_cap = None
# Thread management
camera_threads = {} # { cam_id: Thread }
stop_signals   = {} # { cam_id: Event }

def webcam_capture_thread(cam_id, source, stop_event):
    """Dedicated thread purely for local webcams using DSHOW."""
    global camera_frames
    
    if os.name == 'nt':
        import ctypes
        ctypes.windll.ole32.CoInitialize(0)
        
    print(f"[Cam {cam_id} - WEBCAM] Starting capture thread for source: {source}")

    def _open_webcam():
        if os.name == 'nt':
            c = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            if not c.isOpened():
                c = cv2.VideoCapture(int(source))
        else:
            c = cv2.VideoCapture(int(source))
            
        if c.isOpened():
            # Standard resolution request, avoid forcing FOURCC which breaks some webcams
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return c

    cap = _open_webcam()
    if cap.isOpened():
        print(f"[INFO] Cam {cam_id} (Webcam) connected successfully.")
    else:
        print(f"[ERROR] Cam {cam_id} failed to open webcam: {source}")

    warmup_remaining = 20
    frames_captured = 0

    while not stop_event.is_set():
        if not cap.isOpened():
            print(f"[WARN] Cam {cam_id} (Webcam) not opened. Retrying in 2s...")
            time.sleep(2)
            cap = _open_webcam()
            warmup_remaining = 20
            continue

        success, frame = cap.read()
        if success and frame is not None and frame.size > 0:
            if warmup_remaining > 0:
                warmup_remaining -= 1
                time.sleep(0.01)
                continue
            
            # Debug: log first few frames
            if frames_captured < 5:
                print(f"[Cam {cam_id}] Frame captured: shape={frame.shape}, dtype={frame.dtype}")
            
            # Validate frame is proper BGR (3-channel, not corrupt)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_clone = frame.copy()
                with _frame_lock:
                    if cam_id not in camera_frames: camera_frames[cam_id] = {}
                    camera_frames[cam_id]["raw"] = frame_clone
                    if frames_captured < 5:
                        print(f"[Cam {cam_id}] Frame stored in camera_frames[{cam_id}]")
                
                frames_captured += 1
                if frames_captured % 100 == 0:
                    print(f"[HEARTBEAT] Cam {cam_id} captured {frames_captured} frames.", flush=True)
            else:
                print(f"[WARN] Cam {cam_id} unexpected frame shape {frame.shape}. Skipping.", flush=True)
        else:
            print(f"[WARN] Capture failed or empty frame for Cam {cam_id} (Webcam). Reconnecting...")
            cap.release()
            time.sleep(2)
            cap = _open_webcam()
            warmup_remaining = 20
        
        # Micro-sleep to prevent 100% CPU
        time.sleep(0.01)
    
    if cap:
        cap.release()
    print(f"[INFO] Capture thread for Cam {cam_id} (Webcam) stopped.")


def rtsp_capture_thread(cam_id, source, stop_event):
    """Dedicated thread purely for RTSP/Network streams."""
    global camera_frames
    
    print(f"[Cam {cam_id} - NETWORK] Starting capture thread for source: {source}")
    is_rtsp = isinstance(source, str) and source.startswith('rtsp')
    
    def _open_rtsp():
        # Use CAP_PROP_OPEN_TIMEOUT_MSEC instead of global env var
        # to avoid contaminating other VideoCapture instances in the process
        if is_rtsp:
            c = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if c.isOpened():
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                c.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                c.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            return c
        return cv2.VideoCapture(source)

    cap = _open_rtsp()
    if cap.isOpened():
        print(f"[INFO] Cam {cam_id} (Network) connected successfully.")
    else:
        print(f"[ERROR] Cam {cam_id} failed to open Network source: {source}")

    consecutive_failures = 0
    while not stop_event.is_set():
        if not cap.isOpened():
            print(f"[WARN] Cam {cam_id} (Network) not opened. Retrying in 2s...")
            time.sleep(2)
            cap = _open_rtsp()
            consecutive_failures = 0
            continue

        success, frame = cap.read()
        if success and frame is not None:
            consecutive_failures = 0
            frame_clone = frame.copy()
            with _frame_lock:
                if cam_id not in camera_frames: camera_frames[cam_id] = {}
                camera_frames[cam_id]["raw"] = frame_clone
        else:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                print(f"[WARN] Capture failed 10 consecutive times for Cam {cam_id} (Network). Reconnecting...")
                cap.release()
                time.sleep(2)
                consecutive_failures = 0
            else:
                time.sleep(0.05)
        
        # Micro-sleep to prevent 100% CPU
        time.sleep(0.01)
    
    if cap:
        cap.release()
    print(f"[INFO] Capture thread for Cam {cam_id} (Network) stopped.")


def camera_supervisor_loop():
    """Manager thread that ensures a capture thread exists for every source."""
    global camera_threads, stop_signals
    while True:
        current_ids = [c['id'] for c in camera_sources]
        
        # 1. Stop threads for removed cameras
        for cid in list(camera_threads.keys()):
            if cid not in current_ids:
                stop_signals[cid].set()
                camera_threads[cid].join(timeout=1)
                del camera_threads[cid]
                del stop_signals[cid]

        # 2. Start threads for new cameras
        for cam in camera_sources:
            cid = cam['id']
            source = cam['source']
            if cid not in camera_threads:
                stop_event = threading.Event()
                is_local = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
                
                if is_local:
                    t = threading.Thread(target=webcam_capture_thread, args=(cid, source, stop_event), daemon=True)
                else:
                    t = threading.Thread(target=rtsp_capture_thread, args=(cid, source, stop_event), daemon=True)
                
                t.start()
                camera_threads[cid] = t
                stop_signals[cid] = stop_event
        
        time.sleep(1)

# supervisor thread started in main

# Detection persistence to prevent flickering
latest_detections = {} # { cam_id: { "face": {"dets": [], "seen": []}, "fire": [], "obj": [] } }

def _init_detections(cam_id):
    if cam_id not in latest_detections:
        latest_detections[cam_id] = {
            "face": {"dets": [], "seen": []},
            "fire": [],
            "obj": [],
            "vehicle": []
        }
    
def detection_loop():
    global camera_frames, camera_statuses, last_screenshot_time, latest_detections
    print("[DEBUG DetectionLoop] AI thread starting (Non-Flicker Mode)...")
    frame_count = 0
    
    while True:
        try:
            # 0. Ensure all state for every camera is initialized once to prevent KeyErrors
            for c_src in camera_sources:
                cid = c_src['id']
                _init_cam_status(cid)
                _init_detections(cid)
                if cid not in camera_frames:
                    with _frame_lock:
                        camera_frames[cid] = {"raw": None}
            
            if fire_sys is None or face_sys is None or obj_sys is None or veh_sys is None:
                initialize_detectors()
                # If still failing, wait longer before retrying to save CPU
                if fire_sys is None or face_sys is None or obj_sys is None or veh_sys is None:
                    time.sleep(10)
                    continue

            # 1. Background Fire Monitoring (Low Frequency)
            for cam in camera_sources:
                cid = cam['id']
                # Background: Check fire only once every 60 frames (approx 3s)
                # Active: Check fire every 10 frames (approx 0.5s)
                skip_val = 10 if cid == active_camera_id else 60
                if frame_count % skip_val == 0 and model_toggles.get("fire", True):
                    f_frame = None
                    with _frame_lock:
                        data = camera_frames.get(cid)
                        if data and data.get("raw") is not None:
                            f_frame = data["raw"].copy() # Must copy to release lock immediately
                            if f_frame.shape[1] != 640:
                                f_frame = cv2.resize(f_frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                            
                    if f_frame is not None and fire_sys:
                        if cid not in fire_detectors:
                            fire_detectors[cid] = FireDetector(confidence_threshold=0.6)
                        fire_detected, detections = fire_detectors[cid].detect(f_frame)
                        _init_cam_status(cid)
                        _init_detections(cid)
                        
                        camera_statuses[cid]["fire"]["detected"] = fire_detected
                        latest_detections[cid]["fire"] = detections if fire_detected else []
                        if fire_detected:
                            max_conf = max([d["confidence"] for d in detections])
                            camera_statuses[cid]["fire"]["confidence"] = round(max_conf * 100, 2)
                        else:
                            camera_statuses[cid]["fire"]["confidence"] = 0.0

            # 2. Focused Face/Object Detection (ACTIVE CAMERA ONLY)
            # Relaxed frequency: Face (20 frames), Object/Vehicle (15 frames)
            process_now = (frame_count % 20 == 0 or frame_count % 15 == 0)
            if process_now:
                frame = None
                with _frame_lock:
                    data = camera_frames.get(active_camera_id)
                    if data and data.get("raw") is not None:
                        frame = data["raw"].copy()
                        if frame.shape[1] != 640:
                            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                
                if frame is not None:
                    # Face (Every 20 frames - 1 FPS approx)
                    if frame_count % 20 == 0 and model_toggles.get("face", True) and face_sys:
                        dets, seen = face_sys.detect_and_recognize(frame)
                        latest_detections[active_camera_id]["face"]["dets"] = dets
                        latest_detections[active_camera_id]["face"]["seen"] = seen
                        camera_statuses[active_camera_id]["attendance"]["recent_faces"] = seen
                    
                    # Object (Every 15 frames - 1.5 FPS approx)
                    if frame_count % 15 == 0 and model_toggles.get("object", True) and obj_sys:
                        fire_bboxes = latest_detections.get(active_camera_id, {}).get("fire", [])
                        weapon_det, weapon_labels, nearby_objs, obj_dets = obj_sys.detect(
                            frame, 
                            fire_bboxes=fire_bboxes
                        )
                        camera_statuses[active_camera_id]["object"]["weapon"]        = weapon_det
                        camera_statuses[active_camera_id]["object"]["weapon_labels"] = weapon_labels
                        camera_statuses[active_camera_id]["object"]["nearby"]        = nearby_objs
                        latest_detections[active_camera_id]["obj"] = obj_dets

                    # Vehicle & Helmet (Every 15 frames - 1.5 FPS approx)
                    if frame_count % 15 == 0 and (model_toggles.get("vehicle", True) or model_toggles.get("helmet", True) or model_toggles.get("plate", True)) and veh_sys:
                        f_dets = latest_detections.get(active_camera_id, {}).get("face", {}).get("dets", [])
                        v_dets = veh_sys.detect(
                            frame, 
                            detect_vehicle=model_toggles.get("vehicle", True),
                            detect_helmet=model_toggles.get("helmet", True),
                            face_bboxes=f_dets,
                            detect_plate=model_toggles.get("plate", True)
                        )
                        latest_detections[active_camera_id]["vehicle"] = v_dets
                        camera_statuses[active_camera_id]["vehicle"]["detected"] = v_dets
                        
                        # Persist vehicle detection if not recently recorded
                        if v_dets:
                            for v in v_dets:
                                # Simple throttling: only record if v_label not recorded in last 5s
                                # (Normally would be better tracking, but this fulfills "records")
                                _record_vehicle(v)

            frame_count = (frame_count + 1) % 1200
            time.sleep(0.01)

        except Exception as e:
            print(f"[ERROR DetectionLoop] {e}", flush=True)
            traceback.print_exc()
            time.sleep(1)

# detection thread started in main

# ── MJPEG stream ──────────────────────────────────────────────────────────────
def _stream_generator(cam_id):
    # Balanced quality (60) for speed and clarity
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
    frame_count = 0
    print(f"[StreamGen] Started for cam_id: {cam_id}")
    while True:
        try:
            raw = None
            with _frame_lock:
                data = camera_frames.get(cam_id)
                if data is not None and data.get("raw") is not None:
                    raw = data["raw"].copy()
                    
            if raw is None:
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"[StreamGen] No frame available for cam {cam_id}, camera_frames keys: {list(camera_frames.keys())}")
                    # Debug: check if camera_frames has data
                    if cam_id in camera_frames:
                        print(f"[StreamGen] cam {cam_id} data: {camera_frames[cam_id]}")
                time.sleep(0.05)
                continue
            
            frame_count = 0  # Reset when we get a frame
            print(f"[StreamGen] Yielding frame for cam {cam_id}, shape: {raw.shape}")
                
            # Fast downsample if needed, use INTER_NEAREST (fastest)
            if raw.shape[1] > 640:
                frame = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_LINEAR)
            else:
                frame = raw
            
            # Apply persistent detections outside the main lock (only if model is enabled)
            if cam_id in latest_detections:
                dets = latest_detections[cam_id]
                try:
                    is_admin = session.get('user') == 'admin'
                except Exception as e:
                    # In some contexts session might be unavailable
                    is_admin = False
                
                if model_toggles.get("fire", True) and dets.get("fire") and fire_sys: 
                    frame = fire_sys.draw_detections(frame, dets["fire"])
                if model_toggles.get("face", True) and dets.get("face") and dets["face"].get("dets") and face_sys: 
                    frame = face_sys.draw_faces(frame, dets["face"]["dets"])
                if model_toggles.get("object", True) and dets.get("obj") and obj_sys: 
                    frame = obj_sys.draw_detections(frame, dets["obj"])
                    
                # Restrict Vehicle and Helmet overlays to Admin only
                if is_admin:
                    if (model_toggles.get("vehicle", True) or model_toggles.get("helmet", True)) and dets.get("vehicle") and veh_sys: 
                        frame = veh_sys.draw_detections(frame, dets["vehicle"])
            
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                print(f"[StreamGen] Failed to encode frame for cam {cam_id}")
                time.sleep(0.01)
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[ERROR StreamGen] Cam {cam_id}: {e}")
            # Optionally yield a placeholder or just wait
        
        # Target 20 FPS comfortably to save CPU
        time.sleep(0.05)

def generate_frames():
    """Stream from the currently active camera source."""
    global active_camera_id
    yield from _stream_generator(active_camera_id)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/monitor')
@login_required
def index():
    return render_template('index.html')

@app.route('/cameras')
@login_required
@admin_required
def cameras_page():
    return render_template('cameras.html')

@app.route('/vehicles')
@login_required
@admin_required
def vehicles_page():
    return render_template('vehicles.html')

@app.route('/models')
@login_required
@admin_required
def models_page():
    return render_template('models.html')

@app.route('/api/models', methods=['GET'])
@login_required
@admin_required
def get_models():
    return jsonify(model_toggles)

@app.route('/api/models/toggle', methods=['POST'])
@login_required
@admin_required
def toggle_model():
    data = request.get_json()
    model = data.get('model')
    enabled = data.get('enabled', True)
    if model in model_toggles:
        model_toggles[model] = bool(enabled)
        # When disabling, clear cached results so overlays disappear immediately
        if not enabled:
            for cid in list(latest_detections.keys()):
                if model == "fire":
                    latest_detections[cid]["fire"] = []
                    camera_statuses.get(cid, {}).get("fire", {})["detected"] = False
                    camera_statuses.get(cid, {}).get("fire", {})["confidence"] = 0.0
                elif model == "face":
                    latest_detections[cid]["face"] = {"dets": [], "seen": []}
                    camera_statuses.get(cid, {}).get("attendance", {})["recent_faces"] = []
                elif model == "object":
                    latest_detections[cid]["obj"] = []
                    obj_status = camera_statuses.get(cid, {}).get("object", {})
                    if obj_status:
                        obj_status["weapon"] = False
                        obj_status["weapon_labels"] = []
                        obj_status["nearby"] = []
                elif model in ["vehicle", "helmet"]:
                    latest_detections[cid]["vehicle"] = []
                    if "vehicle" in camera_statuses.get(cid, {}):
                        camera_statuses[cid]["vehicle"]["detected"] = []
        return jsonify({"status": "ok", "model": model, "enabled": model_toggles[model]})
    return jsonify({"status": "error", "message": "Unknown model"}), 400

@app.route('/test_camera')
def test_camera():
    return render_template('test_camera.html')

@app.route('/video_feed')
def video_feed():
    print(f"[VideoFeed] Route called, active_camera_id: {active_camera_id}", flush=True)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<int:cam_id>')
def video_feed_cam(cam_id):
    """Per-camera MJPEG stream for the multi-camera grid."""
    return Response(_stream_generator(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
@login_required
def get_status():
    current_user = session.get('user')
    _init_cam_status(active_camera_id)
    active_status = camera_statuses[active_camera_id]
    
    if face_sys is None:
        return jsonify({
            "fire": active_status["fire"],
            "attendance": {"recent_faces": [], "table": []},
            "object": active_status["object"],
            "vehicle": active_status["vehicle"],
            "status": "initializing"
        })
    
    table = face_sys.get_today_table()
    recent = list(active_status["attendance"]["recent_faces"])

    # Filter for user
    if current_user and current_user.lower() != 'admin':
        table  = [row for row in table  if normalize_name(row['Name']) == normalize_name(current_user)]
        recent = [f   for f   in recent if normalize_name(f)           == normalize_name(current_user)]

    active_cam = next((c for c in camera_sources if c['id'] == active_camera_id), camera_sources[0])

    is_admin = current_user == 'admin'
    
    status_resp = {
        "fire": active_status["fire"],
        "attendance": {"recent_faces": recent, "table": table},
        "object": active_status["object"],
        "active_camera_id": active_camera_id,
        "active_camera_name": active_cam['name'],
        "cameras": camera_sources
    }

    if is_admin:
        status_resp["vehicle"] = active_status["vehicle"]
        status_resp["all_statuses"] = camera_statuses
    else:
        status_resp["vehicle"] = {"detected": [], "alert": False}
        # Filter all_statuses to only include non-restricted info
        filtered_all = {}
        for cid, stat in camera_statuses.items():
            filtered_all[cid] = {
                "fire": stat.get("fire"),
                "object": stat.get("object"),
                "attendance": stat.get("attendance")
            }
        status_resp["all_statuses"] = filtered_all

    return jsonify(status_resp)


# ── Camera Management API ──────────────────────────────────────────────────────
@app.route('/api/cameras', methods=['GET', 'POST'])
@login_required
@admin_required
def manage_cameras():
    global active_camera_id, camera_sources
    if request.method == 'POST':
        data = request.get_json()
        new_cam = {
            "id": int(time.time()),
            "name": data.get('name', 'New Camera'),
            "source": data.get('source', 0)
        }
        try:
            if str(new_cam['source']).isdigit():
                new_cam['source'] = int(new_cam['source'])
        except: pass
        camera_sources.append(new_cam)
        return jsonify({"status": "success", "camera": new_cam})
    return jsonify({"cameras": camera_sources, "active_id": active_camera_id})

@app.route('/api/cameras/switch', methods=['POST'])
@login_required
@admin_required
def switch_camera():
    global active_camera_id, _processed_frame
    data = request.get_json()
    cam_id = int(data.get('id', 0))
    if any(c['id'] == cam_id for c in camera_sources):
        active_camera_id = cam_id
        return jsonify({"status": "success", "active_id": cam_id})
    return jsonify({"status": "error", "message": "Camera not found"}), 404

@app.route('/api/cameras/delete', methods=['POST'])
@login_required
@admin_required
def delete_camera():
    global camera_sources, active_camera_id
    data = request.get_json()
    cam_id = int(data.get('id', -1))
    if cam_id == 0:
        return jsonify({"status": "error", "message": "Cannot delete default camera"}), 400
    camera_sources = [c for c in camera_sources if c['id'] != cam_id]
    if active_camera_id == cam_id:
        active_camera_id = 0
    return jsonify({"status": "success"})

@app.route('/export_attendance')
def export_attendance():
    import pandas as pd
    db_path = os.path.join('database', 'attendance.csv')
    if not os.path.exists(db_path):
        return jsonify({"status": "error", "message": "No attendance data yet."}), 404
    try:
        df = pd.read_csv(db_path)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
            ws = writer.sheets['Attendance']
            for col in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = max_len + 4
        output.seek(0)
        date_str = __import__('datetime').datetime.now().strftime('%Y-%m-%d')
        return send_file(output,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True, download_name=f'attendance_{date_str}.xlsx')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/register_face', methods=['POST'])
@login_required
def register_face():
    data = request.get_json() or {}
    current_user = session.get('user')
    name = data.get('name', '').strip()
    photo_num = data.get('photo_num', 1)
    if current_user != 'admin':
        name = current_user
    else:
        name = name if name else 'Admin'
    if not name:
        return jsonify({"status": "error", "message": "Name is required."}), 400
    if face_sys is None:
        return jsonify({"status": "error", "message": "Detectors are still initializing. Please wait 10 seconds."}), 503
    with _frame_lock:
        data = camera_frames.get(active_camera_id)
        if data is None or data.get("raw") is None:
            return jsonify({"status": "error", "message": "No camera feed available."}), 500
        frame = data["raw"].copy()
    success, msg = face_sys._register_new_face(frame, name, photo_num=photo_num)
    if not success:
        return jsonify({"status": "error", "message": msg}), 400
    return jsonify({"status": "success", "message": msg})

@app.route('/faces')
@login_required
def faces_page():
    is_admin = session.get('user') == 'admin'
    return render_template('faces.html', is_admin=is_admin)

@app.route('/known_faces_list')
@login_required
def known_faces_list():
    current_user = session.get('user')
    faces = []
    folder = 'known_faces'
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                name = face_sys._clean_name(f)
                if current_user == 'admin' or normalize_name(name) == normalize_name(current_user):
                    faces.append({"filename": f, "name": name})
    return jsonify({"faces": faces})

@app.route('/face_image/<filename>')
def face_image(filename):
    fname_lower = filename.lower()
    if fname_lower not in face_sys.face_images:
        return '', 404
    return send_file(io.BytesIO(face_sys.face_images[fname_lower]), mimetype='image/jpeg')

@app.route('/delete_face', methods=['POST'])
@login_required
def delete_face():
    try:
        data = request.get_json()
        filename = (data or {}).get('filename', '')
        basename = os.path.basename(filename)
        if not basename or basename != filename:
            return jsonify({'status': 'error', 'message': 'Invalid filename.'}), 400
        current_user = session.get('user')
        if face_sys is None:
             return jsonify({'status': 'error', 'message': 'System initializing.'}), 503
        face_name = face_sys._clean_name(basename)
        if current_user != 'admin' and normalize_name(face_name) != normalize_name(current_user):
            return jsonify({'status': 'error', 'message': f'Unauthorized.'}), 403
        folder = os.path.abspath('known_faces')
        filepath = os.path.join(folder, basename)
        if not os.path.exists(filepath):
            return jsonify({'status': 'error', 'message': f'File not found.'}), 404
        if basename.lower() in face_sys.face_images:
            del face_sys.face_images[basename.lower()]
        import gc; gc.collect()
        for i in range(5):
            try:
                os.remove(filepath); break
            except PermissionError:
                time.sleep(0.3 * (i + 1))
        face_sys.reload_faces()
        return jsonify({'status': 'success', 'message': f'Removed {basename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/attendance')
@login_required
def attendance_page():
    return render_template('attendance.html')

@app.route('/all_attendance')
@login_required
def all_attendance():
    import pandas as pd, datetime as dt
    db_path = os.path.join('database', 'attendance.csv')
    if not os.path.exists(db_path):
        return jsonify({"records": []})
    try:
        current_user = session.get('user')
        today = dt.datetime.now().strftime('%Y-%m-%d')
        df = pd.read_csv(db_path).fillna('')
        if current_user != 'admin':
            df = df[df['Name'].apply(lambda x: "".join(str(x).split()).lower() == "".join(current_user.split()).lower())]
        records = df.to_dict(orient='records')
        live_table = {row['Name'].strip(): row for row in face_sys.get_today_table()}
        if current_user != 'admin':
            live_table = {k: v for k, v in live_table.items() if "".join(k.split()).lower() == "".join(current_user.split()).lower()}
        for r in records:
            if r.get('Date') == today and r['Name'] in live_table:
                live = live_table.pop(r['Name'])
                r['Check-In']  = live['Check-In']  or r.get('Check-In', '')
                r['Check-Out'] = live['Check-Out'] if live['Check-Out'] != '—' else (r.get('Check-Out', '') or '—')
        for name, row in live_table.items():
            records.insert(0, {'Name': name, 'Date': today, 'Check-In': row['Check-In'], 'Check-Out': row['Check-Out']})
        for r in records:
            for k in ('Check-In', 'Check-Out'):
                if not r.get(k): r[k] = '—'
        return jsonify({"records": records})
    except Exception as e:
        return jsonify({"error": str(e), "records": []})

@app.route('/update_record', methods=['POST'])
def update_record():
    try:
        data = request.get_json()
        import pandas as pd, datetime as dt
        db_path = os.path.join('database', 'attendance.csv')
        df = pd.read_csv(db_path) if os.path.exists(db_path) else pd.DataFrame(columns=['Name','Date','Check-In','Check-Out'])
        mask = (df['Name'] == data['old_name']) & (df['Date'] == data['old_date']) & (df['Check-In'] == data['old_in'])
        if mask.any():
            idx = mask.idxmax()
            df.loc[idx, 'Name'] = data['new_name']; df.loc[idx, 'Date'] = data['new_date']
            df.loc[idx, 'Check-In'] = data['new_in']; df.loc[idx, 'Check-Out'] = data['new_out']
            df.to_csv(db_path, index=False)
        else:
            new_row = pd.DataFrame([{'Name': data['new_name'], 'Date': data['new_date'], 'Check-In': data['new_in'], 'Check-Out': data['new_out']}])
            pd.concat([df, new_row], ignore_index=True).to_csv(db_path, index=False)
        today = dt.datetime.now().strftime('%Y-%m-%d')
        if data['old_date'] == today and data['old_name'] in face_sys.active_sessions:
            sess = face_sys.active_sessions.pop(data['old_name'])
            sess.update({'date': data['new_date'], 'check_in': data['new_in'], 'check_out': data['new_out'] or None})
            if data['new_date'] == today:
                face_sys.active_sessions[data['new_name']] = sess
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/delete_record', methods=['POST'])
@login_required
def delete_record():
    try:
        data = request.get_json()
        current_user = session.get('user')
        if current_user != 'admin' and normalize_name(data['name']) != normalize_name(current_user):
            return jsonify({'status': 'error', 'message': 'Unauthorized.'}), 403
        import pandas as pd, datetime as dt
        db_path = os.path.join('database', 'attendance.csv')
        if os.path.exists(db_path):
            df = pd.read_csv(db_path)
            mask = (df['Name'] == data['name']) & (df['Date'] == data['date']) & (df['Check-In'] == data['check_in'])
            df[~mask].to_csv(db_path, index=False)
        today = dt.datetime.now().strftime('%Y-%m-%d')
        if data['date'] == today and data['name'] in face_sys.active_sessions:
            if face_sys.active_sessions[data['name']]['check_in'] == data['check_in']:
                face_sys.active_sessions.pop(data['name'])
                face_sys._currently_visible.discard(data['name'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ── Auth Routes ────────────────────────────────────────────────────────────────
def get_users_db_path():
    os.makedirs('database', exist_ok=True)
    return os.path.join('database', 'users.csv')

def init_users_db():
    db_path = get_users_db_path()
    if not os.path.exists(db_path):
        pd.DataFrame(columns=['Username', 'PasswordHash']).to_csv(db_path, index=False)
    df = pd.read_csv(db_path)
    if 'admin' not in df['Username'].values:
        hashed = generate_password_hash('admin')
        pd.concat([df, pd.DataFrame([{'Username': 'admin', 'PasswordHash': hashed}])], ignore_index=True).to_csv(db_path, index=False)

init_users_db()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'
        try:
            df = pd.read_csv(get_users_db_path())
            user_row = df[df['Username'] == username]
            
            if df.empty and username == 'admin' and password == 'admin':
                session.permanent = remember
                session['user'] = username
                return redirect(url_for('index'))
                
            if not user_row.empty and check_password_hash(user_row.iloc[0]['PasswordHash'], password):
                session.permanent = remember
                session['user'] = username
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
            flash('Invalid username or password', 'error')
        except Exception as e:
            flash(f'Error accessing database: {e}', 'error')
    return render_template('login.html')

@app.route('/register_user', methods=['GET', 'POST'])
def register_user():
    df = pd.read_csv(get_users_db_path())
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')
        if not username or not password:
            flash('All fields are required.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        elif username.lower() in df['Username'].str.lower().values:
            flash('Username already exists.', 'error')
        else:
            hashed = generate_password_hash(password)
            pd.concat([df, pd.DataFrame([{'Username': username, 'PasswordHash': hashed}])], ignore_index=True).to_csv(get_users_db_path(), index=False)
            session['user'] = username
            return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/manage_users')
@login_required
def manage_users():
    if session.get('user') != 'admin':
        flash('Access denied. Admin only.', 'error')
        return redirect(url_for('index'))
    df = pd.read_csv(get_users_db_path())
    users = df[df['Username'] != 'admin'].to_dict(orient='records')
    return render_template('users.html', users=users)

@app.route('/update_user_password', methods=['POST'])
@login_required
def update_user_password():
    if session.get('user') != 'admin':
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    data = request.get_json()
    username, new_password = data.get('username'), data.get('password')
    if not username or not new_password:
        return jsonify({'status': 'error', 'message': 'Username and password required'}), 400
    df = pd.read_csv(get_users_db_path())
    if username in df['Username'].values:
        df.loc[df['Username'] == username, 'PasswordHash'] = generate_password_hash(new_password)
        df.to_csv(get_users_db_path(), index=False)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'User not found'}), 404

@app.route('/delete_user', methods=['POST'])
@login_required
def delete_user():
    if session.get('user') != 'admin':
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    data = request.get_json()
    username = data.get('username')
    if not username or username == 'admin':
        return jsonify({'status': 'error', 'message': 'Invalid username'}), 400
    df = pd.read_csv(get_users_db_path())
    if username in df['Username'].values:
        df[df['Username'] != username].to_csv(get_users_db_path(), index=False)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'User not found'}), 404

# ── Vehicle Records Logic ───────────────────────────────────────────────────
VEHICLE_LOG = "database/vehicle_log.csv"
last_recorded_vehicles = {} # { "label": timestamp }

def _record_vehicle(v):
    global last_recorded_vehicles
    label = v['label']
    now = time.time()
    if now - last_recorded_vehicles.get(label, 0) > 10: # 10s throttle
        last_recorded_vehicles[label] = now
        try:
            os.makedirs("database", exist_ok=True)
            df_new = pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Vehicle": label,
                "Type": v['type'],
                "Helmet": v['helmet'],
                "Confidence": f"{v['confidence']}%"
            }])
            df_new.to_csv(VEHICLE_LOG, mode='a', header=not os.path.exists(VEHICLE_LOG), index=False)
        except Exception as e:
            print(f"[RECORDS] Vehicle log error: {e}")

@app.route('/api/vehicle_history')
@login_required
@admin_required
def vehicle_history():
    if not os.path.exists(VEHICLE_LOG):
        return jsonify([])
    try:
        df_v = pd.read_csv(VEHICLE_LOG).tail(50)
        records = df_v.to_dict(orient='records')
        records.reverse()
        return jsonify(records)
    except:
        return jsonify([])

@app.route('/vehicle_records')
@login_required
@admin_required
def vehicle_records_page():
    return render_template('vehicle_records.html')

if __name__ == "__main__":
    # In debug mode, Flask runs twice. Only start threads in the actual app process (not the reloader).
    if not os.environ.get("VERCEL") and (os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug):
        print(f"[INFO] Starting background threads (PID: {os.getpid()})...")
        _supervisor_thread = threading.Thread(target=camera_supervisor_loop, daemon=True)
        _supervisor_thread.start()
        _det_thread = threading.Thread(target=detection_loop, daemon=True)
        _det_thread.start()
    
    app.run(debug=True, threaded=True, port=5000)
