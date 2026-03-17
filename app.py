from flask import Flask, render_template, Response, jsonify, request, send_file, session, redirect, url_for, flash
import cv2
import time
import os
import io
import threading
import pandas as pd
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

# ── Detection module imports ───────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'detection_models'))

from fire_detection import FireDetector
from face_attendance import FaceAttendanceSystem
from object_detector import ObjectDetector

app = Flask(__name__)
app.secret_key = 'super-secret-smart-ai-key-change-in-production'

def normalize_name(name):
    if not name: return ""
    return "".join(name.split()).lower()

# ── Authentication Decorator ───────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# ── Detectors ─────────────────────────────────────────────────────────────────
fire_detector   = FireDetector(confidence_threshold=0.6)
face_detector   = FaceAttendanceSystem()
object_detector = ObjectDetector()

# ── Shared state ──────────────────────────────────────────────────────────────
detection_status = {
    "fire":       {"detected": False, "confidence": 0.0, "timestamp": None},
    "attendance": {"recent_faces": []},
    "object":     {"weapon": False, "weapon_labels": [], "nearby": [], "timestamp": None}
}

_raw_frame = None
_processed_frame = None
_current_frame_cache = None
_frame_lock = threading.Lock()
last_screenshot_time = 0

# ── Multi-Camera State ────────────────────────────────────────────────────────
camera_sources = [{"id": 0, "name": "Main Entry - Cam 01", "source": 0}]
active_camera_id = 0

# ── Camera Thread ─────────────────────────────────────────────────────────────
_cap = None
_cap_source = None

def camera_loop():
    global _raw_frame, _cap, _cap_source, active_camera_id
    print("[DEBUG CameraLoop] Camera thread starting...")
    while True:
        src = next((c['source'] for c in camera_sources if c['id'] == active_camera_id), 0)
        
        if _cap is None or _cap_source != src:
            if _cap is not None: 
                try: _cap.release()
                except: pass
            print(f"[DEBUG CameraLoop] Opening source: {src} using DSHOW")
            _cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) # Use DirectShow on Windows for better stability
            _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            _cap_source = src
            
        success, frame = _cap.read()
        if not success:
            print(f"[ERROR CameraLoop] Failed to read from source {src}. Retrying...")
            try: _cap.release()
            except: pass
            _cap = None
            time.sleep(2)
            continue
            
        with _frame_lock:
            _raw_frame = frame.copy()
            
        time.sleep(0.03) # ~30 FPS

_cam_thread = threading.Thread(target=camera_loop, daemon=True)
_cam_thread.start()

# ── Background detection thread ────────────────────────────────────────────────
def detection_loop():
    global _raw_frame, _processed_frame, detection_status, last_screenshot_time
    print("[DEBUG DetectionLoop] Background thread starting...")
    frame_count = 0
    while True:
        try:
            with _frame_lock:
                if _raw_frame is None:
                    if frame_count % 100 == 0:
                        print("[DEBUG DetectionLoop] No raw frame yet, waiting...")
                    frame_count += 1
                    time.sleep(0.1)
                    continue
                frame = _raw_frame.copy()
            
            # ... rest of detection logic ...
            if frame_count % 100 == 0:
                print("[DEBUG DetectionLoop] Processing frame...")
            frame_count += 1
            
            # Fire detection
            fire_detected, detections = fire_detector.detect(frame)
            detection_status["fire"]["detected"] = fire_detected
            if fire_detected:
                max_conf = max([d["confidence"] for d in detections])
                detection_status["fire"]["confidence"] = round(max_conf * 100, 2)
                detection_status["fire"]["timestamp"] = time.strftime("%H:%M:%S")
                frame = fire_detector.draw_detections(frame, detections)
                current_time = time.time()
                if max_conf > 0.70 and (current_time - last_screenshot_time > 10):
                    os.makedirs("static/screenshots", exist_ok=True)
                    cv2.imwrite(f'static/screenshots/fire_{int(current_time)}.jpg', frame)
                    last_screenshot_time = current_time
            else:
                detection_status["fire"]["confidence"] = 0.0

            # Face / Attendance detection
            frame, recent_faces = face_detector.process_frame(frame)
            detection_status["attendance"]["recent_faces"] = list(face_detector._currently_visible)

            # Object / Weapon detection
            weapon_det, weapon_labels, nearby_objs, obj_dets = object_detector.detect(frame)
            detection_status["object"]["weapon"]        = weapon_det
            detection_status["object"]["weapon_labels"] = weapon_labels
            detection_status["object"]["nearby"]        = nearby_objs
            if weapon_det:
                detection_status["object"]["timestamp"] = time.strftime("%H:%M:%S")
            frame = object_detector.draw_detections(frame, obj_dets)

            with _frame_lock:
                _processed_frame = frame

        except Exception as e:
            print(f"[ERROR DetectionLoop] Exception: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

        time.sleep(0.05)

_det_thread = threading.Thread(target=detection_loop, daemon=True)
_det_thread.start()

# ── MJPEG stream ──────────────────────────────────────────────────────────────
def _stream_generator(source):
    # This now just pulls from the globally updated frames
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
    while True:
        with _frame_lock:
            if _processed_frame is not None:
                out_frame = _processed_frame
            elif _raw_frame is not None:
                out_frame = _raw_frame
            else:
                time.sleep(0.1)
                continue
        
        ret, buffer = cv2.imencode('.jpg', out_frame, encode_params)
        if not ret:
            time.sleep(0.1)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

def generate_frames():
    """Stream from the currently active camera source."""
    global active_camera_id
    src = next((c['source'] for c in camera_sources if c['id'] == active_camera_id), 0)
    yield from _stream_generator(src)

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
def cameras_page():
    return render_template('cameras.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<int:cam_id>')
def video_feed_cam(cam_id):
    """Per-camera MJPEG stream for the multi-camera grid."""
    src = next((c['source'] for c in camera_sources if c['id'] == cam_id), 0)
    return Response(_stream_generator(src), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
@login_required
def get_status():
    current_user = session.get('user')
    table = face_detector.get_today_table()
    recent = list(detection_status["attendance"]["recent_faces"])

    # Admin sees all records; non-admins only see their own
    if current_user and current_user.lower() != 'admin':
        table  = [row for row in table  if normalize_name(row['Name']) == normalize_name(current_user)]
        recent = [f   for f   in recent if normalize_name(f)           == normalize_name(current_user)]

    print(f"[DEBUG /status] User: {current_user} | Recent: {recent} | Table Len: {len(table)}")

    return jsonify({
        "fire": detection_status["fire"],
        "attendance": {"recent_faces": recent, "table": table},
        "object": detection_status["object"],
        "active_camera_id": active_camera_id,
        "cameras": camera_sources,
        "debug_sessions": face_detector.active_sessions # Added for debugging visibility
    })


# ── Camera Management API ──────────────────────────────────────────────────────
@app.route('/api/cameras', methods=['GET', 'POST'])
@login_required
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
def switch_camera():
    global active_camera_id, _processed_frame
    data = request.get_json()
    cam_id = int(data.get('id', 0))
    if any(c['id'] == cam_id for c in camera_sources):
        active_camera_id = cam_id
        with _frame_lock:
            _processed_frame = None
        return jsonify({"status": "success", "active_id": cam_id})
    return jsonify({"status": "error", "message": "Camera not found"}), 404

@app.route('/api/cameras/delete', methods=['POST'])
@login_required
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
    with _frame_lock:
        if _raw_frame is None:
            return jsonify({"status": "error", "message": "No camera feed available."}), 500
        frame = _raw_frame.copy()
    success, msg = face_detector._register_new_face(frame, name, photo_num=photo_num)
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
                name = face_detector._clean_name(f)
                if current_user == 'admin' or normalize_name(name) == normalize_name(current_user):
                    faces.append({"filename": f, "name": name})
    return jsonify({"faces": faces})

@app.route('/face_image/<filename>')
def face_image(filename):
    fname_lower = filename.lower()
    if fname_lower not in face_detector.face_images:
        return '', 404
    return send_file(io.BytesIO(face_detector.face_images[fname_lower]), mimetype='image/jpeg')

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
        face_name = face_detector._clean_name(basename)
        if current_user != 'admin' and normalize_name(face_name) != normalize_name(current_user):
            return jsonify({'status': 'error', 'message': f'Unauthorized.'}), 403
        folder = os.path.abspath('known_faces')
        filepath = os.path.join(folder, basename)
        if not os.path.exists(filepath):
            return jsonify({'status': 'error', 'message': f'File not found.'}), 404
        if basename.lower() in face_detector.face_images:
            del face_detector.face_images[basename.lower()]
        import gc; gc.collect()
        for i in range(5):
            try:
                os.remove(filepath); break
            except PermissionError:
                time.sleep(0.3 * (i + 1))
        face_detector.reload_faces()
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
        live_table = {row['Name'].strip(): row for row in face_detector.get_today_table()}
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
        if data['old_date'] == today and data['old_name'] in face_detector.active_sessions:
            sess = face_detector.active_sessions.pop(data['old_name'])
            sess.update({'date': data['new_date'], 'check_in': data['new_in'], 'check_out': data['new_out'] or None})
            if data['new_date'] == today:
                face_detector.active_sessions[data['new_name']] = sess
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
        if data['date'] == today and data['name'] in face_detector.active_sessions:
            if face_detector.active_sessions[data['name']]['check_in'] == data['check_in']:
                face_detector.active_sessions.pop(data['name'])
                face_detector._currently_visible.discard(data['name'])
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
        try:
            df = pd.read_csv(get_users_db_path())
            user_row = df[df['Username'] == username]
            if df.empty and username == 'admin' and password == 'admin':
                session['user'] = username
                return redirect(url_for('index'))
            if not user_row.empty and check_password_hash(user_row.iloc[0]['PasswordHash'], password):
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

if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5000)
