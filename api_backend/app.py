from flask import Flask, render_template, Response, jsonify, request, send_file, session, redirect, url_for, flash
import cv2
import time
import os
import io
import threading
import pandas as pd
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sys
from pathlib import Path

# Add project root to path for modular imports
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from detection_models.fire_detection import FireDetector
from detection_models.face_attendance import FaceAttendanceSystem
from detection_models.object_detector import ObjectDetector
from camera_manager.manager import CameraManager

# Point Flask to the parent directory for templates and static files
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent.parent / 'templates'),
    static_folder=str(Path(__file__).parent.parent / 'static')
)
app.secret_key = 'super-secret-smart-ai-key-change-in-production'

def normalize_name(name):
    """Normalize names for comparison: 'Admin Admin' -> 'adminadmin'"""
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
# Define paths relative to the project root (parent of api_backend)
project_root = Path(__file__).parent.parent
known_faces_path = str(project_root / 'known_faces')
attendance_db_path = str(project_root / 'database' / 'attendance.csv')

fire_detector    = FireDetector(confidence_threshold=0.6)
face_detector    = FaceAttendanceSystem(
    known_faces_dir=known_faces_path,
    db_path=attendance_db_path
)
object_detector  = ObjectDetector()

# ── Shared state ──────────────────────────────────────────────────────────────
detection_status = {
    "fire":       {"detected": False, "confidence": 0.0, "timestamp": None},
    "attendance": {"recent_faces": []},
    "object":     {"weapon": False, "weapon_labels": [], "nearby": [], "timestamp": None}
}

active_camera_id = 0 # Track active camera globally
_raw_frame = None
_processed_frame = None
_current_frame_cache = None
_frame_lock = threading.Lock()

# ── Multi-Camera Management ──────────────────────────────────────────────────
camera_manager = CameraManager()
# Default local camera
camera_manager.add_source(0, 0, "Main Entry - Cam 01")

# Global detection results per camera
global_results = {}

# ── Background detection thread (runs BOTH detectors simultaneously) ───────────
def detection_loop():
    """Continuously runs AI models on ALL active camera streams."""
    global global_results
    while True:
        cams = camera_manager.list_cameras()
        for cam in cams:
            cam_id = cam["id"]
            frame = camera_manager.get_frame(cam_id)
            if frame is None: continue

            # Create entry if missing
            if cam_id not in global_results:
                global_results[cam_id] = {
                    "fire": {"detected": False, "confidence": 0, "timestamp": None},
                    "attendance": {"recent_faces": []},
                    "object": {"weapon": False, "weapon_labels": [], "nearby": [], "timestamp": None},
                    "processed_frame": None
                }

            # ── 1. Fire Detection ──
            fire_detected, detections = fire_detector.detect(frame)
            global_results[cam_id]["fire"]["detected"] = fire_detected
            if fire_detected:
                max_conf = max([d["confidence"] for d in detections])
                global_results[cam_id]["fire"]["confidence"] = round(max_conf * 100, 2)
                global_results[cam_id]["fire"]["timestamp"] = time.strftime("%H:%M:%S")
                frame = fire_detector.draw_detections(frame, detections)

            # ── 2. Face / Attendance ──
            frame, recent_faces = face_detector.process_frame(frame)
            global_results[cam_id]["attendance"]["recent_faces"] = recent_faces

            # ── 3. Object / Weapon ──
            weapon_det, labels, nearby, obj_dets = object_detector.detect(frame)
            global_results[cam_id]["object"]["weapon"]        = weapon_det
            global_results[cam_id]["object"]["weapon_labels"] = labels
            global_results[cam_id]["object"]["nearby"]        = nearby
            frame = object_detector.draw_detections(frame, obj_dets)

            global_results[cam_id]["processed_frame"] = frame

        time.sleep(0.01) # Rapid iteration across streams


_det_thread = threading.Thread(target=detection_loop, daemon=True)
_det_thread.start()

# ── MJPEG stream ──────────────────────────────────────────────────────────────
@app.route('/video_feed/<int:cam_id>')
def video_feed_cam(cam_id):
    def generate(cid):
        print(f"[Streaming] Started stream for camera {cid}")
        while True:
            # Prefer processed frame with detections
            res = global_results.get(cid)
            frame = res.get("processed_frame") if res else None
            
            # Fallback to raw frame
            if frame is None:
                frame = camera_manager.get_frame(cid)
            
            if frame is None:
                # print(f"[Streaming] No frame for camera {cid}")
                time.sleep(0.1)
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                print(f"[Streaming] FAILED to encode frame for camera {cid}")
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

    return Response(generate(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

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

# Default feed route (endpoint 'video_feed' for backward compat)
@app.route('/video_feed')
def video_feed():
    cams = camera_manager.list_cameras()
    cid = cams[0]['id'] if cams else 0
    print(f"[Video Feed] Serving default camera {cid}")
    
    def _gen(target_id):
        while True:
            res = global_results.get(target_id)
            frame = res.get("processed_frame") if res else None
            
            if frame is None:
                frame = camera_manager.get_frame(target_id)
            
            if frame is None:
                time.sleep(0.1)
                continue
                
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret: continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.04)
            
    return Response(_gen(cid), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
@login_required
def get_status():
    current_user = session.get('user')
    table = face_detector.get_today_table()
    
    # Filter table for non-admin
    if current_user != 'admin':
        table = [row for row in table if normalize_name(row['Name']) == normalize_name(current_user)]
        
    # Filter recent_faces for non-admin
    recent = detection_status["attendance"]["recent_faces"]
    if current_user != 'admin':
        recent = [f for f in recent if normalize_name(f) == normalize_name(current_user)]

    return jsonify({
        "fire": detection_status["fire"],
        "attendance": {
            "recent_faces": recent,
            "table": table
        },
        "object": detection_status["object"],
        "active_camera_id": active_camera_id,
        "cameras": camera_manager.list_cameras()
    })

@app.route('/api/cameras', methods=['GET', 'POST'])
@login_required
def manage_cameras():
    global camera_manager
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name', 'New Camera')
        source = data.get('source', 0)
        # Convert numeric string sources to int
        try:
            if str(source).isdigit():
                source = int(source)
        except: pass
        cam_id = int(time.time())
        camera_manager.add_source(cam_id, source, name)
        return jsonify({'status': 'success', 'camera': {'id': cam_id, 'name': name, 'source': source}})
    
    cameras_list = camera_manager.list_cameras()
    return jsonify({'cameras': cameras_list, 'active_id': active_camera_id})

@app.route('/api/cameras/switch', methods=['POST'])
@login_required
def switch_camera():
    # With multi-cam grid, switching updates which cam is shown in the monitor
    data = request.get_json()
    return jsonify({'status': 'success', 'active_id': data.get('id', 0)})

@app.route('/api/cameras/delete', methods=['POST'])
@login_required
def delete_camera():
    global camera_manager
    data = request.get_json()
    cam_id = int(data.get('id', -1))
    if cam_id == 0:
        return jsonify({'status': 'error', 'message': 'Cannot delete default camera'}), 400
    camera_manager.sources.pop(cam_id, None)
    camera_manager.frames.pop(cam_id, None)
    global_results.pop(cam_id, None)
    return jsonify({'status': 'success'})

@app.route('/export_attendance')
def export_attendance():
    """Stream the attendance CSV as a properly formatted Excel download."""
    import pandas as pd
    db_path = os.path.join('database', 'attendance.csv')
    if not os.path.exists(db_path):
        return jsonify({"status": "error", "message": "No attendance data yet."}), 404
    
    try:
        df = pd.read_csv(db_path)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
            # Auto-fit column widths
            ws = writer.sheets['Attendance']
            for col in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = max_len + 4
        output.seek(0)
        date_str = __import__('datetime').datetime.now().strftime('%Y-%m-%d')
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'attendance_{date_str}.xlsx'
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/register_face', methods=['POST'])
@login_required
def register_face():
    """Register a new face from the live video feed."""
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
    # Pass whether they are admin to the template so it can hide the name field
    is_admin = session.get('user') == 'admin'
    return render_template('faces.html', is_admin=is_admin)

@app.route('/known_faces_list')
@login_required
def known_faces_list():
    """Return list of registered faces. Scoped to the current user."""
    current_user = session.get('user')
    faces = []
    folder = 'known_faces'
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                # Use central cleaning helper
                name = face_detector._clean_name(f)
                
                # If admin, show all. If normal user, only show their own face.
                if current_user == 'admin' or normalize_name(name) == normalize_name(current_user):
                    faces.append({"filename": f, "name": name})
    return jsonify({"faces": faces})

@app.route('/face_image/<filename>')
def face_image(filename):
    """Serve a face image safely from memory to avoid Windows file locks."""
    fname_lower = filename.lower()
    if fname_lower not in face_detector.face_images:
        return '', 404
    
    import io
    return send_file(
        io.BytesIO(face_detector.face_images[fname_lower]), 
        mimetype='image/jpeg'
    )

@app.route('/delete_face', methods=['POST'])
@login_required
def delete_face():
    """Delete a registered face image and reload the detector."""
    try:
        data = request.get_json()
        filename = (data or {}).get('filename', '')
        print(f"DEBUG: Deletion request for {filename}")
        basename = os.path.basename(filename)
        if not basename or basename != filename:
            print(f"DEBUG: Invalid filename validation failed: {basename} vs {filename}")
            return jsonify({'status': 'error', 'message': 'Invalid filename.'}), 400

        current_user = session.get('user')
        face_name = face_detector._clean_name(basename)
        
        # Security check: User can only delete their own face unless they are admin
        if current_user != 'admin' and normalize_name(face_name) != normalize_name(current_user):
            print(f"DEBUG: Security check FAILED. User:{current_user} != Face:{face_name}")
            return jsonify({'status': 'error', 'message': f'Unauthorized. Only {face_name} or admin can delete this.'}), 403

        folder = os.path.abspath('known_faces')
        filepath = os.path.join(folder, basename)
        print(f"DEBUG: Attempting to delete {filepath}")

        if not os.path.exists(filepath):
            print(f"DEBUG: File not found: {filepath}")
            return jsonify({'status': 'error', 'message': f'File not found: {filepath}'}), 404

        # Important: Clear image from memory cache before deleting file to release potential locks
        if basename.lower() in face_detector.face_images:
            print(f"DEBUG: Removing {basename} from detector cache")
            del face_detector.face_images[basename.lower()]

        import time
        import gc
        gc.collect()

        # Retry loop for Windows file locks
        for i in range(5):
            try:
                os.remove(filepath)
                print(f"DEBUG: Successfully removed {filepath}")
                break
            except PermissionError as e:
                print(f"DEBUG: PermissionError on attempt {i+1}: {e}")
                time.sleep(0.3 * (i + 1))
        else:
            print(f"DEBUG: All deletion attempts failed")
            try:
                os.remove(filepath)
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Windows File Lock: {str(e)}. Try again in a moment.'}), 500

        print(f"DEBUG: Reloading faces...")
        face_detector.reload_faces()
        print(f"DEBUG: Face detector reloaded.")
        return jsonify({'status': 'success', 'message': f'Removed {basename}'})
    except Exception as e:
        print(f"DEBUG: Global exception in delete_face: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/attendance')
@login_required
def attendance_page():
    return render_template('attendance.html')

@app.route('/all_attendance')
@login_required
def all_attendance():
    """Return full attendance CSV as JSON for the attendance page. Scoped by user."""
    import pandas as pd
    db_path = os.path.join('database', 'attendance.csv')
    if not os.path.exists(db_path):
        return jsonify({"records": []})
    try:
        current_user = session.get('user')
        import pandas as pd
        import datetime as dt
        today = dt.datetime.now().strftime('%Y-%m-%d')

        df = pd.read_csv(db_path)
        df = df.fillna('')
        
        # Scope standard users to only see themselves
        if current_user != 'admin':
            df = df[df['Name'].apply(lambda x: "".join(str(x).split()).lower() == "".join(current_user.split()).lower())]

        records = df.to_dict(orient='records')

        # Build a map of today's live sessions
        live_table = {row['Name'].strip(): row for row in face_detector.get_today_table()}
        
        # Filter live sessions for non-admin
        if current_user != 'admin':
            live_table = {k: v for k, v in live_table.items() if "".join(k.split()).lower() == "".join(current_user.split()).lower()}

        # Overlay live data on top of CSV for today's rows
        for r in records:
            if r.get('Date') == today and r['Name'] in live_table:
                live = live_table.pop(r['Name'])
                r['Check-In']  = live['Check-In']  or r.get('Check-In', '')
                r['Check-Out'] = live['Check-Out'] if live['Check-Out'] != '—' else (r.get('Check-Out', '') or '—')

        # Add any live sessions not yet in CSV
        for name, row in live_table.items():
            records.insert(0, {
                'Name': name,
                'Date': today,
                'Check-In': row['Check-In'],
                'Check-Out': row['Check-Out']
            })

        # Replace empty strings with '—' for display
        for r in records:
            for k in ('Check-In', 'Check-Out'):
                if not r.get(k):
                    r[k] = '—'

        return jsonify({"records": records})
    except Exception as e:
        return jsonify({"error": str(e), "records": []})

@app.route('/update_record', methods=['POST'])
def update_record():
    """Update an attendance record in the CSV and active_sessions if applicable."""
    try:
        data = request.get_json()
        import pandas as pd
        db_path = os.path.join('database', 'attendance.csv')
        df = pd.read_csv(db_path) if os.path.exists(db_path) else pd.DataFrame(columns=['Name', 'Date', 'Check-In', 'Check-Out'])
        
        # 1. Update CSV
        mask = (df['Name'] == data['old_name']) & (df['Date'] == data['old_date']) & (df['Check-In'] == data['old_in'])
        if mask.any():
            idx = mask.idxmax()
            df.loc[idx, 'Name'] = data['new_name']
            df.loc[idx, 'Date'] = data['new_date']
            df.loc[idx, 'Check-In'] = data['new_in']
            df.loc[idx, 'Check-Out'] = data['new_out']
            df.to_csv(db_path, index=False)
        else:
            # If not in CSV yet (pure live session), we'll save it now
            new_row = pd.DataFrame([{
                'Name': data['new_name'], 'Date': data['new_date'],
                'Check-In': data['new_in'], 'Check-Out': data['new_out']
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(db_path, index=False)
            
        # 2. Update active in-memory session if it's today's live session
        import datetime as dt
        today = dt.datetime.now().strftime('%Y-%m-%d')
        if data['old_date'] == today and data['old_name'] in face_detector.active_sessions:
            # We remove old name key and replace with new if name changed
            session = face_detector.active_sessions.pop(data['old_name'])
            session['date'] = data['new_date']
            session['check_in'] = data['new_in']
            session['check_out'] = data['new_out'] or None
            # Only put back if still today
            if data['new_date'] == today:
                face_detector.active_sessions[data['new_name']] = session

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/delete_record', methods=['POST'])
@login_required
def delete_record():
    """Delete an attendance record from the CSV and active_sessions."""
    try:
        data = request.get_json()
        current_user = session.get('user')
        
        # Security: User is only allowed to delete their own records
        if current_user != 'admin' and normalize_name(data['name']) != normalize_name(current_user):
            return jsonify({'status': 'error', 'message': 'Unauthorized.'}), 403

        import pandas as pd
        db_path = os.path.join('database', 'attendance.csv')
        if os.path.exists(db_path):
            df = pd.read_csv(db_path)
            # Filter out the matching row
            mask = (df['Name'] == data['name']) & (df['Date'] == data['date']) & (df['Check-In'] == data['check_in'])
            df = df[~mask]
            df.to_csv(db_path, index=False)
            
        # Remove from live active_sessions if present
        import datetime as dt
        today = dt.datetime.now().strftime('%Y-%m-%d')
        if data['date'] == today and data['name'] in face_detector.active_sessions:
            s_in = face_detector.active_sessions[data['name']]['check_in']
            if s_in == data['check_in']:
                face_detector.active_sessions.pop(data['name'])
                face_detector._currently_visible.discard(data['name'])

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ── Authentication Routes ──────────────────────────────────────────────────────
def get_users_db_path():
    os.makedirs('database', exist_ok=True)
    return os.path.join('database', 'users.csv')

def init_users_db():
    db_path = get_users_db_path()
    if not os.path.exists(db_path):
        pd.DataFrame(columns=['Username', 'PasswordHash']).to_csv(db_path, index=False)
    
    # Ensure default admin always exists
    df = pd.read_csv(db_path)
    if 'admin' not in df['Username'].values:
        hashed = generate_password_hash('admin')
        new_row = pd.DataFrame([{'Username': 'admin', 'PasswordHash': hashed}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(db_path, index=False)

init_users_db()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        try:
            df = pd.read_csv(get_users_db_path())
            user_row = df[df['Username'] == username]
            
            # Allow default admin if DB is completely empty (first run fallback)
            if df.empty and username == 'admin' and password == 'admin':
                session['user'] = username
                return redirect(url_for('index'))
                
            if not user_row.empty:
                stored_hash = user_row.iloc[0]['PasswordHash']
                if check_password_hash(stored_hash, password):
                    session['user'] = username
                    next_page = request.args.get('next')
                    return redirect(next_page or url_for('index'))
                    
            flash('Invalid username or password', 'error')
        except Exception as e:
            flash(f'Error accessing database: {e}', 'error')
            
    return render_template('login.html')

@app.route('/register_user', methods=['GET', 'POST'])
def register_user():
    # Registration is OPEN to allow new employees to sign up
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
            flash('Username already exists. Please choose another.', 'error')
        else:
            hashed = generate_password_hash(password)
            new_row = pd.DataFrame([{'Username': username, 'PasswordHash': hashed}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(get_users_db_path(), index=False)
            flash(f'Account created successfully! Welcome, {username}.', 'success')
            
            # Auto-login newly created users
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
    # Don't let admin delete themselves in the UI
    users = df[df['Username'] != 'admin'].to_dict(orient='records')
    return render_template('users.html', users=users)

@app.route('/update_user_password', methods=['POST'])
@login_required
def update_user_password():
    if session.get('user') != 'admin':
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    data = request.get_json()
    username = data.get('username')
    new_password = data.get('password')
    
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
        df = df[df['Username'] != username]
        df.to_csv(get_users_db_path(), index=False)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'User not found'}), 404

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
