import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import threading

class FaceAttendanceSystem:
    def _clean_name(self, raw_input):
        """Standardize names: 'Sawan_1.jpg' or 'Sawan ' -> 'Sawan'"""
        if not raw_input: return ""
        name = os.path.splitext(raw_input)[0] 
        name = name.replace("_", " ")
        name = "".join([c for c in name if not c.isdigit()])
        return name.strip().title()

    def __init__(self, known_faces_dir='known_faces', db_path='database/attendance.csv'):
        self.known_faces_dir = known_faces_dir
        self.db_path = db_path
        self.lock = threading.RLock()

        # Models Path
        model_dir = os.path.join(os.getcwd(), "models")
        detect_model_path = os.path.join(model_dir, "face_detection_yunet.onnx")
        recog_model_path = os.path.join(model_dir, "face_recognition_sface.onnx")

        # 1. YuNet Face Detector
        # Positional args: model, config, inputSize
        self.detector = cv2.FaceDetectorYN.create(
            detect_model_path,
            "",
            (640, 480), # Default, will update per frame
            0.50,      # score_threshold (Increased for higher quality faces)
            0.3,       # nms_threshold
            5000       # top_k
        )

        # 2. SFace Face Recognizer
        # Positional args: model, config
        self.recognizer = cv2.FaceRecognizerSF.create(
            recog_model_path,
            ""
        )

        self.known_face_names = []      # index -> name
        self.known_face_features = []   # list of 128D vectors
        self.face_images = {}           # { filename: bytes } for serving

        # Check-In / Check-Out state
        self.active_sessions = {}
        self._load_active_sessions_from_db()

        self._currently_visible = set()
        self._last_seen = {}
        self._absent_counts = {}
        self.ABSENT_THRESH = 45

        self.today_str = datetime.now().strftime('%Y-%m-%d')

        self._init_db()
        self._load_known_faces()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            pd.DataFrame(columns=['Name', 'Date', 'Check-In', 'Check-Out']).to_csv(
                self.db_path, index=False)

    def _load_known_faces(self):
        os.makedirs(self.known_faces_dir, exist_ok=True)
        self.known_face_names = []
        self.known_face_features = []
        
        print("Loading known faces with YuNet + SFace...")
        filenames = sorted(os.listdir(self.known_faces_dir))
        
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(self.known_faces_dir, filename)
                clean_name = self._clean_name(filename)

                # Cache raw bytes for UI
                with open(filepath, 'rb') as f:
                    self.face_images[filename.lower()] = f.read()

                img = cv2.imread(filepath)
                if img is None: continue
                
                # Detect and extract feature vector
                self.detector.setInputSize((img.shape[1], img.shape[0]))
                _, faces = self.detector.detect(img)
                
                if faces is not None:
                    # Align and extract feature
                    aligned_face = self.recognizer.alignCrop(img, faces[0])
                    feature = self.recognizer.feature(aligned_face)
                    
                    self.known_face_names.append(clean_name)
                    self.known_face_features.append(feature)
                    print(f"[DEBUG load_faces] Success: {clean_name}")

        print(f"SFace ready with {len(self.known_face_features)} registered faces.")

    def reload_faces(self):
        with self.lock:
            self.known_face_names = []
            self.known_face_features = []
            self.face_images = {}
            self._last_seen = {}
            self._absent_counts = {}
            self._currently_visible = set()
            self._load_known_faces()

    def _load_active_sessions_from_db(self):
        try:
            if os.path.exists(self.db_path):
                df = pd.read_csv(self.db_path)
                if df.empty: return
                today = datetime.now().strftime('%Y-%m-%d')
                for _, row in df[df['Date'] == today].iterrows():
                    name = str(row['Name']).strip()
                    cout = row.get('Check-Out', '')
                    if pd.isna(cout) or str(cout).strip() == '' or cout == '—': cout = None
                    self.active_sessions[name] = {
                        'date': today, 'check_in': row['Check-In'], 'check_out': cout
                    }
        except Exception as e:
            print(f"Error loading active sessions: {e}")

    def _now_str(self):
        return datetime.now().strftime('%H:%M:%S')

    def _check_in(self, name):
        today = datetime.now().strftime('%Y-%m-%d')
        with self.lock:
            if name in self.active_sessions and self.active_sessions[name]['date'] == today:
                if self.active_sessions[name].get('check_out'):
                    self.active_sessions[name]['check_out'] = None
                    self._save_session(name, self.active_sessions[name])
                return
            
            session = {'date': today, 'check_in': self._now_str(), 'check_out': None}
            self.active_sessions[name] = session
            self._save_session(name, session)

    def _check_out(self, name):
        if name not in self.active_sessions: return
        session = self.active_sessions[name]
        if session['check_out'] is not None: return
        session['check_out'] = self._last_seen.get(name, self._now_str())
        self._save_session(name, session)

    def _save_session(self, name, session):
        try:
            if os.path.exists(self.db_path):
                df = pd.read_csv(self.db_path, dtype=str).fillna('')
                mask = (df['Name'] == name) & (df['Date'] == session['date'])
                if mask.any():
                    df.loc[mask, 'Check-In'] = session['check_in']
                    df.loc[mask, 'Check-Out'] = session['check_out'] or ''
                    df.to_csv(self.db_path, index=False)
                    return
            row = pd.DataFrame([[name, session['date'], session['check_in'], session['check_out'] or '']], 
                               columns=['Name', 'Date', 'Check-In', 'Check-Out'])
            row.to_csv(self.db_path, mode='a', header=not os.path.exists(self.db_path), index=False)
        except Exception as e:
            print(f"DB write error: {e}")

    def get_today_table(self):
        today = datetime.now().strftime('%Y-%m-%d')
        rows = []
        with self.lock:
            for name, s in self.active_sessions.items():
                if s.get('date') == today:
                    rows.append({'Name': name, 'Check-In': s['check_in'], 'Check-Out': s['check_out'] or '—'})
        return rows

    def process_frame(self, frame):
        # 1. Detection
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        
        if faces is None:
            if h > 0: # Only print if we actually have a frame
                pass # Already silent
        else:
            print(f"[DEBUG FaceAttendance] Found {len(faces)} potential faces in {w}x{h} frame at {self._now_str()}")
        
        seen_this_frame = set()

        if faces is not None:
            for face in faces:
                # face is [x, y, w, h, eye1_x, eye1_y, ... score]
                coords = face[:4].astype(int)
                x, y, fw, fh = coords
                score = face[-1]
                
                if score < 0.50: continue # Low confidence face
                
                print(f"[DEBUG FaceAttendance] Face detected with score: {score:.4f}")

                # 2. Alignment & Recognition
                aligned_face = self.recognizer.alignCrop(frame, face)
                feature = self.recognizer.feature(aligned_face)

                name = "Unknown"
                name_display = "Unknown"
                max_sim = 0.0

                for idx, known_feat in enumerate(self.known_face_features):
                    # match score (Cosine Similarity)
                    sim = self.recognizer.match(feature, known_feat, cv2.FACE_RECOGNIZER_SF_FR_COSINE)
                    if sim > max_sim:
                        max_sim = sim
                        name = self.known_face_names[idx]
                
                print(f"[DEBUG FaceAttendance] Face sim: {max_sim:.4f} for {name} (score: {score:.4f})")
                
                if name != "Unknown" and max_sim > 0.40: # STRICTER match threshold
                    # Map 0.40 - 1.0 to 0 - 100%
                    conf_pct = min(100, int(((max_sim - 0.4) / 0.6) * 100))
                    name_display = f"{name} {conf_pct}%"
                    seen_this_frame.add(name)
                    self._last_seen[name] = self._now_str()
                    print(f"[DEBUG FaceAttendance] Recognized: {name} (sim: {max_sim:.4f})")
                else:
                    # Optionally show 0% or low score if match < 0.4
                    name_display = f"Unknown"

                color = (0, 220, 0) if name_display != "Unknown" else (0, 0, 220)
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
                cv2.rectangle(frame, (x, y+fh-30), (x+fw, y+fh), color, cv2.FILLED)
                cv2.putText(frame, name_display, (x+5, y+fh-7), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

        # Check-In / Check-Out
        for name in seen_this_frame:
            self._absent_counts.pop(name, None)
            if name not in self._currently_visible:
                self._check_in(name)
                self._currently_visible.add(name)

        just_left = set(self._currently_visible) - seen_this_frame
        for name in just_left:
            self._absent_counts[name] = self._absent_counts.get(name, 0) + 1
            if self._absent_counts[name] >= self.ABSENT_THRESH:
                self._check_out(name)
                self._absent_counts.pop(name, None)
                self._currently_visible.remove(name)

        return frame, list(seen_this_frame)

    def _register_new_face(self, frame, name, photo_num=1):
        try:
            os.makedirs(self.known_faces_dir, exist_ok=True)
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)

            if faces is None or len(faces) == 0:
                print("Registration failed: No face detected.")
                return False, "No face detected. Please face the camera clearly."

            # Align and save (saving aligned face improves recognizer speed later)
            aligned_face = self.recognizer.alignCrop(frame, faces[0])
            
            clean_name = self._clean_name(name)
            filename = f"{clean_name.replace(' ', '_')}_{photo_num}.jpg"
            filepath = os.path.join(self.known_faces_dir, filename)
            cv2.imwrite(filepath, aligned_face)
            
            self.reload_faces()
            return True, f"Photo {photo_num}/3 saved."
        except Exception as e:
            return False, str(e)
