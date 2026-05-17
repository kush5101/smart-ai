import cv2
import numpy as np
import os
import urllib.request
import threading
from ultralytics import YOLO
import easyocr

class VehicleDetector:
    def __init__(self, conf_threshold=0.45):
        self.conf_threshold = conf_threshold
        
        # Base models directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Lightweight face detector to veto false-positive helmets (alt2 is better for glasses)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        
        # Number plate detector
        self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        
        # Initialize OCR Reader for Number Plates (Warning: First run downloads models)
        print("[INFO] Loading OCR Model...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        self.last_ocr_time = 0
        
        # 1. Standard YOLOv8n for Vehicles (Car, Motorcycle, Bus, Truck)
        yolo_path = os.path.join(base_dir, "..", "yolov8n.pt")
        if not os.path.exists(yolo_path):
            yolo_path = "yolov8n.pt"
        
        print("[INFO] Loading Vehicle Model...")
        self.vehicle_model = YOLO(yolo_path)
        
        # COCO mapping for vehicles
        self.vehicle_classes = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }
        
        # 2. Custom Helmet Model
        self.helmet_model_path = os.path.join(self.models_dir, "helmet_best.pt")
        self.helmet_model = None
        self._download_helmet_model()
        
    def _download_helmet_model(self):
        """Downloads a pre-trained YOLOv8 helmet detection model if not present."""
        if not os.path.exists(self.helmet_model_path):
            print("[INFO] Downloading Helmet Detection Model...")
            url = "https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt"
            try:
                urllib.request.urlretrieve(url, self.helmet_model_path)
                print("[INFO] Helmet model downloaded successfully.")
            except Exception as e:
                print(f"[WARN] Failed to download helmet model: {e}")
        
        if os.path.exists(self.helmet_model_path):
            try:
                print("[INFO] Loading Helmet Model...")
                self.helmet_model = YOLO(self.helmet_model_path)
            except Exception as e:
                print(f"[ERROR] Failed to load helmet model: {e}")
                self.helmet_model = None
                
    def _get_dominant_color(self, img):
        if img is None or img.size == 0:
            return "Unknown"
        img = cv2.resize(img, (32, 32))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        colors = {
            "Red": [([0, 50, 50], [10, 255, 255]), ([170, 50, 50], [180, 255, 255])],
            "Blue": [([100, 50, 50], [130, 255, 255])],
            "Green": [([40, 50, 50], [80, 255, 255])],
            "Yellow": [([20, 50, 50], [35, 255, 255])],
            "White": [([0, 0, 200], [180, 30, 255])],
            "Black": [([0, 0, 0], [180, 255, 30])],
            "Silver/Grey": [([0, 0, 30], [180, 30, 200])]
        }
        max_pixels = 0
        dominant = "Unknown"
        for color_name, boundaries in colors.items():
            pixels = 0
            for (lower, upper) in boundaries:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixels += cv2.countNonZero(mask)
            if pixels > max_pixels:
                max_pixels = pixels
                dominant = color_name
        return dominant if max_pixels > (32 * 32 * 0.1) else "Unknown"

    def detect(self, frame, detect_vehicle=True, detect_helmet=True, face_bboxes=None, detect_plate=True):
        results_list = []
        if frame is None: return results_list
            
        try:
            motorcycles = []
            plates_found = []
            
            # 1. VEHICLE DETECTION
            # We must detect vehicles if either vehicle detection OR plate detection is active
            if detect_vehicle or detect_plate:
                v_results = self.vehicle_model(frame, verbose=False, conf=self.conf_threshold)
            else:
                v_results = []
                
            motorcycles = []
            for r in v_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in self.vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        v_type = self.vehicle_classes[cls]
                        v_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        v_color = self._get_dominant_color(v_crop)
                        v_label = f"{v_color} {v_type}" if v_color != "Unknown" else v_type
                        
                        det = {
                            "bbox": [x1, y1, x2, y2],
                            "type": v_type,
                            "label": v_label,
                            "confidence": round(conf * 100),
                            "helmet": "N/A",
                            "plate": "N/A"
                        }
                        results_list.append(det)
                        if v_type == "Motorcycle":
                            motorcycles.append(det)

                        # Detect Number Plate using pure OCR inside the vehicle crop
                        if detect_plate and v_crop.size > 0:
                            # 1. Image Enhancement for higher OCR accuracy
                            v_gray = cv2.cvtColor(v_crop, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            v_enhanced = clahe.apply(v_gray)
                            
                            # 2. Upscale 2x (Text resolution is critical for OCR algorithms)
                            v_enhanced = cv2.resize(v_enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            
                            # 3. Direct OCR with an explicit characters allow-list to prevent hallucinating noise
                            # PERFORMANCE: Only run OCR once every 2 seconds to reduce lag
                            import time as t_perf
                            if t_perf.time() - self.last_ocr_time > 2.0:
                                ocr_results = self.ocr_reader.readtext(v_enhanced, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -')
                                self.last_ocr_time = t_perf.time()
                            else:
                                ocr_results = []
                            
                            for bbox, text, prob in ocr_results:
                                # Keep text that has at least 2 characters and okay confidence
                                # Lowered prob threshold slightly because allow-list inherently filters bad reads
                                if prob > 0.15 and len(text) >= 2:
                                    x_coords = [int(p[0]) for p in bbox]
                                    y_coords = [int(p[1]) for p in bbox]
                                    # Remember to downscale the bbox coords mapping them back to the original crop size
                                    px = int(min(x_coords) / 2.0)
                                    py = int(min(y_coords) / 2.0)
                                    pw = int((max(x_coords) - min(x_coords)) / 2.0)
                                    ph = int((max(y_coords) - min(y_coords)) / 2.0)
                                    
                                    plate_text_clean = text.upper()
                                    
                                    # Translate coordinates to full frame
                                    pl_x1 = x1 + px
                                    pl_y1 = y1 + py
                                    pl_x2 = x1 + px + pw
                                    pl_y2 = y1 + py + ph
                                    
                                    det["plate"] = plate_text_clean
                                    
                                    results_list.append({
                                        "bbox": [pl_x1, pl_y1, pl_x2, pl_y2],
                                        "type": "NumberPlate",
                                        "label": plate_text_clean,
                                        "confidence": int(prob * 100),
                                        "helmet": "N/A"
                                    })

            # 2. STANDALONE HELMET/HEAD DETECTION
            if detect_helmet and self.helmet_model is not None:
                # Use a very strict threshold for standalone safety items (0.75) to avoid bare heads
                h_results = self.helmet_model(frame, verbose=False, conf=0.3)
                
                for hr in h_results:
                    for h_box in hr.boxes:
                        hx1, hy1, hx2, hy2 = map(int, h_box.xyxy[0])
                        h_conf = float(h_box.conf[0])
                        h_cls = int(h_box.cls[0])
                        h_label = "Helmet" if h_cls == 0 else "No Helmet"
                        
                        # ANTI-FALSE-POSITIVE: Verify it's not actually just a bare face
                        if h_label == "Helmet":
                            vetoed = False
                            
                            # 1. Use high-accuracy Face Bboxes passed from App
                            if face_bboxes is not None:
                                for f_det in face_bboxes:
                                    fx1, fy1, fx2, fy2 = f_det["bbox"]
                                    ix1 = max(hx1, fx1)
                                    iy1 = max(hy1, fy1)
                                    ix2 = min(hx2, fx2)
                                    iy2 = min(hy2, fy2)
                                    if ix2 > ix1 and iy2 > iy1:
                                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                                        helmet_area = max(1, (hx2 - hx1) * (hy2 - hy1))
                                        if inter_area / helmet_area > 0.05:
                                            h_label = "No Helmet"
                                            h_conf = 0.99
                                            vetoed = True
                                            break
                                            
                            # 2. Fallback to Haarcascade for crop if no face passed
                            if not vetoed:
                                h_crop = frame[max(0, hy1):min(frame.shape[0], hy2), max(0, hx1):min(frame.shape[1], hx2)]
                                if h_crop.size > 0:
                                    gray_crop = cv2.cvtColor(h_crop, cv2.COLOR_BGR2GRAY)
                                    # Relaxed params (minNeighbors=1 instead of 3) to be ultra sensitive and over-veto bare heads
                                    faces = self.face_cascade.detectMultiScale(gray_crop, scaleFactor=1.05, minNeighbors=1, minSize=(20, 20))
                                    
                                    for (fx, fy, fw, fh) in faces:
                                        face_area = fw * fh
                                        helmet_area = max(1, (hx2 - hx1) * (hy2 - hy1))
                                        if face_area / helmet_area > 0.05:
                                            h_label = "No Helmet"
                                            h_conf = 0.99
                                            break
                                        
                        # ASSOCIATION LOGIC
                        is_near_motorcycle = False
                        for m in motorcycles:
                            mx1, my1, mx2, my2 = m["bbox"]
                            m_top = my1 - int((my2 - my1) * 0.4)
                            m_bottom = my1 + int((my2 - my1) * 0.5)
                            
                            if hx2 > mx1 and hx1 < mx2 and hy2 > m_top and hy1 < m_bottom:
                                is_near_motorcycle = True
                                if h_label == "Helmet":
                                    m["helmet"] = "YES"
                                elif h_label == "No Helmet" and m["helmet"] != "YES":
                                    m["helmet"] = "NO"
                        
                        # STANDALONE DISPLAY LOGIC
                        # Only show standalone helmets if confidence is > 75% OR it's near a motorcycle
                        if h_conf > 0.75 or is_near_motorcycle:
                            results_list.append({
                                "bbox": [hx1, hy1, hx2, hy2],
                                "type": "SafetyItem",
                                "label": h_label,
                                "confidence": round(h_conf * 100),
                                "helmet": "N/A"
                            })
                        
            # Final pass for motorcyclists missing helmet status
            if detect_helmet:
                for m in motorcycles:
                    if m["helmet"] == "N/A":
                        m["helmet"] = "NO"
                        
        except Exception as e:
            print(f"[ERROR VehicleDetector] {e}")
            
        return results_list

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['label']} {det['confidence']}%"
            color = (255, 165, 0)
            
            if det['type'] == 'Motorcycle':
                if det['helmet'] == 'YES':
                    color, label = (0, 255, 0), label + " | Helmet: YES"
                else: # NO or N/A
                    color, label = (0, 0, 255), label + " | Helmet: NO"
            elif det['type'] == 'SafetyItem':
                color = (0, 255, 0) if det['label'] == 'Helmet' else (0, 0, 255)
            elif det['type'] == 'NumberPlate':
                color, label = (255, 0, 0), f"{det['label']}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
        return frame
