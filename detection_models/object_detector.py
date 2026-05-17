import cv2
import numpy as np
import os

# ── Object Detection Constants ────────────────────────────────────────────────
# Standard COCO classes that we'll treat as weapons
WEAPON_CLASSES = {
    43: "knife",
    76: "scissors"
}

# Potential "proxy" classes for guns (mismatched in COCO)
GUN_PROXIES = {
    67: "cell phone",
    65: "remote"
}

# Classes to SKIP for "nearby" detection
SKIP_NEARBY = {0} # 0 = person

# Confidence thresholds
WEAPON_CONF   = 0.65   # Increased from 0.40 to reduce false positives
LIGHTER_CONF  = 0.45
NEARBY_CONF   = 0.30   # Lowered to catch more everyday objects
# Gun proxies (cell phone/remote) need VERY high confidence to be a threat
# This prevents fire glow from being mislabeled as a phone/gun at low conf.
GUN_PROXY_CONF = 0.85
NEARBY_AREA_RATIO = 0.01 # 1% of frame to detect smaller/distant items


class ObjectDetector:
    def __init__(self, model_path=None):
        from ultralytics import YOLO
        if model_path is None:
            # Try relative path from smart_ai_monitoring/
            candidates = [
                os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt'),
                os.path.join(os.path.dirname(__file__), 'yolov8n.pt'),
                'yolov8n.pt',
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), 'yolov8n.pt')

        print(f"[ObjectDetector] Loading YOLO model from: {os.path.abspath(model_path)}")
        self.model = YOLO(model_path)
        self.model.fuse()   # slight inference speed-up
        print("[ObjectDetector] Model ready.")

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, frame, fire_bboxes=None):
        """
        Run YOLOv8 on frame.
        Args:
            frame: cv2 BGR image
            fire_bboxes (list[dict]): Optional list of {'bbox': [x1,y1,x2,y2]} from FireDetector
        Returns:
            weapon_detected (bool)
            weapon_labels   (list[str])
            nearby_objects  (list[str])
            all_detections  (list[dict])
        """
        h, w = frame.shape[:2]
        frame_area = h * w

        # Use imgsz=480 for better accuracy (2.25x more pixels than 320)
        results = self.model(frame, verbose=False, imgsz=480)[0]

        weapon_detected = False
        weapon_labels   = []
        nearby_objects  = []
        all_detections  = []

        for box in results.boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.model.names[cls_id]
            
            w_box = x2 - x1
            h_box = y2 - y1
            area_pct = (w_box * h_box) / frame_area
            aspect_ratio = w_box / h_box if h_box != 0 else 0

            # ── 1. Color Check (Fire Avoidance) ──────────────────────────────────────
            # Optimized: Resize to 20x20 for ultra-fast color analysis
            is_candidate = (cls_id in WEAPON_CLASSES or cls_id in GUN_PROXIES)
            if is_candidate and 0 < y2 < h and 0 < x2 < w:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_small = cv2.resize(roi, (20, 20))
                    hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([0, 80, 100]), np.array([30, 255, 255]))
                    if cv2.countNonZero(mask) / 400.0 > 0.45:
                        continue

            # ── 2. Weapon/Threat Check ───────────────────────────────────────
            is_threat = False
            label_override = None

            # Actual weapon (knife/scissors)
            if cls_id in WEAPON_CLASSES and conf >= WEAPON_CONF:
                is_threat = True

            # (Removed the 'class 39 = LIGHTER' hack so bottles are just detected as normal bottles)

            # Gun Heuristic (Cell phone/Remote proxies)
            elif cls_id in GUN_PROXIES and conf >= GUN_PROXY_CONF:
                if (0.3 < aspect_ratio < 3.0) and 0.015 < area_pct < 0.60:
                    is_threat = True
                    label_override = f"POSSIBLE FIREARM ({class_name}) {conf:.0%}"

            if is_threat:
                # ── EXTREME Fire Avoidance: If ANY fire is in the frame, we are suspicious ──
                # If the 'weapon' is anywhere near fire, skip it entirely.
                if fire_bboxes:
                    for fbox in fire_bboxes:
                        fx1, fy1, fx2, fy2 = fbox["bbox"]
                        # Loose containment check: Is weapon box center inside or near fire?
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        # If center is near fire (padding of 50px), it's likely a false positive
                        if (fx1 - 50 < cx < fx2 + 50) and (fy1 - 50 < cy < fy2 + 50):
                            is_threat = False
                            break
                        
                        # High-overlap check
                        ix1, iy1 = max(x1, fx1), max(y1, fy1)
                        ix2, iy2 = min(x2, fx2), min(y2, fy2)
                        if ix1 < ix2 and iy1 < iy2:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            weapon_area = (x2 - x1) * (y2 - y1)
                            if inter_area / weapon_area > 0.30: # Much lower threshold
                                is_threat = False
                                break

            if is_threat:
                weapon_detected = True
                label = label_override if label_override else f"{class_name} {conf:.0%}"
                weapon_labels.append(label)
                all_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "type": "weapon",
                    "confidence": conf
                })


            # ── 3. Standard Nearby Check ─────────────────────────────────────
            elif cls_id not in SKIP_NEARBY and conf >= NEARBY_CONF:
                if area_pct >= NEARBY_AREA_RATIO:
                    if class_name not in nearby_objects:
                        nearby_objects.append(class_name)
                    all_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "label": f"{class_name} {conf:.0%}",
                        "type": "nearby",
                        "confidence": conf
                    })

        return weapon_detected, weapon_labels, nearby_objects, all_detections

    # ── Drawing ───────────────────────────────────────────────────────────────
    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            det_type = det.get("type", "nearby")

            if det_type == "weapon":
                color = (0, 0, 255)    # Red
            else:
                color = (0, 215, 255)  # Amber/yellow

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return frame
