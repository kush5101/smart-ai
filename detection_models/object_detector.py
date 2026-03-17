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
WEAPON_CONF   = 0.40   # Reduced false positives for knives/scissors
LIGHTER_CONF  = 0.30   # Lowered to 0.30 to catch proxy misclassifications before the weapon heuristic
NEARBY_CONF   = 0.45
NEARBY_AREA_RATIO = 0.08


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
    def detect(self, frame):
        """
        Run YOLOv8 on frame.
        Returns:
            weapon_detected (bool)
            weapon_labels   (list[str])   e.g. ["knife 87%"]
            nearby_objects  (list[str])   e.g. ["bottle", "cell phone"]
            all_detections  (list[dict])  raw boxes for drawing
        """
        h, w = frame.shape[:2]
        frame_area = h * w

        # Use lower imgsz for better CPU framerate to reduce lag
        results = self.model(frame, verbose=False, imgsz=320)[0]

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

            # ── 1. Weapon/Threat Check ───────────────────────────────────────
            is_threat = False
            label_override = None

            # Actual weapon classes
            if cls_id in WEAPON_CLASSES and conf >= WEAPON_CONF:
                is_threat = True

            # ── 2. Lighter Heuristic (Fixing bottle/lighter confusion) ───────
            # Lighters are small, rectangular, and often mislabeled as bottles (39) or phones (67)
            # Area widened to catch lighters held close to the camera (up to 20% of screen)
            elif (cls_id == 39 or cls_id == 67) and conf >= LIGHTER_CONF and area_pct < 0.20:
                # Support vertical (0.2-0.5) and horizontal (2.0-5.0) lighters
                if (0.15 < aspect_ratio < 0.6) or (1.8 < aspect_ratio < 6.0):
                    label = f"LIGHTER {conf:.0%}"
                    if "LIGHTER" not in nearby_objects:
                        nearby_objects.append("LIGHTER")
                    all_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "label": label,
                        "type": "nearby",
                        "confidence": conf
                    })
                    continue # Skip further checks like gun heuristic

            # ── 3. Gun Heuristic: Large hand-held objects labeled as phone/remote ─
            elif cls_id in GUN_PROXIES and conf >= 0.35:
                # Pistols often have a specific horizontal aspect ratio when held
                # Supporting both horizontal (1.2-3.0) and gripped orientations (0.5-1.0)
                # Area loosened to catch images shown on phones
                if (0.3 < aspect_ratio < 3.0) and 0.015 < area_pct < 0.60:
                    is_threat = True
                    label_override = f"POSSIBLE FIREARM ({class_name}) {conf:.0%}"

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
