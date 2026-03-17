import cv2
import numpy as np
from collections import deque

class FireDetector:
    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold

        # Temporal flicker history: store per-region pixel-change scores
        # We keep last N frames of masks to detect movement/flicker
        self._history_len = 6
        self._mask_history = deque(maxlen=self._history_len)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_fire_mask(self, frame):
        """
        Returns a binary mask of pixels that COULD be fire colour.
        Uses two sub-ranges:
          A) Orange-Yellow flame core  (Hue 10-35, HIGH saturation, HIGH value)
          B) Deep red embers           (Hue 0-10 AND 160-180, HIGH sat, HIGH value)
        Solid-coloured objects (cane, bottle) typically have:
          - High saturation but NO incandescent core nearby
          - Uniform appearance across frames (no flicker)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -- Sub-range A: orange/yellow --
        lower_a = np.array([10, 110, 180])   # sat>=110 & val>=180 → more inclusive orange
        upper_a = np.array([35, 255, 255])
        mask_a  = cv2.inRange(hsv, lower_a, upper_a)

        # -- Sub-range B: deep red embers --
        lower_b1 = np.array([0,  110, 180])
        upper_b1 = np.array([10, 255, 255])
        lower_b2 = np.array([160, 110, 180])
        upper_b2 = np.array([180, 255, 255])
        mask_b   = cv2.bitwise_or(
            cv2.inRange(hsv, lower_b1, upper_b1),
            cv2.inRange(hsv, lower_b2, upper_b2)
        )

        color_mask = cv2.bitwise_or(mask_a, mask_b)

        # -- Incandescent core: extremely bright pixels (>225 luma) --
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, luma_mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

        # Dilate the bright core only slightly so only nearby coloured pixels qualify
        kernel_small = np.ones((9, 9), np.uint8)
        luma_dilated = cv2.dilate(luma_mask, kernel_small, iterations=2)

        # Final mask = fire-colour AND near a bright core
        final_mask = cv2.bitwise_and(color_mask, luma_dilated)

        # Small morphological cleanup to remove noise
        kernel_open = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN,  kernel_open)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_open)

        return final_mask

    def _solidity(self, contour):
        """
        Solidity = contour area / convex hull area.
        Real fire: low solidity (irregular, flickery edges) → < 0.85
        Solid objects (cane, bottle): high solidity → > 0.90
        """
        area = cv2.contourArea(contour)
        if area == 0:
            return 1.0
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area == 0:
            return 1.0
        return area / hull_area

    def _aspect_ratio_ok(self, w, h):
        """
        A cane / stick will have an extreme aspect ratio (very tall & thin).
        Reject detections that are more than 4× taller than wide,
        or more than 4× wider than tall.
        """
        if h == 0 or w == 0:
            return False
        ratio = h / w
        return 0.25 <= ratio <= 4.0

    def _flicker_score(self, current_mask):
        """
        Returns a score 0-1 representing how much this mask region changes
        over recent frames.  Fire flickers (high score); solid objects don't (low).
        """
        self._mask_history.append(current_mask.copy())
        if len(self._mask_history) < 3:
            # Not enough history → be lenient (don't reject on first frames)
            return 1.0

        # XOR consecutive pairs of masks and average non-zero fraction
        changes = []
        hist = list(self._mask_history)
        total_pixels = current_mask.size
        for i in range(1, len(hist)):
            diff = cv2.bitwise_xor(hist[i], hist[i - 1])
            changes.append(np.count_nonzero(diff) / total_pixels)

        return float(np.mean(changes))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _detect_opencv_heuristic(self, frame):
        final_mask = self._build_fire_mask(frame)
        flicker    = self._flicker_score(final_mask)

        contours, _ = cv2.findContours(
            final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections   = []
        fire_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # ── Gate 1: minimum area (lower threshold for matches/lighters) ──
            if area < 20: # Lowered further to 20 to catch very small lighter flames
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # ── Gate 2: aspect ratio — relax for smaller fire ──
            if not self._aspect_ratio_ok(w, h):
                # Allow slightly more extreme ratios for small flames
                ratio = h / w
                if not (0.15 <= ratio <= 6.0):
                    continue

            # ── Gate 3: solidity — allow slightly more solid for small cores ──
            sol = self._solidity(cnt)
            if sol > 0.92:
                continue

            # ── Gate 4: temporal flicker — relax for small, fast flames ──
            # Removed flicker check for small flames to make it incredibly sensitive
            if area > 200 and flicker < 0.0005: 
                continue

            # ── Confidence: blend of area and flicker energy ──
            area_score    = min(1.0, area / 20000.0)
            flicker_score = min(1.0, flicker / 0.05)
            mock_conf     = 0.55 + 0.25 * area_score + 0.20 * flicker_score
            mock_conf     = min(0.98, mock_conf)

            if mock_conf > self.confidence_threshold:
                detections.append({
                    "bbox":       [x, y, x + w, y + h],
                    "label":      f"FIRE {mock_conf:.2%}",
                    "confidence": mock_conf,
                })
                fire_detected = True

        return fire_detected, detections

    def detect(self, frame):
        return self._detect_opencv_heuristic(frame)

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + tw, y1), (0, 0, 255), -1)
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        return frame
