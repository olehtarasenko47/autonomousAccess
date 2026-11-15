#!/usr/bin/env python3
"""
YOLO-Pose Shirt Detector.

Uses lightweight YOLO-pose to detect people, extracts the upper-body
shirt frame polygon (shoulders & hips keypoints), and checks if the shirt
color matches the calibrated HSV ranges.
No person identification, re-id, or tracking.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12


class ShirtDetector:
    """Detects people and verifies shirt color on the extracted torso frame."""

    def __init__(self, model_path: str = "yolo11n-pose.pt", conf: float = 0.25, threshold: float = 0.35) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.threshold = threshold

    def extract_torso_polygon(self, kpts: np.ndarray, box: Sequence[int], kp_conf: float = 0.5) -> np.ndarray:
        """Constructs 2D torso polygon using upper-body keypoints."""
        x1, y1, x2, y2 = box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)

        ls, rs = kpts[KP_LEFT_SHOULDER], kpts[KP_RIGHT_SHOULDER]
        lh, rh = kpts[KP_LEFT_HIP], kpts[KP_RIGHT_HIP]

        # Full Torso visible
        if ls[2] >= kp_conf and rs[2] >= kp_conf and lh[2] >= kp_conf and rh[2] >= kp_conf:
            quad = np.array([[ls[0], ls[1]], [rs[0], rs[1]], [rh[0], rh[1]], [lh[0], lh[1]]], dtype=np.float32)
            centroid = np.mean(quad, axis=0)
            # Inward shrink by 15% to avoid arms/background edges
            return np.round(quad + 0.15 * (centroid - quad)).astype(np.int32)

        # Shoulders visible, hips occluded (staircases, desks)
        if ls[2] >= kp_conf and rs[2] >= kp_conf:
            drop = 0.80 * max(np.hypot(rs[0] - ls[0], rs[1] - ls[1]), bh * 0.25)
            return np.array([
                [ls[0], ls[1]],
                [rs[0], rs[1]],
                [rs[0], rs[1] + drop],
                [ls[0], ls[1] + drop],
            ], dtype=np.int32)

        # Fallback to inner chest core
        return np.array([
            [x1 + 0.35 * bw, y1 + 0.28 * bh],
            [x1 + 0.65 * bw, y1 + 0.28 * bh],
            [x1 + 0.65 * bw, y1 + 0.52 * bh],
            [x1 + 0.35 * bw, y1 + 0.52 * bh],
        ], dtype=np.int32)

    def evaluate_color(
        self,
        frame: np.ndarray,
        poly: np.ndarray,
        ranges: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[bool, float]:
        """Checks if the torso polygon pixels match the calibrated HSV ranges."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Filter valid chromatic pixels (drop deep shadows and specular glare)
        valid = (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 35) & (mask > 0)
        n_valid = int(np.count_nonzero(valid))
        if n_valid < 10:
            return False, 0.0

        match_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            match_mask |= cv2.inRange(hsv, lo, hi)

        matched_pixels = int(np.count_nonzero((match_mask > 0) & valid))
        torso_pixels = max(1, cv2.countNonZero(mask))
        denom = max(n_valid, int(0.25 * torso_pixels), 1)

        ratio = matched_pixels / denom
        return ratio >= self.threshold, ratio

    def detect(
        self,
        frame: np.ndarray,
        calibrated_ranges: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> List[dict]:
        """Runs YOLO-pose, extracts shirt frames, and evaluates shirt colors."""
        results = self.model(frame, imgsz=640, conf=self.conf, verbose=False)
        if not results or len(results) == 0 or results[0].boxes is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        kpts = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else []

        detections = []
        for i, box in enumerate(boxes):
            poly = self.extract_torso_polygon(kpts[i] if i < len(kpts) else None, box)
            is_match, ratio = False, 0.0
            if calibrated_ranges:
                is_match, ratio = self.evaluate_color(frame, poly, calibrated_ranges)

            detections.append({
                "box": box,
                "poly": poly,
                "is_match": is_match,
                "ratio": ratio,
            })

        return detections
