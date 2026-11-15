#!/usr/bin/env python3
"""
Shirt Color Click Calibration.

Handles click-sampling on shirt fabric, HSV percentile range calculation
with hue wrap-around support, and loading/saving to colour.json.
Can be imported as a module or executed standalone to calibrate from camera:
  python3 calibration.py --phone-ip 192.168.1.15 --app droidcam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class ClickCalibrator:
    """Manages click-sampling and HSV calibration on live frames."""

    def __init__(self, patch_radius: int = 4, margin_h: int = 6, margin_sv: int = 30) -> None:
        self.patch_radius = patch_radius
        self.margin_h = margin_h
        self.margin_sv = margin_sv
        self.samples: List[List[int]] = []

    def add_click(self, frame: np.ndarray, x: int, y: int) -> int:
        """Samples patch around (x, y) and stores HSV pixels."""
        h, w = frame.shape[:2]
        r = self.patch_radius
        patch = frame[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        self.samples.extend(hsv_patch.tolist())
        return len(hsv_patch)

    def compute_ranges(self) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        """Calculates 3rd-97th percentile HSV bounds with hue wrap-around."""
        px = np.asarray(self.samples, dtype=np.int32)
        if len(px) == 0:
            return None

        # Filter extreme shadows / unsaturated pixels
        valid = px[(px[:, 1] >= 50) & (px[:, 2] >= 40)]
        if len(valid) < 10:
            return None

        h, s, v = valid[:, 0], valid[:, 1], valid[:, 2]
        spread = np.percentile(h, 97) - np.percentile(h, 3)
        rotated = (h + 90) % 180
        spread_rot = np.percentile(rotated, 97) - np.percentile(rotated, 3)

        s_lo = max(50, int(np.percentile(s, 3)) - self.margin_sv)
        v_lo = max(40, int(np.percentile(v, 3)) - self.margin_sv)

        if spread_rot < spread:
            # Hue wraps around 0/180 (e.g. red)
            lo = (int(np.percentile(rotated, 3)) - self.margin_h - 90) % 180
            hi = (int(np.percentile(rotated, 97)) + self.margin_h - 90) % 180
            return [
                (np.array([lo, s_lo, v_lo], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
                (np.array([0, s_lo, v_lo], dtype=np.uint8), np.array([hi, 255, 255], dtype=np.uint8)),
            ]
        else:
            h_lo = max(0, int(np.percentile(h, 3)) - self.margin_h)
            h_hi = min(179, int(np.percentile(h, 97)) + self.margin_h)
            return [(np.array([h_lo, s_lo, v_lo], dtype=np.uint8), np.array([h_hi, 255, 255], dtype=np.uint8))]

    def reset(self) -> None:
        self.samples.clear()

    @property
    def total_samples(self) -> int:
        return len(self.samples)


def load_calibration(path: str = "colour.json") -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """Loads calibrated HSV ranges from JSON file. Returns None if not found."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ranges = []
        for item in data.get("hsv_ranges", []):
            if len(item) == 2:
                ranges.append((np.array(item[0], dtype=np.uint8), np.array(item[1], dtype=np.uint8)))
        return ranges if ranges else None
    except Exception as e:
        print(f"[Calibration] Could not load {path}: {e}")
        return None


def save_calibration(ranges: List[Tuple[np.ndarray, np.ndarray]], path: str = "colour.json", name: str = "calibrated_shirt") -> None:
    data = {
        "name": name,
        "hsv_ranges": [[lo.tolist(), hi.tolist()] for lo, hi in ranges],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[Calibration] Saved to {path}")


# ------------------------------------------------------------- Standalone CLI ---

def main() -> None:
    from phone_stream import PhoneStream

    parser = argparse.ArgumentParser(description="Standalone Shirt Color Click Calibrator")
    parser.add_argument("--source", default=None, help="Phone stream URL, device index (0), or image file")
    parser.add_argument("--phone-ip", default=None, help="Phone IP address (paired with --app)")
    parser.add_argument("--app", default="droidcam", choices=["droidcam", "ipwebcam"], help="Phone streaming app")
    parser.add_argument("--out", default="colour.json", help="Path to save calibrated ranges")
    parser.add_argument("--name", default="calibrated_shirt", help="Name label for calibrated shirt")
    args = parser.parse_args()

    if args.phone_ip:
        port = 4747 if args.app == "droidcam" else 8080
        source = f"http://{args.phone_ip}:{port}/video"
    elif args.source:
        source = args.source
    else:
        source = "http://127.0.0.1:4747/video"

    print(f"[Calibration] Connecting to: {source}")
    cam = PhoneStream(source)
    calibrator = ClickCalibrator()
    show_mask = False
    window_name = "Calibrate Shirt (Click: Sample | m: Mask | s: Save | r: Reset | q: Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_frame: Optional[np.ndarray] = None
    computed_ranges: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    def on_click(event: int, x: int, y: int, *_) -> None:
        nonlocal current_frame, computed_ranges
        if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
            n_px = calibrator.add_click(current_frame, x, y)
            computed_ranges = calibrator.compute_ranges()
            print(f"[Calibration] Click at ({x}, {y}): +{n_px} px (Total: {calibrator.total_samples})")

    cv2.setMouseCallback(window_name, on_click)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(canvas, f"Waiting for stream: {source}", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
                cv2.imshow(window_name, canvas)
                if (cv2.waitKey(50) & 0xFF) in (ord("q"), 27):
                    break
                continue

            current_frame = frame
            view = frame.copy()

            if computed_ranges:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                for lo, hi in computed_ranges:
                    mask |= cv2.inRange(hsv, lo, hi)

                if show_mask:
                    view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                else:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        if cv2.contourArea(c) > 300:
                            cv2.drawContours(view, [c], -1, (0, 255, 0), 2)

            hud = f"Samples: {calibrator.total_samples} | Ranges: {len(computed_ranges) if computed_ranges else 0}"
            cv2.rectangle(view, (6, 6), (450, 36), (20, 20, 20), -1)
            cv2.putText(view, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if computed_ranges else (0, 220, 255), 1)

            cv2.imshow(window_name, view)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("r"):
                calibrator.reset()
                computed_ranges = None
                print("[Calibration] Reset samples.")
            elif key == ord("s"):
                if computed_ranges:
                    save_calibration(computed_ranges, path=args.out, name=args.name)
                else:
                    print("[Calibration] Click on the shirt first to sample colors.")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
