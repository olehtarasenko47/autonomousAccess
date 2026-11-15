#!/usr/bin/env python3
"""
Live Phone Camera Shirt Color Recognition — Main Orchestration.

Orchestrates:
- Phone camera streaming (phone_stream.py)
- Shirt color click-calibration (calibration.py)
- YOLO-pose torso extraction & color detection (shirt_detector.py)
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np

from calibration import ClickCalibrator, load_calibration, save_calibration
from phone_stream import PhoneStream
from shirt_detector import ShirtDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Phone Camera Shirt Color Recognition")
    parser.add_argument("--source", default=None, help="Phone URL (http://IP:4747/video), device index (0), or file")
    parser.add_argument("--phone-ip", default=None, help="Phone IP address (paired with --app)")
    parser.add_argument("--app", default="droidcam", choices=["droidcam", "ipwebcam"], help="Phone streaming app")
    parser.add_argument("--colour-file", default="colour.json", help="Path to calibrated colors JSON")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO-pose model weights")
    parser.add_argument("--threshold", type=float, default=0.35, help="Match ratio threshold")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Exit after N frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Resolve phone source
    if args.phone_ip:
        port = 4747 if args.app == "droidcam" else 8080
        source = f"http://{args.phone_ip}:{port}/video"
    elif args.source:
        source = args.source
    else:
        source = "http://127.0.0.1:4747/video"

    print(f"[Main] Connecting to phone stream: {source}")
    cam = PhoneStream(source)

    # 2. Load calibration (strictly no uncalibrated presets)
    calibrated_ranges = load_calibration(args.colour_file)
    if calibrated_ranges:
        print(f"[Main] Loaded calibrated ranges from {args.colour_file}")
    else:
        print(f"[Main] No calibration found in {args.colour_file}. Click on shirt in camera to calibrate.")

    # 3. Initialize YOLO-pose shirt detector
    print(f"[Main] Initializing YOLO-pose detector: {args.model}")
    detector = ShirtDetector(model_path=args.model, threshold=args.threshold)

    # 4. Initialize in-place click calibrator
    calibrator = ClickCalibrator()
    current_frame: Optional[np.ndarray] = None

    window_name = "Shirt Color Recognition (Click: Calibrate | s: Save | r: Reset | q: Quit)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_click(event: int, x: int, y: int, *_) -> None:
        nonlocal calibrated_ranges, current_frame
        if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
            n_px = calibrator.add_click(current_frame, x, y)
            new_ranges = calibrator.compute_ranges()
            if new_ranges:
                calibrated_ranges = new_ranges
                save_calibration(calibrated_ranges, args.colour_file)
                print(f"[Calibrate] Sampled {n_px} px (Total: {calibrator.total_samples}). Updated {args.colour_file}!")

    if not args.headless:
        cv2.setMouseCallback(window_name, on_click)

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                if not args.headless:
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(canvas, f"Waiting for Phone Camera: {source}", (40, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
                    cv2.imshow(window_name, canvas)
                    if (cv2.waitKey(50) & 0xFF) in (ord("q"), 27):
                        break
                continue

            current_frame = frame
            frame_count += 1
            annotated = frame.copy()

            # Run detection
            detections = detector.detect(frame, calibrated_ranges)
            matched_count = 0

            for det in detections:
                poly = det["poly"]
                if det["is_match"]:
                    matched_count += 1
                    cv2.polylines(annotated, [poly], True, (0, 255, 0), 3)
                    lbl = f"SHIRT MATCH ({int(det['ratio'] * 100)}%)"
                    cv2.putText(annotated, lbl, (poly[0][0], max(20, poly[0][1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.polylines(annotated, [poly], True, (120, 120, 120), 1)

            # HUD
            fps = frame_count / max(1e-4, time.time() - t0)
            cal_status = "CALIBRATED" if calibrated_ranges else "UNSET (Click on shirt)"
            hud = f"FPS: {fps:.1f} | People: {len(detections)} | Shirts: {matched_count} | Color: {cal_status}"
            cv2.rectangle(annotated, (6, 6), (620, 36), (20, 20, 20), -1)
            cv2.putText(annotated, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 255, 0) if matched_count > 0 else (0, 220, 255), 1)

            if args.headless:
                if frame_count % 30 == 0 or matched_count > 0:
                    print(f"[Frame {frame_count:04d}] People: {len(detections)}, Shirts: {matched_count} | {fps:.1f} FPS")
            else:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    calibrator.reset()
                    print("[Calibration] Reset samples.")
                elif key == ord("s") and calibrated_ranges:
                    save_calibration(calibrated_ranges, args.colour_file)

            if args.max_frames and frame_count >= args.max_frames:
                break

    finally:
        cam.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
