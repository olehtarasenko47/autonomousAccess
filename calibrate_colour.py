#!/usr/bin/env python3
"""
Ambassador T-shirt colour calibration.

Click on the ambassador's T-shirt a few times (different lighting / angles),
check the mask looks right, then press 's' to save the HSV range to colour.json.
The detector (Phase 2) loads colour.json, so every node uses the same colour.

Usage
  Pi camera (with a screen/VNC):   python3 calibrate_colour.py
  Laptop webcam:                   python3 calibrate_colour.py --source 0
  Phone stream:                    python3 calibrate_colour.py --source http://PHONE_IP:8080/video
  Phone recording:                 python3 calibrate_colour.py --source walk.mp4
  From photos (no screen on Pi):   python3 calibrate_colour.py --image shirt1.jpg shirt2.jpg
  Take a photo on a headless Pi:   python3 calibrate_colour.py --snapshot shirt1.jpg

Keys:  click = sample   m = toggle mask view   u = undo last sample
       r = reset        n = next image          s = save           q = quit
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PATCH = 9          # size of the square sampled around each click (pixels)
MARGIN_H = 6       # extra hue tolerance added on each side
MARGIN_SV = 30     # extra saturation/value tolerance
MIN_S = 60         # never accept very grey pixels as "shirt colour"
MIN_V = 40         # never accept very dark pixels


# ---------------------------------------------------------------- camera ---
class Camera:
    """Any source video_source.py understands: pi, webcam index, phone URL, video file."""

    def __init__(self, source):
        from video_source import open_source
        try:
            self.src = open_source(source)
        except Exception as e:
            if source != "pi":
                sys.exit(f"Could not open {source}: {e}")
            print(f"Picamera2 unavailable ({e}); falling back to webcam 0")
            self.src = open_source("0")
        self.last = None

    def read(self):
        frame, _ = self.src.read()
        if frame is None:          # end of a video file: keep showing last frame
            return self.last
        self.last = frame
        return frame

    def close(self):
        self.src.close()


# ------------------------------------------------------------ the maths ---
def sample_patch(bgr, x, y, size=PATCH):
    """Return the HSV pixels in a size x size square centred on (x, y)."""
    h, w = bgr.shape[:2]
    r = size // 2
    patch = bgr[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)


def compute_ranges(hsv_samples):
    """
    Turn sampled HSV pixels into one or two OpenCV inRange() bounds.
    OpenCV hue runs 0-179 and wraps (red sits at both ends), so if the
    hues straddle 0/179 we return two ranges.
    """
    px = np.asarray(hsv_samples, dtype=np.int32)
    px = px[(px[:, 1] >= MIN_S) & (px[:, 2] >= MIN_V)]  # drop grey/dark pixels
    if len(px) < 10:
        raise ValueError("Not enough coloured pixels sampled - click on the shirt, "
                         "not on shadows or white logos.")

    hue, sat, val = px[:, 0], px[:, 1], px[:, 2]

    # Detect wrap-around: rotate hue by 90 and see if the spread gets smaller.
    spread = np.percentile(hue, 97) - np.percentile(hue, 3)
    rotated = (hue + 90) % 180
    spread_rot = np.percentile(rotated, 97) - np.percentile(rotated, 3)
    wraps = spread_rot < spread

    if wraps:
        lo = int(np.percentile(rotated, 3)) - MARGIN_H
        hi = int(np.percentile(rotated, 97)) + MARGIN_H
        h_lo, h_hi = (lo - 90) % 180, (hi - 90) % 180
    else:
        h_lo = max(0, int(np.percentile(hue, 3)) - MARGIN_H)
        h_hi = min(179, int(np.percentile(hue, 97)) + MARGIN_H)

    s_lo = max(MIN_S, int(np.percentile(sat, 3)) - MARGIN_SV)
    s_hi = 255
    v_lo = max(MIN_V, int(np.percentile(val, 3)) - MARGIN_SV)
    v_hi = 255

    if h_lo <= h_hi:
        return [[[h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi]]]
    # wrapped: e.g. 172..179 and 0..8
    return [[[h_lo, s_lo, v_lo], [179, s_hi, v_hi]],
            [[0, s_lo, v_lo], [h_hi, s_hi, v_hi]]]


def make_mask(bgr, ranges):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    # clean speckle noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def save(ranges, n_pixels, path):
    data = {
        "name": "ambassador_shirt",
        "hsv_ranges": ranges,
        "samples": int(n_pixels),
        "min_blob_area": 1500,  # tune in Phase 2 depending on camera distance
    }
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"Saved {path}: {ranges}")


# ------------------------------------------------------------------- UI ---
def run(args):
    images = [cv2.imread(p) for p in args.image] if args.image else None
    if images and any(im is None for im in images):
        sys.exit("Could not read one of the --image files")
    cam = None if images else Camera(args.source)

    samples, history = [], []   # history = sample counts per click, for undo
    state = {"frame": None, "show_mask": False, "img_idx": 0}
    win = "calibrate (click shirt | m mask | u undo | r reset | n next | s save | q quit)"

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and state["frame"] is not None:
            px = sample_patch(state["frame"], x, y)
            samples.extend(px.tolist())
            history.append(len(px))
            print(f"sample {len(history)} at ({x},{y}) mean HSV {px.mean(0).astype(int)}")

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    while True:
        frame = images[state["img_idx"]].copy() if images else cam.read()
        if frame is None:
            continue
        state["frame"] = frame

        ranges = None
        if samples:
            try:
                ranges = compute_ranges(samples)
            except ValueError:
                ranges = None

        view = frame.copy()
        if ranges:
            mask = make_mask(frame, ranges)
            if state["show_mask"]:
                view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) > 1500:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(view, f"samples: {len(history)}  range: {ranges}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow(win, view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            state["show_mask"] = not state["show_mask"]
        elif key == ord("u") and history:
            del samples[-history.pop():]
        elif key == ord("r"):
            samples.clear(); history.clear()
        elif key == ord("n") and images:
            state["img_idx"] = (state["img_idx"] + 1) % len(images)
        elif key == ord("s"):
            if ranges:
                save(ranges, len(samples), args.out)
            else:
                print("Nothing to save yet - click on the shirt first.")

    if cam:
        cam.close()
    cv2.destroyAllWindows()


def snapshot(path, source):
    cam = Camera(source)
    for _ in range(10):      # let auto-exposure settle
        frame = cam.read()
    cam.close()
    cv2.imwrite(path, frame)
    print(f"Saved photo {path} - copy it to a laptop and run with --image {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="pi",
                    help="'pi' (default), webcam index like 0, phone URL, or video file")
    ap.add_argument("--image", nargs="+", help="calibrate from photo(s) instead of a camera")
    ap.add_argument("--snapshot", metavar="FILE", help="just take one photo and exit")
    ap.add_argument("--out", default="colour.json")
    a = ap.parse_args()
    if a.snapshot:
        snapshot(a.snapshot, a.source)
    else:
        run(a)
