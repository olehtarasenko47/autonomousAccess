# Live Phone Camera Shirt Color Recognition

Captures live video from a phone camera and **only identifies calibrated shirt colors** using YOLO-pose.

- **No person identification, re-ID, tracking, or facial recognition.**
- **No uncalibrated colors:** only recognizes colors calibrated by the user.
- **In-place calibration:** left-click on any shirt in the live camera window to calibrate and save to `colour.json`.

---

## Architecture (Exactly 4 Sensible Files)

1. **[`phone_stream.py`](file:///home/ot/palantirTypeShit/autonomousAccess/phone_stream.py)** — Phone camera streaming (DroidCam, IP Webcam, RTSP, USB tethering) with low-latency background capture.
2. **[`calibration.py`](file:///home/ot/palantirTypeShit/autonomousAccess/calibration.py)** — Click-sampling calibration on shirt fabric, HSV percentile range computation with wrap-around, and loading/saving `colour.json`.
3. **[`shirt_detector.py`](file:///home/ot/palantirTypeShit/autonomousAccess/shirt_detector.py)** — Lightweight YOLO-pose person detection, upper-body keypoints (shoulders & hips) torso extraction, and calibrated color matching.
4. **[`main.py`](file:///home/ot/palantirTypeShit/autonomousAccess/main.py)** — Final orchestration: live loop, HUD, in-place click-sampling, and controls.

---

## Quick Usage

```bash
cd /home/ot/palantirTypeShit/autonomousAccess

# DroidCam (default port 4747):
.venv/bin/python main.py --phone-ip 192.168.1.15 --app droidcam

# IP Webcam (default port 8080):
.venv/bin/python main.py --phone-ip 192.168.1.15 --app ipwebcam

# Direct stream URL, USB webcam index, or test image:
.venv/bin/python main.py --source http://192.168.1.15:4747/video
.venv/bin/python main.py --source 0
```

### In-Place Calibration

1. Run `main.py`.
2. **Left-click on the shirt** in the video window (click 2–4 times across lighting/shadows).
3. The program automatically calculates the HSV bounds and updates `colour.json` immediately.
4. The shirt is now recognized!

### Controls

| Key / Action | Description |
|:---|:---|
| **Left Click** | Sample clicked pixel patch on shirt & update calibration |
| **`s`** | Save current calibration to `colour.json` |
| **`r`** | Reset sampled pixels |
| **`q` / `Esc`** | Quit |
