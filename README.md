# Autonomous Access — Shirt Colour Recognition

Live shirt colour recognition using calibrated HSV ranges and YOLO-pose torso isolation.

## Files

| File | Purpose |
|------|---------|
| `calibrate_colour.py` | Click on the shirt to sample HSV, saves `colour.json` |
| `main.py` | Live recognition loop — detects people, classifies shirt colour |
| `torso.py` | 4-tier adaptive torso isolation from YOLO-pose keypoints |
| `video_source.py` | Camera streaming (phone URL, webcam, Pi camera, video file) |
| `colour.json` | Calibrated HSV ranges (created by calibration) |

## Setup

```bash
# Uses shared venv and YOLO weights (symlinked)
pip install ultralytics opencv-python numpy
```

## Usage

### 1. Calibrate

```bash
# Phone camera
python calibrate_colour.py --source http://PHONE_IP:8080/video

# Webcam
python calibrate_colour.py --source 0

# From photo
python calibrate_colour.py --image shirt.jpg
```

Click on the shirt a few times, press `m` to check mask, press `s` to save.

### 2. Run

```bash
# Phone camera
python main.py --source http://PHONE_IP:8080/video

# Webcam
python main.py --source 0
```

Keys: `q` to quit.
