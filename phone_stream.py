#!/usr/bin/env python3
"""
Phone Camera Streaming.

Handles live video from a mobile phone (DroidCam / IP Webcam / RTSP),
USB tethered webcam, or video/image files. Runs a lightweight background
capture thread so frame queues never build up.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Optional

import cv2
import numpy as np


class PhoneStream:
    """Non-blocking phone camera capture with automatic reconnection."""

    def __init__(self, source: str) -> None:
        self.url = self._format_source(source)
        self.frame: Optional[np.ndarray] = None
        self.running = True
        self.connected = False
        self._lock = threading.Lock()

        # Static image support for quick offline testing
        p = Path(self.url)
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            self.static_image = cv2.imread(str(p))
            self.connected = self.static_image is not None
            self.is_static = True
            return

        self.is_static = False
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def _format_source(self, src: str) -> str:
        s = str(src).strip()
        # Direct IP shorthand e.g. "192.168.1.15:4747" -> "http://192.168.1.15:4747/video"
        if s.count(".") == 3 and ":" in s and not s.startswith("http"):
            return f"http://{s}/video"
        return s

    def _capture_worker(self) -> None:
        target = int(self.url) if self.url.isdigit() else self.url
        while self.running:
            cap = cv2.VideoCapture(target)
            if not cap.isOpened():
                self.connected = False
                time.sleep(1.0)
                continue

            self.connected = True
            while self.running:
                ok, img = cap.read()
                if not ok or img is None:
                    break
                with self._lock:
                    self.frame = img

            cap.release()
            self.connected = False
            time.sleep(1.0)

    def read(self) -> Optional[np.ndarray]:
        if self.is_static:
            return self.static_image.copy() if self.static_image is not None else None
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def close(self) -> None:
        self.running = False
