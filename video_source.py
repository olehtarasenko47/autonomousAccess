#!/usr/bin/env python3
"""Camera sources and responsive live preview. Frames are OpenCV BGR arrays.

read() returns (frame, Unix timestamp), or (None, None) on a live timeout.
For files, (None, None) means EOF. Live timestamps are local receive times.
Each source is intended for one consumer; StreamSource keeps the newest frame.
"""
import argparse
import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from urllib.parse import urlsplit

import cv2
import numpy as np


def safe_url(url):
    """Never print URL credentials, query tokens or fragments ourselves."""
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc.rsplit('@', 1)[-1]}{p.path}"


class PiSource:
    def __init__(self, size=(640, 480)):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_preview_configuration(
            main={"format": "RGB888", "size": size}))
        self.cam.start()

    def read(self):
        return self.cam.capture_array(), time.time()

    def close(self):
        self.cam.stop()


class WebcamSource:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"cannot open webcam {index}")

    def read(self):
        ok, frame = self.cap.read()
        return (frame, time.time()) if ok else (None, None)

    def close(self):
        self.cap.release()


class StreamSource:
    """Background decoder; one latest-frame slot; reconnect on failure."""
    def __init__(self, url, reconnect_delay=2.0, open_timeout_ms=4000,
                 read_timeout_ms=3000):
        self.url = url
        self.reconnect_delay = max(0.2, reconnect_delay)
        self._params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_ms),
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout_ms)]
        self.frame, self.ts, self.seq = None, None, 0
        self._last_seq = 0
        self.last_generation = 0
        self._connected = False
        self._generation = 0
        self._arrivals = deque(maxlen=120)
        self._new = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="camera-reader")
        self._thread.start()

    @property
    def connected(self):
        with self._new:
            return self._connected

    @property
    def generation(self):
        """Increments on the first valid frame of each connection."""
        with self._new:
            return self._generation

    @property
    def capture_fps(self):
        with self._new:
            times = self._arrivals
            if (len(times) < 2 or time.monotonic() - times[-1] > 1.0):
                return 0.0
            return (len(times) - 1) / max(1e-6, times[-1] - times[0])

    def _offline(self):
        with self._new:
            self._connected = False
            self.frame, self.ts = None, None
            self._arrivals.clear()
            self._new.notify_all()

    def _run(self):
        label = safe_url(self.url)
        while not self._stop.is_set():
            cap = None
            try:
                # FFmpeg supports these open-only timeout parameters.
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, self._params)
                if not cap.isOpened():
                    print(f"[video] cannot reach {label}; retrying", flush=True)
                else:
                    first = True
                    while not self._stop.is_set():
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            print(f"[video] stream lost: {label}", flush=True)
                            break
                        now, ts = time.monotonic(), time.time()
                        with self._new:
                            if self._stop.is_set():
                                break
                            if first:
                                self._generation += 1
                                self._connected = True
                            self.frame, self.ts = frame, ts
                            self.seq += 1
                            self._arrivals.append(now)
                            while (len(self._arrivals) > 2 and
                                   now - self._arrivals[0] > 3.0):
                                self._arrivals.popleft()
                            self._new.notify_all()
                        if first:
                            print(f"[video] connected to {label}", flush=True)
                            first = False
            except (cv2.error, OSError) as exc:
                # Don't log exception text: it may contain the full URL.
                print(f"[video] {type(exc).__name__} at {label}; retrying",
                      flush=True)
            finally:
                self._offline()
                if cap is not None:
                    cap.release()
            # Unlike sleep(), this wakes immediately when close() is called.
            self._stop.wait(self.reconnect_delay)

    def read(self, timeout=0.1):
        """Return a new frame once; never return a disconnected stale frame."""
        with self._new:
            self._new.wait_for(
                lambda: self._stop.is_set() or
                (self._connected and self.seq > self._last_seq), timeout)
            if (self._stop.is_set() or not self._connected or
                    self.seq <= self._last_seq):
                return None, None
            self._last_seq = self.seq
            self.last_generation = self._generation
            return self.frame, self.ts

    def close(self):
        self._stop.set()
        self._offline()
        # Only the worker owns/releases VideoCapture (no cross-thread release).
        # Bound shutdown even if the native backend ignores its timeout.
        self._thread.join(timeout=4.5)


class FileSource:
    """Replay timestamps = recording start + video position."""
    def __init__(self, path, start_time=None, realtime=False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"cannot open {path}; try converting to H.264 MP4")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if start_time is None:
            duration = n / self.fps if math.isfinite(n) and n > 0 else 0
            start_time = os.path.getmtime(path) - duration
        self.start_time, self.realtime = start_time, realtime
        self._t0, self._index, self._last_pos = None, 0, None

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # Some decoders provide missing or repeated timestamps.
        if not math.isfinite(pos) or pos < 0:
            pos = self._index / self.fps
        if self._last_pos is not None and pos <= self._last_pos:
            pos = self._last_pos + 1.0 / self.fps
        self._last_pos = pos
        self._index += 1
        if self.realtime:
            if self._t0 is None:
                self._t0 = time.monotonic() - pos
            delay = self._t0 + pos - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return frame, self.start_time + pos

    def close(self):
        self.cap.release()


def open_source(spec, start_time=None, realtime=False):
    spec = str(spec)
    if spec == "pi":
        return PiSource()
    if spec.isdigit():
        return WebcamSource(int(spec))
    if spec.split("://")[0].lower() in ("http", "https", "rtsp", "rtmp", "srt", "udp"):
        return StreamSource(spec)
    return FileSource(spec, start_time=start_time, realtime=realtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("--start", help="recording start: YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    a = ap.parse_args()
    start = (datetime.strptime(a.start, "%Y-%m-%d %H:%M:%S").timestamp()
             if a.start else None)
    src = open_source(a.source, start_time=start, realtime=a.realtime)
    live = isinstance(src, StreamSource)
    window = "video_source (q to quit)"
    last_frame = None
    last_received = None
    n = 0
    recent = deque(maxlen=120)
    generation = None
    try:
        if not a.no_display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 640, 480)
        while True:
            frame, ts = src.read()
            now = time.monotonic()
            if frame is None and not live:
                break
            if frame is not None:
                if live and generation != src.last_generation:
                    generation = src.last_generation
                    recent.clear()
                last_frame, last_received = frame, now
                n += 1
                recent.append(now)
                while len(recent) > 2 and now - recent[0] > 3.0:
                    recent.popleft()
                fps = ((len(recent) - 1) / max(1e-6, recent[-1] - recent[0])
                       if len(recent) > 1 else 0.0)
                if n == 1 or n % 30 == 0:
                    print(f"frame {n} {frame.shape[1]}x{frame.shape[0]}  "
                          f"time {datetime.fromtimestamp(ts):%H:%M:%S.%f}"[:-3]
                          + f"  processed {fps:.1f} fps"
                          + (f" | capture {src.capture_fps:.1f} fps" if live else ""),
                          flush=True)
            if not a.no_display:
                stale = live and (not src.connected or last_received is None or
                                  now - last_received > 0.75)
                if stale or last_frame is None:
                    img = (last_frame.copy() if last_frame is not None else
                           np.zeros((480, 640, 3), dtype=np.uint8))
                    img = cv2.convertScaleAbs(img, alpha=0.35)
                    cv2.putText(img, "OFFLINE / WAITING FOR CAMERA", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                else:
                    img = last_frame
                cv2.imshow(window, img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
            if a.frames and n >= a.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if not a.no_display:
            cv2.destroyAllWindows()
        src.close()
    print(f"done: {n} frames")


if __name__ == "__main__":
    main()
