"""Opening the webcam robustly (shared by webcam.py and collect.py).

On Windows the default backend (Media Foundation) can refuse to stream for a few seconds right
after another program released the camera ("can't grab frame", error 0xC00D3704). The camera
is therefore only accepted once it actually delivers a frame, the DirectShow backend is tried
as a fallback, and short dropouts during a session are tolerated.
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np


def _first_frame(cap: cv2.VideoCapture, timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        ok, _ = cap.read()
        if ok:
            return True
        time.sleep(0.05)
    return False


def open_camera(index: int = 0, timeout: float = 3.0) -> cv2.VideoCapture | None:
    """Open camera `index` and wait until it delivers a frame. Returns None if it never does."""
    backends = [cv2.CAP_ANY]
    if sys.platform == "win32":
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
    for api in backends:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened() and _first_frame(cap, timeout):
            return cap
        cap.release()
    return None


class FrameReader:
    """cap.read() that tolerates brief dropouts; gives up after `max_gap` seconds without frames."""

    def __init__(self, cap: cv2.VideoCapture, max_gap: float = 3.0):
        self.cap = cap
        self.max_gap = max_gap

    def read(self) -> np.ndarray | None:
        deadline = time.perf_counter() + self.max_gap
        while True:
            ok, frame = self.cap.read()
            if ok:
                return frame
            if time.perf_counter() >= deadline:
                return None
            time.sleep(0.02)
