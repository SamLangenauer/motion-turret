# vision.py — camera capture + motion detection

import cv2
import numpy as np
from picamera2 import Picamera2
import config


class MotionTracker:
    def __init__(self, mode="mog2"):
        """
        mode = 'mog2' for static camera (best detection on a fixed mount).
        mode = 'framediff' for moving camera (best for camera-on-gimbal use).
        """
        self.picam = Picamera2()
        video_config = self.picam.create_video_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                  "format": "RGB888"},
            sensor={"output_size": (1640, 1232), "bit_depth": 8}
        )
        self.picam.configure(video_config)
        self.picam.start()

        self._mode = mode
        if mode == "mog2":
            self._bgsub = cv2.createBackgroundSubtractorMOG2(
                history=200,
                varThreshold=16,
                detectShadows=False
            )
        elif mode == "framediff":
            self._prev_gray = None
        else:
            raise ValueError(f"unknown mode {mode}")

        self._smooth_cx = None
        self._smooth_cy = None
        self._smooth_alpha = 0.75

    def _read_gray_blur(self):
        frame = self.picam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        return frame, gray

    def reset_background(self):
        """Call after camera moves so the detector starts fresh."""
        if self._mode == "framediff":
            self._prev_gray = None
            self._skip_next = True
        elif self._mode == "mog2":
            self._bgsub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False
            )
        self._smooth_cx = None
        self._smooth_cy = None

    def detect(self):
        frame, gray = self._read_gray_blur()

        if self._mode == "mog2":
            mask = self._bgsub.apply(gray, learningRate=0.005)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.dilate(mask, None, iterations=2)
        else:  # framediff
            if self._prev_gray is None:
                self._prev_gray = gray
                return frame, None, np.zeros_like(gray)
            if getattr(self, "_skip_next", False):
                self._skip_next = False
                self._prev_gray = gray
                return frame, None, np.zeros_like(gray)
            delta = cv2.absdiff(self._prev_gray, gray)
            _, mask = cv2.threshold(delta, config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, None, iterations=2)
            self._prev_gray = gray

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area >= config.MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest)
                if w > 0 and h > 0:
                    raw_cx = x + w // 2
                    raw_cy = y + int(h * config.AIM_VERTICAL_BIAS)
                    if self._smooth_cx is None:
                        self._smooth_cx = float(raw_cx)
                        self._smooth_cy = float(raw_cy)
                    else:
                        a = self._smooth_alpha
                        self._smooth_cx = a * raw_cx + (1 - a) * self._smooth_cx
                        self._smooth_cy = a * raw_cy + (1 - a) * self._smooth_cy
                    target = (int(self._smooth_cx), int(self._smooth_cy), area)
        else:
            self._smooth_cx = None
            self._smooth_cy = None

        return frame, target, mask

    def close(self):
        self.picam.stop()
