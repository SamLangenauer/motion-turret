# vision.py — camera capture + motion detection

import cv2
import numpy as np
from picamera2 import Picamera2
import config


class MotionTracker:
    """
    Grabs frames from the Pi camera, runs frame-differencing motion detection,
    and reports the centroid of the largest moving object.
    """

    def __init__(self):
        self.picam = Picamera2()
        video_config = self.picam.create_video_configuration(
    		main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT),
          		"format": "RGB888"},
    		sensor={"output_size": (1640, 1232), "bit_depth": 8}
	)
        self.picam.configure(video_config)
        self.picam.start()

        # Running background model, float32 for smooth accumulation.
        self._background = None

        # How quickly the background adapts to new frames (0-1).
        # Lower = background updates slowly, good for tracking.
        # Higher = background updates fast, good for lighting changes.
        self._learning_rate = 0.05

    def _read_gray_blur(self):
        """Capture one frame, convert to grayscale, blur to reduce noise."""
        frame = self.picam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return frame, gray

    def detect(self):
        """
        Run one detection cycle.
        Returns: (frame, target, mask)
          frame  — color frame for display
          target — (cx, cy, area) of largest moving blob, or None
          mask   — binary motion mask (useful for debug display)
        """
        frame, gray = self._read_gray_blur()

        # Bootstrap the background on first frame
        if self._background is None:
            self._background = gray.astype("float32")
            return frame, None, np.zeros_like(gray)

        # Update running average, then compute absolute difference
        cv2.accumulateWeighted(gray, self._background, self._learning_rate)
        bg_uint8 = cv2.convertScaleAbs(self._background)
        delta = cv2.absdiff(bg_uint8, gray)

        # Threshold + dilate to fill small gaps in the motion blob
        _, mask = cv2.threshold(delta, config.MOTION_THRESHOLD,
                                255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours; pick the biggest one above min area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        target = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area >= config.MIN_CONTOUR_AREA:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    target = (cx, cy, area)

        return frame, target, mask

    def close(self):
        self.picam.stop()
