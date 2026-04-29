# vision.py — camera capture + two-stage acquisition / tracking
#
# Modes:
#   "mog2"      — static-camera background subtractor. Best for test_vision.py
#                 on a fixed mount.
#   "framediff" — frame differencing. Legacy mode, kept for diagnostics.
#   "tracker"   — two-stage: framediff to acquire, CSRT to track. Designed for
#                 camera-on-gimbal use where the camera itself moves.

import cv2
import numpy as np
from picamera2 import Picamera2
import config


class MotionTracker:
    # State machine for "tracker" mode
    STATE_SEARCHING = "searching"
    STATE_ACQUIRING = "acquiring"
    STATE_TRACKING  = "tracking"
    STATE_LOST      = "lost"

    def __init__(self, mode="mog2"):
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
                history=200, varThreshold=16, detectShadows=False
            )
        elif mode in ("framediff", "tracker"):
            self._prev_gray = None
        else:
            raise ValueError(f"unknown mode {mode}")

        # State for "tracker" mode
        self._state = self.STATE_SEARCHING
        self._csrt = None
        self._candidate_box = None       # bbox awaiting confirmation
        self._tracker_lost_streak = 0    # how many consecutive update() failures
        self._MAX_LOST = 5               # before declaring track dead
        self._last_bbox = None

        # Smoothing on output position
        self._smooth_cx = None
        self._smooth_cy = None
        self._smooth_alpha = 0.6

    # ---- shared utilities ----

    def _read_frame(self):
        """Capture and return (rgb_frame, gray_blurred)."""
        frame = self.picam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        return frame, gray

    def _smooth(self, raw_cx, raw_cy):
        if self._smooth_cx is None:
            self._smooth_cx = float(raw_cx)
            self._smooth_cy = float(raw_cy)
        else:
            a = self._smooth_alpha
            self._smooth_cx = a * raw_cx + (1 - a) * self._smooth_cx
            self._smooth_cy = a * raw_cy + (1 - a) * self._smooth_cy
        return int(self._smooth_cx), int(self._smooth_cy)

    def reset_background(self):
        """Force a fresh background reference. Called after camera moves."""
        if self._mode == "framediff":
            self._prev_gray = None
        elif self._mode == "tracker":
            # Don't reset CSRT — it's tracking the target, not the background.
            # Only reset the search-stage prev_gray.
            self._prev_gray = None
        elif self._mode == "mog2":
            self._bgsub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False
            )
        self._smooth_cx = None
        self._smooth_cy = None

    @property
    def state(self):
        return self._state

    @property
    def is_tracking(self):
        return self._state == self.STATE_TRACKING

    # ---- detection: framediff helper, used by both legacy and tracker modes ----

    def _framediff_largest_blob(self, gray):
        """Return (bbox, area) of largest motion blob, or (None, 0)."""
        if self._prev_gray is None:
            self._prev_gray = gray
            return None, 0

        delta = cv2.absdiff(self._prev_gray, gray)
        _, mask = cv2.threshold(delta, config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        self._prev_gray = gray

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < config.MIN_CONTOUR_AREA:
            return None, 0
        return cv2.boundingRect(largest), area

    # ---- public detect() — branches on mode ----

    def detect(self):
        frame, gray = self._read_frame()

        if self._mode == "mog2":
            return self._detect_mog2(frame, gray)
        elif self._mode == "framediff":
            return self._detect_framediff(frame, gray)
        else:  # tracker
            return self._detect_tracker(frame, gray)

    # ---- mode implementations ----

    def _detect_mog2(self, frame, gray):
        mask = self._bgsub.apply(gray, learningRate=0.005)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, None, iterations=2)

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
                    cx, cy = self._smooth(raw_cx, raw_cy)
                    target = (cx, cy, area)
        else:
            self._smooth_cx = None
            self._smooth_cy = None
        return frame, target, mask

    def _detect_framediff(self, frame, gray):
        bbox, area = self._framediff_largest_blob(gray)
        target = None
        if bbox is not None:
            x, y, w, h = bbox
            raw_cx = x + w // 2
            raw_cy = y + int(h * config.AIM_VERTICAL_BIAS)
            cx, cy = self._smooth(raw_cx, raw_cy)
            target = (cx, cy, area)
        else:
            self._smooth_cx = None
            self._smooth_cy = None
        # No mask returned for framediff (we don't need it)
        return frame, target, np.zeros_like(gray)

    def _detect_tracker(self, frame, gray):
        target = None

        if self._state in (self.STATE_SEARCHING, self.STATE_LOST):
            # Look for motion to acquire
            bbox, area = self._framediff_largest_blob(gray)
            if bbox is not None:
                self._candidate_box = bbox
                self._state = self.STATE_ACQUIRING
            # No actionable target yet
            return frame, None, np.zeros_like(gray)

        if self._state == self.STATE_ACQUIRING:
            # Confirm motion is still there roughly where we saw it
            bbox, area = self._framediff_largest_blob(gray)
            if bbox is None:
                # Fizzled — back to searching
                self._state = self.STATE_SEARCHING
                self._candidate_box = None
                return frame, None, np.zeros_like(gray)

            # Initialize CSRT on the confirmed bbox using the RGB frame
            # (CSRT works on color, not grayscale)
            try:
                self._csrt = cv2.TrackerKCF_create()
            except AttributeError:
                self._csrt = cv2.legacy.TrackerKCF_create()

            # Expand the motion bbox into a torso-biased aim box. Framediff
            # tends to find limbs (which move most); we want the tracker to
            # lock onto the body, not the hand or shoulder.
            x, y, w, h = bbox
            fh = frame.shape[0]
            # Expand height: 50% upward, 200% downward (toward feet), capped at frame
            new_h = int(h * 3.5)
            new_y = max(0, y - int(h * 0.5))
            new_h = min(new_h, fh - new_y)
            # Slight horizontal expansion too, but less aggressive
            new_x = max(0, x - int(w * 0.3))
            new_w = int(w * 1.6)
            new_w = min(new_w, frame.shape[1] - new_x)
            bbox = (new_x, new_y, new_w, new_h)
            self._last_bbox = bbox

            self._csrt.init(frame, bbox)
            self._state = self.STATE_TRACKING
            self._tracker_lost_streak = 0
            self._candidate_box = None

            x, y, w, h = bbox
            raw_cx = x + w // 2
            raw_cy = y + int(h * config.AIM_VERTICAL_BIAS)
            cx, cy = self._smooth(raw_cx, raw_cy)
            target = (cx, cy, w * h)
            return frame, target, np.zeros_like(gray)

        if self._state == self.STATE_TRACKING:
            ok, bbox = self._csrt.update(frame)

            # KCF sometimes returns ok=True with a degenerate bbox. Validate.
            x, y, w, h = bbox
            fh, fw = frame.shape[:2]
            bbox_invalid = (
                w < 20 or h < 20 or                  # collapsed
                w > fw * 0.9 or h > fh * 0.9 or      # exploded
                x < 0 or y < 0 or                    # out of frame
                x + w > fw or y + h > fh
            )

            if not ok or bbox_invalid:
                self._tracker_lost_streak += 1
                if self._tracker_lost_streak >= self._MAX_LOST:
                    self._state = self.STATE_LOST
                    self._csrt = None
                    self._last_bbox = None
                    self._smooth_cx = None
                    self._smooth_cy = None
                return frame, None, np.zeros_like(gray)

            self._tracker_lost_streak = 0
            self._last_bbox = bbox

            raw_cx = int(x + w / 2)
            raw_cy = int(y + h * config.AIM_VERTICAL_BIAS)
            cx, cy = self._smooth(raw_cx, raw_cy)
            target = (cx, cy, w * h)
            return frame, target, np.zeros_like(gray)

        # Shouldn't reach here
        return frame, None, np.zeros_like(gray)

    def close(self):
        self.picam.stop()

    @property
    def last_bbox(self):
        return self._last_bbox
