# vision.py — camera capture + two-stage acquisition / tracking.
#
# Modes:
#   "mog2"      — static-camera background subtractor.
#   "framediff" — frame differencing. Legacy mode, kept for diagnostics.
#   "tracker"   — two-stage: framediff acquires, MOSSE tracks. For camera-on-gimbal.
#
# Phase 4.5+:
#   - capture_request() so we can read libcamera SensorTimestamp.
#   - detect() and helpers accept an optional `timer` (instrument.StageTimer).
#   - Tracker is MOSSE (was KCF). MOSSE is ~5-10x faster on Pi 4 and is
#     sufficient for short-horizon tracking; the state machine handles loss.
#   - Tracker now consumes the precomputed grayscale image instead of RGB.

import time
import cv2
import numpy as np
from picamera2 import Picamera2
import config


_CLOCK_BOOTTIME = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)


def _now_boottime_ns():
    return time.clock_gettime_ns(_CLOCK_BOOTTIME)


def _make_tracker():
    """Build the fastest available correlation tracker.
    Preference: MOSSE (legacy) > KCF (new API) > KCF (legacy)."""
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except (AttributeError, cv2.error):
        pass
    try:
        return cv2.TrackerKCF_create()
    except AttributeError:
        return cv2.legacy.TrackerKCF_create()


class MotionTracker:
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

        # Variable kept named `_csrt` for legacy reasons; actual tracker is
        # built by _make_tracker() (MOSSE preferred).
        self._state = self.STATE_SEARCHING
        self._csrt = None
        self._candidate_box = None
        self._tracker_lost_streak = 0
        self._MAX_LOST = 5
        self._last_bbox = None

        self._smooth_cx = None
        self._smooth_cy = None
        self._smooth_alpha = 0.6

        self._last_sensor_ts_ns = 0

    # ---- timestamp accessors ----

    @property
    def last_sensor_ts_ns(self):
        return self._last_sensor_ts_ns

    def sensor_age_ms(self):
        if not self._last_sensor_ts_ns:
            return float("nan")
        return (_now_boottime_ns() - self._last_sensor_ts_ns) / 1e6

    # ---- shared utilities ----

    def _read_frame(self, timer=None):
        if timer is not None: timer.mark("capture")
        request = self.picam.capture_request()
        try:
            frame = request.make_array("main")
            metadata = request.get_metadata()
            self._last_sensor_ts_ns = metadata.get("SensorTimestamp", 0)
        finally:
            request.release()
        if timer is not None:
            timer.lap("capture")
            timer.mark("preproc")
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        if timer is not None: timer.lap("preproc")
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
        if self._mode == "framediff":
            self._prev_gray = None
        elif self._mode == "tracker":
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

    # ---- detection helpers ----

    def _framediff_largest_blob(self, gray):
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

    def detect(self, timer=None):
        frame, gray = self._read_frame(timer)
        if self._mode == "mog2":
            return self._detect_mog2(frame, gray, timer)
        elif self._mode == "framediff":
            return self._detect_framediff(frame, gray, timer)
        else:
            return self._detect_tracker(frame, gray, timer)

    # ---- mode implementations ----

    def _detect_mog2(self, frame, gray, timer=None):
        if timer is not None: timer.mark("mog2")
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
        if timer is not None: timer.lap("mog2")
        return frame, target, mask

    def _detect_framediff(self, frame, gray, timer=None):
        if timer is not None: timer.mark("framediff")
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
        if timer is not None: timer.lap("framediff")
        return frame, target, np.zeros_like(gray)

    def _detect_tracker(self, frame, gray, timer=None):
        target = None

        if self._state in (self.STATE_SEARCHING, self.STATE_LOST):
            if timer is not None: timer.mark("framediff")
            bbox, area = self._framediff_largest_blob(gray)
            if timer is not None: timer.lap("framediff")
            if bbox is not None:
                self._candidate_box = bbox
                self._state = self.STATE_ACQUIRING
            return frame, None, np.zeros_like(gray)

        if self._state == self.STATE_ACQUIRING:
            if timer is not None: timer.mark("framediff")
            bbox, area = self._framediff_largest_blob(gray)
            if timer is not None: timer.lap("framediff")
            if bbox is None:
                self._state = self.STATE_SEARCHING
                self._candidate_box = None
                return frame, None, np.zeros_like(gray)

            # Build the tracker (MOSSE preferred, KCF fallback)
            if timer is not None: timer.mark("kcf_init")
            self._csrt = _make_tracker()

            # Expand the motion bbox into a torso-biased aim box.
            x, y, w, h = bbox
            fh = frame.shape[0]
            new_h = int(h * 3.5)
            new_y = max(0, y - int(h * 0.5))
            new_h = min(new_h, fh - new_y)
            new_x = max(0, x - int(w * 0.3))
            new_w = int(w * 1.6)
            new_w = min(new_w, frame.shape[1] - new_x)
            bbox = (new_x, new_y, new_w, new_h)
            self._last_bbox = bbox

            # Initialize on grayscale (faster than RGB; both KCF and MOSSE
            # accept single-channel input).
            self._csrt.init(gray, bbox)
            if timer is not None: timer.lap("kcf_init")

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
            if timer is not None: timer.mark("kcf_update")
            ok, bbox = self._csrt.update(gray)
            if timer is not None: timer.lap("kcf_update")

            x, y, w, h = bbox
            fh, fw = frame.shape[:2]
            bbox_invalid = (
                w < 20 or h < 20 or
                w > fw * 0.9 or h > fh * 0.9 or
                x < 0 or y < 0 or
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

        return frame, None, np.zeros_like(gray)

    def close(self):
        self.picam.stop()

    @property
    def last_bbox(self):
        return self._last_bbox
