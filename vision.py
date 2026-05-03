# vision.py — camera capture + two-stage acquisition / tracking.
#
# Phase 7 changes (robustness):
#   - Tracker upgraded from MOSSE → CSRT.  CSRT uses channel and spatial
#     reliability weighting so it handles motion blur, partial occlusion, and
#     background clutter far better.  Cost: ~25-35 ms/frame vs ~5 ms for MOSSE.
#     At 80 FPS camera rate the control loop still runs at ~30 FPS during
#     tracking — smooth enough for servo control.
#   - _MAX_LOST raised 5 → 20.  Gives CSRT 20 consecutive frame failures
#     (~0.6 s) before declaring LOST.  Brief occlusions, doorway crossings, and
#     fast direction changes no longer immediately break lock.
#   - DNN acquisition is still fully async (tracker.py owns the thread).
#     force_tracking() is unchanged.

import time
import cv2
import numpy as np
from picamera2 import Picamera2
import config

_CLOCK_BOOTTIME = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)

def _now_boottime_ns():
    return time.clock_gettime_ns(_CLOCK_BOOTTIME)

def _make_tracker():
    """CSRT preferred; fall back to MOSSE then KCF if not available."""
    try:
        return cv2.legacy.TrackerCSRT_create()
    except (AttributeError, cv2.error):
        pass
    try:
        return cv2.TrackerCSRT_create()
    except (AttributeError, cv2.error):
        pass
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except (AttributeError, cv2.error):
        pass
    return cv2.legacy.TrackerKCF_create()

class MotionTracker:
    STATE_SEARCHING = "searching"
    STATE_ACQUIRING = "acquiring"
    STATE_TRACKING  = "tracking"
    STATE_LOST      = "lost"

    def __init__(self, mode="mog2"):
        self.picam = Picamera2()
        cam_controls = {"FrameRate": 80}
        video_config = self.picam.create_video_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT), "format": "BGR888"},
            sensor={"output_size": (1640, 1232), "bit_depth": 8},
            controls=cam_controls
        )
        self.picam.configure(video_config)
        self.picam.start()

        self.net = cv2.dnn.readNetFromCaffe(
            "MobileNetSSD_deploy.prototxt",
            "MobileNetSSD_deploy.caffemodel"
        )

        self._mode = mode
        if mode == "mog2":
            self._bgsub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False)
        elif mode in ("framediff", "tracker"):
            self._prev_gray = None
        else:
            raise ValueError(f"unknown mode {mode}")

        self._state = self.STATE_SEARCHING
        self._csrt = None
        self._candidate_box = None
        self._tracker_lost_streak = 0
        self._MAX_LOST = 20          # was 5; CSRT is robust enough to warrant patience
        self._last_bbox = None
        self._last_sensor_ts_ns = 0

    @property
    def last_sensor_ts_ns(self):
        return self._last_sensor_ts_ns

    def sensor_age_ms(self):
        if not self._last_sensor_ts_ns:
            return float("nan")
        return (_now_boottime_ns() - self._last_sensor_ts_ns) / 1e6

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        if timer is not None: timer.lap("preproc")
        return frame, gray

    def reset_background(self):
        if self._mode in ("framediff", "tracker"):
            self._prev_gray = None
        elif self._mode == "mog2":
            self._bgsub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False)

    @property
    def state(self):
        return self._state

    @property
    def is_tracking(self):
        return self._state == self.STATE_TRACKING

    @property
    def tracker_lost_streak(self):
        return self._tracker_lost_streak

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

    def detect(self, timer=None):
        frame, gray = self._read_frame(timer)
        if self._mode == "mog2":
            return self._detect_mog2(frame, gray, timer)
        elif self._mode == "framediff":
            return self._detect_framediff(frame, gray, timer)
        else:
            return self._detect_tracker(frame, gray, timer)

    def _detect_mog2(self, frame, gray, timer=None):
        if timer is not None: timer.mark("mog2")
        mask = self._bgsub.apply(gray, learningRate=0.005)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area >= config.MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest)
                if w > 0 and h > 0:
                    cx = x + w // 2
                    cy = y + int(h * config.AIM_VERTICAL_BIAS)
                    target = (cx, cy, area)
        if timer is not None: timer.lap("mog2")
        return frame, target, mask

    def _detect_framediff(self, frame, gray, timer=None):
        if timer is not None: timer.mark("framediff")
        bbox, area = self._framediff_largest_blob(gray)
        target = None
        if bbox is not None:
            x, y, w, h = bbox
            cx = x + w // 2
            cy = y + int(h * config.AIM_VERTICAL_BIAS)
            target = (cx, cy, area)
        if timer is not None: timer.lap("framediff")
        return frame, target, np.zeros_like(gray)

    def _detect_human(self, frame):
        """MobileNet-SSD inference. ~150 ms on Pi 4 — ONLY call from a background thread."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        best_box = None
        max_area = 0
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if idx == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX, startY = max(0, startX), max(0, startY)
                    endX,   endY   = min(w-1, endX), min(h-1, endY)
                    bw = endX - startX
                    bh = endY - startY
                    area = bw * bh
                    if area > max_area:
                        max_area = area
                        best_box = (startX, startY, bw, bh)
        if best_box is None:
            return None, 0
        return best_box, max_area

    def force_tracking(self, gray, bbox):
        """
        Called by tracker.py's DNN thread when a person is detected.
        Initialises CSRT and transitions directly to TRACKING.
        """
        self._csrt = _make_tracker()
        self._csrt.init(gray, bbox)
        self._state = self.STATE_TRACKING
        self._tracker_lost_streak = 0
        self._last_bbox = bbox

    def _detect_tracker(self, frame, gray, timer=None):
        """
        SEARCHING / LOST: return immediately — DNN is running async in tracker.py.
        TRACKING: run CSRT update.
        """
        if self._state in (self.STATE_SEARCHING, self.STATE_LOST):
            return frame, None, np.zeros_like(gray)

        if self._state == self.STATE_TRACKING:
            if timer is not None: timer.mark("kcf_update")
            ok, bbox = self._csrt.update(gray)
            if timer is not None: timer.lap("kcf_update")

            fh, fw = frame.shape[:2]
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                bbox = (x, y, w, h)
            else:
                x, y, w, h = 0, 0, 0, 0
                bbox = (0, 0, 0, 0)

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
                return frame, None, np.zeros_like(gray)

            self._tracker_lost_streak = 0
            self._last_bbox = bbox
            x, y, w, h = bbox
            cx = int(x + w / 2)
            cy = int(y + h * config.AIM_VERTICAL_BIAS)
            return frame, (cx, cy, w * h), np.zeros_like(gray)

        return frame, None, np.zeros_like(gray)

    def close(self):
        self.picam.stop()

    @property
    def last_bbox(self):
        return self._last_bbox
