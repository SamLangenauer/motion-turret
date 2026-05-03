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
#
# Phase 5 (Kalman):
#   - EMA smoother (_smooth / _smooth_cx / _smooth_cy) removed entirely.
#     tracker.py's TargetKalman now owns all smoothing and velocity estimation.
#   - detect() returns raw centroid; no position filtering done here.

import time
import cv2
import numpy as np
from picamera2 import Picamera2
import config


_CLOCK_BOOTTIME = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)


def _now_boottime_ns():
    return time.clock_gettime_ns(_CLOCK_BOOTTIME)

def _make_tracker():
    """Build the fastest available correlation tracker (MOSSE)."""
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
        cam_controls = {"FrameRate": 80}
        video_config = self.picam.create_video_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                  "format": "BGR888"},
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        if timer is not None: timer.lap("preproc")
        return frame, gray

    def reset_background(self):
        if self._mode == "framediff":
            self._prev_gray = None
        elif self._mode == "tracker":
            self._prev_gray = None
        elif self._mode == "mog2":
            self._bgsub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False
            )

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
        """Scans for humans using a MobileNet-SSD Neural Network."""
        h, w = frame.shape[:2]
        
        # Format the image for the neural net (300x300 is what MobileNet expects)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                     0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        best_box = None
        max_area = 0

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections (tune this between 0.4 and 0.6)
            if confidence > 0.5:
                # Extract the class ID (Class 15 is 'Person' in MobileNet-SSD)
                idx = int(detections[0, 0, i, 1])
                
                if idx == 15:
                    # Compute the (x, y) coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure bounding box is within frame
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w - 1, endX), min(h - 1, endY)
                    
                    bw = endX - startX
                    bh = endY - startY
                    area = bw * bh
                    
                    if area > max_area:
                        max_area = area
                        best_box = (startX, startY, bw, bh)

        if best_box is None:
            return None, 0
            
        return best_box, max_area

    def _detect_tracker(self, frame, gray, timer=None):
        target = None

        if self._state in (self.STATE_SEARCHING, self.STATE_LOST):
            if timer is not None: timer.mark("dnn_search")
            bbox, area = self._detect_human(frame)
            if timer is not None: timer.lap("dnn_search")

            # If no human found, keep searching
            if bbox is None or area <= 2000:
                return frame, None, np.zeros_like(gray)

            # --- Human found! Initialize MOSSE immediately ---
            if timer is not None: timer.mark("kcf_init") 
            self._csrt = _make_tracker()

            x, y, w, h = bbox
            
            # --- THE FIX: Tight upper-body crop for MOSSE ---
            # Don't inflate the box. MobileNet already found the whole body.
            # We want a tight patch on the head/chest so MOSSE doesn't memorize the wall.
            track_w = min(int(w * 0.6), 150)
            track_h = min(int(h * 0.4), 150)
            
            # Center it horizontally, shift it up to the head/chest area
            track_x = x + (w // 2) - (track_w // 2)
            track_y = y + int(h * 0.1)
            
            # Clamp to frame boundaries so we don't crash at the edges
            fh, fw = frame.shape[:2]
            track_x = max(0, min(track_x, fw - track_w))
            track_y = max(0, min(track_y, fh - track_h))
            
            bbox = (int(track_x), int(track_y), int(track_w), int(track_h))
            self._last_bbox = bbox
            self._csrt.init(gray, bbox)
            if timer is not None: timer.lap("kcf_init")

            # Jump straight to tracking
            self._state = self.STATE_TRACKING
            self._tracker_lost_streak = 0
            self._candidate_box = None

            x, y, w, h = bbox
            cx = x + w // 2
            cy = y + int(h * config.AIM_VERTICAL_BIAS)
            target = (cx, cy, w * h)
            return frame, target, np.zeros_like(gray)

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
            target = (cx, cy, w * h)
            return frame, target, np.zeros_like(gray)

        return frame, None, np.zeros_like(gray)

    def close(self):
        self.picam.stop()

    @property
    def last_bbox(self):
        return self._last_bbox
