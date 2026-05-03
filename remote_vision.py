# remote_vision.py — Pi-side vision client for the hybrid laptop/Pi architecture.
#
# RemoteVisionClient is a drop-in replacement for MotionTracker when
# REMOTE_VISION = True in config.py.  It exposes the same interface:
#   .detect(timer)  →  (frame, target, mask)
#   .state          →  "searching" / "tracking" / "lost"
#   .is_tracking    →  bool
#   .last_bbox      →  (x, y, w, h) | None
#   .sensor_age_ms()
#   .reset_background()   (no-op; laptop owns the CV background model)
#
# What it does each frame:
#   1. Captures a frame with picamera2.
#   2. JPEG-encodes it and sends to the laptop via UDP (non-blocking send).
#   3. Drains the coord receive socket (non-blocking).  If a packet arrived,
#      updates the internal target state.
#   4. Returns the latest known target, or None if stale / not yet seen.
#
# Staleness: if no coord packet has been received for COORD_STALE_MS the
# target is treated as absent.  The Kalman filter in tracker.py will then
# dead-reckon for its own grace window before giving up.
#
# The Pi does NOT run DNN or CSRT — those live entirely on the laptop.
# tracker.py's _maybe_launch_dnn() is disabled in remote mode.

import time
import socket
import cv2
import numpy as np
from picamera2 import Picamera2
import config
from net_protocol import (
    encode_frame, decode_coord,
    STATE_TRACKING, STATE_SEARCHING, STATE_LOST, state_str,
)

# How long without a coord packet before we treat the target as gone.
COORD_STALE_MS = 500


class RemoteVisionClient:
    """Pi-side vision stub.  Sends frames; receives coordinates."""

    STATE_SEARCHING = "searching"
    STATE_ACQUIRING = "acquiring"
    STATE_TRACKING  = "tracking"
    STATE_LOST      = "lost"

    def __init__(self):
        # ---- camera ----
        self.picam = Picamera2()
        video_config = self.picam.create_video_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                  "format": "BGR888"},
            sensor={"output_size": (1640, 1232), "bit_depth": 8},
            controls={"FrameRate": 80},
        )
        self.picam.configure(video_config)
        self.picam.start()

        # ---- UDP sockets ----
        # Outbound: frames to laptop
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tx.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)
        self._laptop = (config.LAPTOP_IP, config.FRAME_TX_PORT)

        # Inbound: coord packets from laptop
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.bind(("0.0.0.0", config.COORD_RX_PORT))
        self._rx.setblocking(False)

        # ---- internal state ----
        self._state     = self.STATE_SEARCHING
        self._last_bbox = None           # (x, y, w, h) synthesised from cx/cy
        self._last_cx   = None
        self._last_cy   = None
        self._last_bbox_w = 0
        self._last_bbox_h = 0
        self._last_conf = 0.0
        self._coord_ts  = 0.0            # monotonic time of last received packet
        self._coord_frame_id = -1            # frame_id of last received coord packet
        self._frame_id  = 0
        self._last_sensor_ts_ns = 0
        # Telemetry included in each frame sent to laptop
        self._tel_lidar_dist = 0    # cm, 0 = no reading
        self._tel_lidar_str  = 0
        self._track_start_ts = 0.0  # monotonic time when lock was acquired

    # ------------------------------------------------------------------ #
    # MotionTracker-compatible interface                                   #
    # ------------------------------------------------------------------ #

    @property
    def state(self):
        return self._state

    @property
    def is_tracking(self):
        return self._state == self.STATE_TRACKING

    @property
    def last_bbox(self):
        return self._last_bbox

    @property
    def coord_frame_id(self):
        """Frame ID of the most recently received coord packet.
        tracker.py compares this against the last frame it commanded on
        to avoid issuing multiple servo moves for the same stale coord."""
        return self._coord_frame_id

    def update_telemetry(self, lidar_dist_cm: int, lidar_strength: int):
        """Called by tracker.py each frame with the latest LiDAR reading."""
        self._tel_lidar_dist = lidar_dist_cm
        self._tel_lidar_str  = lidar_strength

    @property
    def track_ms(self) -> int:
        """Milliseconds continuously locked on target, 0 if not tracking."""
        if self._state != self.STATE_TRACKING or self._track_start_ts == 0.0:
            return 0
        return int((time.monotonic() - self._track_start_ts) * 1000)

    def sensor_age_ms(self):
        if not self._last_sensor_ts_ns:
            return float("nan")
        ref = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)
        return (time.clock_gettime_ns(ref) - self._last_sensor_ts_ns) / 1e6

    def reset_background(self):
        pass   # background model lives on the laptop

    def detect(self, timer=None):
        """
        Capture frame → send to laptop → drain coord socket → return target.
        Always O(1) — no DNN, no CSRT.
        """
        # 1. Capture
        if timer is not None: timer.mark("capture")
        request = self.picam.capture_request()
        try:
            frame = request.make_array("main")
            meta  = request.get_metadata()
            self._last_sensor_ts_ns = meta.get("SensorTimestamp", 0)
        finally:
            request.release()
        if timer is not None:
            timer.lap("capture")
            timer.mark("preproc")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        if timer is not None: timer.lap("preproc")

        # 2. Send frame to laptop (best-effort, non-blocking)
        self._send_frame(frame)

        # 3. Drain coord socket — take the latest packet if multiple queued
        self._recv_coords()

        # 4. Build target from latest coord if fresh
        target = self._current_target()

        return frame, target, np.zeros(frame.shape[:2], dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _send_frame(self, frame):
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, config.REMOTE_FRAME_QUALITY],
        )
        if not ok:
            return
        jpeg = buf.tobytes()
        if len(jpeg) > 65000:
            # Frame too large for single UDP datagram — skip this one.
            # Shouldn't happen at quality 40 with 680×480, but be safe.
            return
        self._frame_id = (self._frame_id + 1) & 0xFFFFFFFF
        pkt = encode_frame(self._frame_id, jpeg,
                           config.FRAME_WIDTH, config.FRAME_HEIGHT,
                           self._tel_lidar_dist, self._tel_lidar_str,
                           self.track_ms)
        try:
            self._tx.sendto(pkt, self._laptop)
        except OSError:
            pass

    def _recv_coords(self):
        """Non-blocking drain — keep only the most recent packet."""
        latest = None
        try:
            while True:
                data, _ = self._rx.recvfrom(64)
                latest = data
        except BlockingIOError:
            pass

        if latest is None:
            return

        try:
            frame_id, state_int, cx, cy, bbox_w, bbox_h, conf = decode_coord(latest)
        except Exception:
            return

        prev_state      = self._state
        self._state     = state_str(state_int)
        # Start/stop lock timer on state transitions
        if self._state == self.STATE_TRACKING and prev_state != self.STATE_TRACKING:
            self._track_start_ts = time.monotonic()
        elif self._state != self.STATE_TRACKING:
            self._track_start_ts = 0.0
        self._last_cx   = cx
        self._last_cy   = cy
        self._last_bbox_w = bbox_w
        self._last_bbox_h = bbox_h
        self._last_conf = conf
        self._coord_ts  = time.monotonic()
        self._coord_frame_id = frame_id

        # Build overlay bbox from exact dimensions sent by the laptop
        if self._state == self.STATE_TRACKING and bbox_w > 0 and bbox_h > 0:
            x = max(0, cx - bbox_w // 2)
            y = max(0, cy - int(bbox_h * 0.5))
            self._last_bbox = (x, y, bbox_w, bbox_h)
        else:
            self._last_bbox = None

    def _current_target(self):
        """Return (cx, cy, area) if we have a fresh tracking coord, else None."""
        if self._state != self.STATE_TRACKING:
            return None
        if self._last_cx is None:
            return None
        age_ms = (time.monotonic() - self._coord_ts) * 1000
        if age_ms > COORD_STALE_MS:
            self._state = self.STATE_LOST
            return None
        return (self._last_cx, self._last_cy,
                self._last_bbox_w * self._last_bbox_h)

    def close(self):
        self.picam.stop()
        self._tx.close()
        self._rx.close()
