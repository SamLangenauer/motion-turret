# kalman.py — constant-velocity Kalman filter for pixel-space target tracking.
#
# State vector:  [cx, cy, vx, vy]  (position + velocity in pixels and pixels/sec)
# Measurement:   [cx, cy]          (raw detection centroid from vision.py)
#
# Three things this replaces / adds vs. the EMA smoother in vision.py:
#
#   1. Smoothing     — same job as the old alpha-blend, but principled:
#                      the filter weights measurements by how noisy they are
#                      relative to how much the state is expected to change.
#
#   2. Velocity      — we now know vx/vy in px/s, logged to latency.csv.
#                      Useful for future ballistic lead or adaptive gain.
#
#   3. Lead comp     — kalman.lead(dt) extrapolates the current state estimate
#                      dt seconds forward.  Set KALMAN_LEAD_MS in config.py to
#                      compensate for capture + servo lag (start at 20 ms).
#
# Tuning cheat-sheet
# ------------------
#   KALMAN_MEAS_NOISE_PX   larger → trust prediction more, smoother but laggier
#                          smaller → trust raw detections more, snappier but noisier
#   KALMAN_PROC_NOISE      larger → allow velocity to change quickly (fast targets)
#                          smaller → smoother velocity, slower to react to direction changes
#
# Reasonable starting point for a person walking across the frame at 1–2 m/s:
#   MEAS_NOISE ~8–12 px, PROC_NOISE ~40–80 px/s²

import time
import numpy as np
import cv2


class TargetKalman:
    """Constant-velocity Kalman filter, one instance per tracked target."""

    def __init__(self, frame_width: int, frame_height: int,
                 meas_noise_px: float = 8.0,
                 proc_noise: float = 60.0):
        self._fw = frame_width
        self._fh = frame_height
        self._meas_noise = meas_noise_px
        self._proc_noise = proc_noise

        self._kf: cv2.KalmanFilter | None = None
        self._initialized = False
        self._last_t: float | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(self, cx: float, cy: float) -> tuple[int, int, float, float]:
        """
        Feed a new measurement (raw detection centroid).
        Returns (smooth_cx, smooth_cy, vx_px_s, vy_px_s).
        Call every frame that vision returns a valid target.
        """
        now = time.monotonic()
        dt = self._dt(now)
        self._last_t = now

        if not self._initialized:
            self._kf = self._build(cx, cy)
            self._initialized = True
            return int(cx), int(cy), 0.0, 0.0

        self._set_transition(dt)
        self._kf.predict()
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        state = self._kf.correct(meas)
        return self._unpack(state)

    def predict(self) -> tuple[int, int, float, float] | None:
        """
        Advance the filter with no measurement (dead-reckoning).
        Returns (pred_cx, pred_cy, vx_px_s, vy_px_s), or None if not yet init.
        Call during settle frames and any frame where vision returns no target
        but the tracker hasn't been declared lost yet.
        """
        if not self._initialized:
            return None

        now = time.monotonic()
        dt = self._dt(now)
        self._last_t = now

        self._set_transition(dt)
        state = self._kf.predict()
        return self._unpack(state)

    def lead(self, lookahead_sec: float) -> tuple[int, int] | None:
        """
        Return the predicted position `lookahead_sec` seconds from now using
        the current velocity estimate.  Does NOT advance the filter state.
        Call after update() to get the lead-compensated aim point.
        Returns (lead_cx, lead_cy) or None if not initialized.
        """
        if not self._initialized:
            return None
        s = self._kf.statePost
        lx = float(s[0]) + float(s[2]) * lookahead_sec
        ly = float(s[1]) + float(s[3]) * lookahead_sec
        return (int(np.clip(lx, 0, self._fw)),
                int(np.clip(ly, 0, self._fh)))

    def reset(self):
        """Discard all state.  Call when the tracker declares LOST."""
        self._kf = None
        self._initialized = False
        self._last_t = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _dt(self, now: float) -> float:
        if self._last_t is None:
            return 0.033          # assume ~30 fps on first call
        return float(np.clip(now - self._last_t, 1e-3, 0.2))

    def _build(self, cx: float, cy: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)          # 4 state vars, 2 measured

        # H: measurement matrix — we observe position only
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # F: transition matrix — set per-call (dt varies)
        kf.transitionMatrix = np.eye(4, dtype=np.float32)

        # R: measurement noise covariance
        kf.measurementNoiseCov = (np.eye(2, dtype=np.float32)
                                  * (self._meas_noise ** 2))

        # Q: process noise covariance — position rows are small,
        #    velocity rows carry the lion's share of uncertainty.
        q = self._proc_noise ** 2
        kf.processNoiseCov = np.diag(
            [1.0, 1.0, q, q]
        ).astype(np.float32)

        # P: initial error covariance — high uncertainty everywhere
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 500.0

        # x: initial state — target is at (cx, cy), velocity unknown
        kf.statePost = np.array(
            [[cx], [cy], [0.0], [0.0]], dtype=np.float32
        )

        return kf

    def _set_transition(self, dt: float):
        self._kf.transitionMatrix = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=np.float32)

    def _unpack(self, state) -> tuple[int, int, float, float]:
        cx = int(np.clip(float(state[0]), 0, self._fw))
        cy = int(np.clip(float(state[1]), 0, self._fh))
        vx = float(state[2])
        vy = float(state[3])
        return cx, cy, vx, vy
