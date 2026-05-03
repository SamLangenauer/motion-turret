# tracker.py — closed-loop PD controller driving the turret to follow a tracked target.
#
# Phase 7 changes (robustness):
#   - Kalman is no longer reset when transitioning TRACKING → LOST.
#     The filter keeps dead-reckoning the predicted position so the turret
#     continues leading toward where the target is going while the async DNN
#     searches.  Kalman is only hard-reset when:
#       (a) the DNN finds someone new  (_dnn_worker calls kalman.reset() before
#           force_tracking so the new detection starts fresh), or
#       (b) KALMAN_GRACE_FRAMES have elapsed with no re-acquisition, at which
#           point velocity is stale enough to be harmful.
#   - vision.reset_background() is still called on LOST so framediff doesn't
#     see the gimbal vibration as motion.
#   - _relock_counter resets on LOST so the DNN fires immediately on the next
#     frame rather than waiting for the 30-frame re-lock window.

import threading
import http.server
import socketserver
import time
import os
import cv2 as _cv2

import config
from servos import Turret
from instrument import StageTimer
from kalman import TargetKalman

if config.REMOTE_VISION:
    from remote_vision import RemoteVisionClient as _VisionClass
else:
    from vision import MotionTracker as _VisionClass

if config.LIDAR_ENABLED:
    from lidar import LidarReader
else:
    LidarReader = None

DNN_RELOCK_FRAMES = 30

# How many frames after going LOST before we give up on the Kalman velocity
# estimate and do a hard reset.  At ~30 fps tracking rate, 60 frames ≈ 2 s.
# Within this window the turret dead-reckons toward the predicted position.
KALMAN_GRACE_FRAMES = 60


class Tracker:
    def __init__(self, enable_stream=True, latency_csv="latency.csv"):
        self.turret = Turret()
        self.vision = _VisionClass() if config.REMOTE_VISION else _VisionClass(mode="tracker")
        self.kalman = TargetKalman(
            frame_width=config.FRAME_WIDTH,
            frame_height=config.FRAME_HEIGHT,
            meas_noise_px=config.KALMAN_MEAS_NOISE_PX,
            proc_noise=config.KALMAN_PROC_NOISE,
        )

        self.lidar = LidarReader() if LidarReader is not None else None

        self._frame_cx = config.FRAME_WIDTH // 2
        self._frame_cy = config.FRAME_HEIGHT // 2

        self._dnn_thread = None
        self._dnn_lock = threading.Lock()
        self._dnn_pending = False
        self._relock_counter = 0

        self._SETTLE_FRAMES = 2
        self._settle = 0
        self._warmup_frames = 30
        self._lost_frames = 0       # frames since last good CSRT track
        self._kalman_grace = 0      # counts down from KALMAN_GRACE_FRAMES on LOST
        self._track_streak = 0

        self._patrol_direction = -1
        self._patrol_speed = 5
        self._patrol_max = config.PAN_MAX_ANGLE
        self._patrol_min = config.PAN_MIN_ANGLE

        self._last_frame = None
        self._last_target = None
        self._prev_vision_state = None

        self._enable_stream = enable_stream
        self._stream_jpg = None
        self._stream_lock = threading.Lock()
        self._stream_frame_counter = 0
        self._STREAM_EVERY_N = 3
        self._STREAM_QUALITY = 40

        self.timer = StageTimer(latency_csv, summary_every=60) if latency_csv else None
        self._prev_err_x = 0.0
        self._prev_err_y = 0.0
        # Remote mode: track last coord frame_id we issued a command on.
        # Prevents multiple servo nudges from the same stale laptop coord.
        self._last_commanded_coord_id = -1

    # ---- async DNN ----

    def _dnn_worker(self, frame_copy):
        bbox, area = self.vision._detect_human(frame_copy)
        if bbox is not None and area > 2000:
            x, y, w, h = bbox
            track_w = min(int(w * 0.6), 150)
            track_h = min(int(h * 0.4), 150)
            track_x = max(0, min(x + (w // 2) - (track_w // 2), config.FRAME_WIDTH - track_w))
            track_y = max(0, min(y + int(h * 0.1), config.FRAME_HEIGHT - track_h))
            tight_bbox = (int(track_x), int(track_y), int(track_w), int(track_h))
            gray = _cv2.cvtColor(frame_copy, _cv2.COLOR_BGR2GRAY)
            gray = _cv2.GaussianBlur(gray, (11, 11), 0)
            with self._dnn_lock:
                self.kalman.reset()            # new target — start filter fresh
                self.vision.force_tracking(gray, tight_bbox)
                self._kalman_grace = 0         # cancel any grace countdown
                self._lost_frames = 0
        with self._dnn_lock:
            self._dnn_pending = False

    def _maybe_launch_dnn(self):
        if self._last_frame is None:
            return
        is_tracking = self.vision.is_tracking
        if is_tracking:
            self._relock_counter += 1
            if self._relock_counter < DNN_RELOCK_FRAMES:
                return
            self._relock_counter = 0
        with self._dnn_lock:
            if self._dnn_pending:
                return
            self._dnn_pending = True
        frame_copy = self._last_frame.copy()
        self._dnn_thread = threading.Thread(
            target=self._dnn_worker, args=(frame_copy,), daemon=True)
        self._dnn_thread.start()

    # ---- range gate ----

    def _range_gate(self, aim):
        if aim is None or self.lidar is None:
            return aim
        reading = self.lidar.get()
        if reading is None:
            return aim
        dist_cm, _ = reading
        if dist_cm < config.LIDAR_MIN_RANGE_CM or dist_cm > config.LIDAR_MAX_RANGE_CM:
            return None
        return aim

    # ---- main step ----

    def step(self):
        if self.timer is not None:
            self.timer.frame_begin()

        range_cm = None
        if self.lidar is not None:
            reading = self.lidar.get()
            if reading is not None:
                range_cm, _ = reading
        if self.timer is not None:
            self.timer.note("range_cm", f"{range_cm:.0f}" if range_cm is not None else "")

        if self._warmup_frames > 0:
            self.vision.detect(timer=self.timer)
            self._warmup_frames -= 1
            if self.timer is not None:
                self.timer.note("phase", "warmup")
                self.timer.note("state", self.vision.state)
                self.timer.note("sensor_age_ms", f"{self.vision.sensor_age_ms():.2f}")
                self.timer.frame_end()
            return None

        if self._settle > 0:
            frame, target, _ = self.vision.detect(timer=self.timer)
            self._last_frame = frame
            self._settle -= 1
            curr_state = self.vision.state
            if target is not None:
                raw_cx, raw_cy, area = target
                kx, ky, vx, vy = self.kalman.update(raw_cx, raw_cy)
                self._last_target = (kx, ky, area)
            else:
                pred = self.kalman.predict()
                if pred is not None:
                    kx, ky, vx, vy = pred
                    self._last_target = (kx, ky, 0)
            self._handle_state_transition(curr_state)
            if not config.REMOTE_VISION:
                self._maybe_launch_dnn()
            if self._enable_stream:
                if self.timer is not None: self.timer.mark("stream")
                self._maybe_encode_stream_frame()
                if self.timer is not None: self.timer.lap("stream")
            if self.timer is not None:
                self.timer.note("phase", "settle")
                self.timer.note("settle_remaining", self._settle)
                self.timer.note("state", curr_state)
                self.timer.note("sensor_age_ms", f"{self.vision.sensor_age_ms():.2f}")
                self.timer.frame_end()
            return None

        # In remote mode the laptop owns the DNN — don't run it locally.
        if not config.REMOTE_VISION:
            self._maybe_launch_dnn()

        frame, target, _ = self.vision.detect(timer=self.timer)
        self._last_frame = frame

        curr_state = self.vision.state
        self._handle_state_transition(curr_state)

        vx, vy = 0.0, 0.0
        aim = None

        if target is not None:
            # CSRT gave us a position — update Kalman with the measurement.
            raw_cx, raw_cy, area = target
            if self.timer is not None: self.timer.mark("kalman")
            kx, ky, vx, vy = self.kalman.update(raw_cx, raw_cy)
            speed = (vx**2 + vy**2) ** 0.5
            if speed > 30 and config.KALMAN_LEAD_MS > 0:
                lead = self.kalman.lead(config.KALMAN_LEAD_MS / 1000.0)
                aim_cx, aim_cy = lead if lead else (kx, ky)
            else:
                aim_cx, aim_cy = kx, ky
            if self.timer is not None: self.timer.lap("kalman")
            aim = (aim_cx, aim_cy, area)

        elif self.kalman.initialized:
            # No CSRT measurement this frame (brief failure or LOST state),
            # but Kalman has a live estimate — dead-reckon.
            # This covers both: CSRT glitch during TRACKING, and the grace
            # window after transitioning to LOST.
            pred = self.kalman.predict()
            if pred is not None:
                kx, ky, vx, vy = pred
                aim = (kx, ky, 0)

        gated_aim = self._range_gate(aim)
        range_gated = (aim is not None and gated_aim is None)
        if self.timer is not None:
            self.timer.note("range_gated", "1" if range_gated else "0")
            self.timer.note("kalman_grace", self._kalman_grace)

        self._last_target = aim

        if self._enable_stream:
            if self.timer is not None: self.timer.mark("stream")
            self._maybe_encode_stream_frame(range_gated=range_gated)
            if self.timer is not None: self.timer.lap("stream")

        result = None
        commanded = False

        # We command servos if we have a gated aim point — whether it came
        # from CSRT or Kalman dead-reckoning.  Patrol only kicks in when
        # Kalman grace expires AND we're past LOST_LOCK_FRAMES_TO_RECENTER.
        if gated_aim is not None:
            self._lost_frames = 0
            self._track_streak += 1
            cx, cy, _area = gated_aim
            err_x = cx - self._frame_cx
            err_y = cy - self._frame_cy
            # In remote mode, only command when we have a genuinely new
            # coord from the laptop — prevents 3-5x repeated nudges from
            # the same stale packet while the Pi outruns the laptop's CV.
            coord_id = (getattr(self.vision, "coord_frame_id", -2)
                        if config.REMOTE_VISION else -2)
            fresh_coord = (not config.REMOTE_VISION or
                           coord_id != self._last_commanded_coord_id)

            if fresh_coord and (abs(err_x) >= config.DEADZONE_PIXELS or
                                abs(err_y) >= config.DEADZONE_PIXELS):
                d_err_x = err_x - self._prev_err_x
                d_err_y = err_y - self._prev_err_y
                dpan  = (config.KP_PAN  * err_x) + (config.KD_PAN  * d_err_x)
                dtilt = -(config.KP_TILT * err_y) - (config.KD_TILT * d_err_y)
                self._prev_err_x = err_x
                self._prev_err_y = err_y
                dpan  = max(-config.MAX_NUDGE_DEG, min(config.MAX_NUDGE_DEG, dpan))
                dtilt = max(-config.MAX_NUDGE_DEG, min(config.MAX_NUDGE_DEG, dtilt))
                if self.timer is not None: self.timer.mark("control")
                self.turret.nudge(dpan, dtilt)
                if self.timer is not None: self.timer.lap("control")
                commanded = True
                self._last_commanded_coord_id = coord_id
                max_step = max(abs(dpan), abs(dtilt))
                self._settle = max(self._SETTLE_FRAMES, int(round(max_step / 3.0)))
                if self.timer is not None:
                    self.timer.note("dpan", f"{dpan:.2f}")
                    self.timer.note("dtilt", f"{dtilt:.2f}")
                    self.timer.note("settle_set", self._settle)
            result = gated_aim
        else:
            self._lost_frames += 1
            self._track_streak = 0
            if self._lost_frames > config.LOST_LOCK_FRAMES_TO_RECENTER:
                if self.timer is not None: self.timer.mark("control")
                pan, tilt = self.turret.position
                next_pan = pan + (self._patrol_speed * self._patrol_direction)
                if next_pan > self._patrol_max:
                    self._patrol_direction = -1
                elif next_pan < self._patrol_min:
                    self._patrol_direction = 1
                dtilt = 0
                if tilt < config.TILT_CENTER - 1: dtilt = 1
                elif tilt > config.TILT_CENTER + 1: dtilt = -1
                self.turret.nudge(self._patrol_speed * self._patrol_direction, dtilt)
                self._settle = 1
                if self.timer is not None: self.timer.lap("control")
                commanded = True

        if self.timer is not None:
            self.timer.note("phase", "active")
            self.timer.note("state", curr_state)
            self.timer.note("commanded", "1" if commanded else "0")
            self.timer.note("sensor_age_ms", f"{self.vision.sensor_age_ms():.2f}")
            self.timer.note("vx_px_s", f"{vx:.1f}")
            self.timer.note("vy_px_s", f"{vy:.1f}")
            self.timer.frame_end()
        return result

    # ---- helpers ----

    def _handle_state_transition(self, curr_state):
        if (curr_state in (_VisionClass.STATE_LOST, _VisionClass.STATE_SEARCHING)
                and self._prev_vision_state == _VisionClass.STATE_TRACKING):
            # Don't reset Kalman here — let it dead-reckon for KALMAN_GRACE_FRAMES.
            # _dnn_worker will call kalman.reset() when it finds someone new.
            self._kalman_grace = KALMAN_GRACE_FRAMES
            self.vision.reset_background()
            self._settle = max(self._settle, self._SETTLE_FRAMES * 5)
            self._relock_counter = 0

        # Count down the grace window each frame we're in LOST/SEARCHING.
        if curr_state in (_VisionClass.STATE_LOST, _VisionClass.STATE_SEARCHING):
            if self._kalman_grace > 0:
                self._kalman_grace -= 1
                if self._kalman_grace == 0:
                    self.kalman.reset()   # grace expired — velocity is stale

        self._prev_vision_state = curr_state

    # ---- stream ----

    def _maybe_encode_stream_frame(self, range_gated=False):
        self._stream_frame_counter += 1
        if self._stream_frame_counter % self._STREAM_EVERY_N != 0:
            return
        if self._last_frame is None:
            return
        bgr = self._last_frame.copy()
        h, w = bgr.shape[:2]
        cx0, cy0 = w // 2, h // 2
        _cv2.line(bgr, (cx0-20, cy0), (cx0+20, cy0), (0,255,0), 1)
        _cv2.line(bgr, (cx0, cy0-20), (cx0, cy0+20), (0,255,0), 1)
        if self._last_target is not None:
            tx, ty, _ = self._last_target
            dot_color = (0,140,255) if range_gated else (0,0,255)
            _cv2.circle(bgr, (tx,ty), 10, dot_color, 2)
            _cv2.arrowedLine(bgr, (cx0,cy0), (tx,ty), (255,100,0), 1, tipLength=0.15)
        bbox = self.vision.last_bbox
        if bbox is not None and self.vision.is_tracking:
            bx,by,bw2,bh2 = [int(v) for v in bbox]
            box_color = (0,140,255) if range_gated else (0,255,255)
            _cv2.rectangle(bgr, (bx,by), (bx+bw2,by+bh2), box_color, 2)
        state_label = self.vision.state
        if self._kalman_grace > 0 and not self.vision.is_tracking:
            state_label += f" (grace {self._kalman_grace})"
        _cv2.putText(bgr, f"state: {state_label}", (10,25),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        pan, tilt = self.turret.position
        _cv2.putText(bgr, f"pan={pan:5.1f}  tilt={tilt:5.1f}", (10,55),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        if self.lidar is not None:
            reading = self.lidar.get()
            if reading is not None:
                dist_cm, strength = reading
                gate_label = " [GATED]" if range_gated else ""
                color = (0,140,255) if range_gated else (180,255,180)
                _cv2.putText(bgr, f"{dist_cm} cm  str={strength}{gate_label}",
                             (10,80), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        ok, jpg = _cv2.imencode(".jpg", bgr, [_cv2.IMWRITE_JPEG_QUALITY, self._STREAM_QUALITY])
        if ok:
            with self._stream_lock:
                self._stream_jpg = jpg.tobytes()

    def start_mjpeg_server(self, port=8080):
        tracker_self = self
        class StreamHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_sent = None
                while True:
                    with tracker_self._stream_lock:
                        jpg = tracker_self._stream_jpg
                    if jpg is None or jpg is last_sent:
                        time.sleep(0.025)
                        continue
                    last_sent = jpg
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    time.sleep(0.05)
            def log_message(self, *_): pass
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        srv = socketserver.ThreadingTCPServer(("0.0.0.0", port), StreamHandler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"MJPEG stream: http://<pi>:{port}/")

    def run(self, duration_sec=None):
        start = time.time()
        try:
            while True:
                self.step()
                if duration_sec and (time.time() - start) > duration_sec:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        if self.timer is not None:
            self.timer.close()
        self.vision.close()
        self.turret.center()
        if self.lidar is not None:
            self.lidar.close()


if __name__ == "__main__":
    enable_stream = os.environ.get("STREAM", "1") != "0"
    duration = os.environ.get("DURATION")
    duration = float(duration) if duration else None
    t = Tracker(enable_stream=enable_stream)
    if enable_stream:
        t.start_mjpeg_server(8080)
    t.run(duration_sec=duration)
