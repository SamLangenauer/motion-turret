# tracker.py — closed-loop P-controller driving the turret to follow a tracked target.
#
# Phase 4.5+ instrumentation:
#   - StageTimer logs per-stage latency to latency.csv with periodic summaries.
#   - Stream encoding toggled via STREAM env var (STREAM=0 disables).
#   - DURATION env var caps run length in seconds.
#
# Phase 5 (Kalman):
#   - TargetKalman replaces the EMA smoother that was in vision.py.
#   - Every frame with a valid detection feeds kalman.update(); the filter
#     returns a smoothed position + velocity estimate.
#   - The control loop aims at kalman.lead(KALMAN_LEAD_MS/1000) — the
#     predicted position N ms from now — compensating for capture + servo lag.
#   - During settle frames, kalman.predict() dead-reckons the target's position
#     so the filter stays warm and velocity doesn't decay to zero.
#   - kalman.reset() is called exactly when the vision state transitions to LOST,
#     so a fresh acquisition gets a fresh filter with no stale velocity.
#   - vx / vy (px/s) are logged to latency.csv for analysis.

import threading
import http.server
import socketserver
import time
import os
import cv2 as _cv2

import config
from servos import Turret
from vision import MotionTracker
from instrument import StageTimer
from kalman import TargetKalman


class Tracker:
    def __init__(self, enable_stream=True, latency_csv="latency.csv"):
        self.turret = Turret()
        self.vision = MotionTracker(mode="tracker")
        self.kalman = TargetKalman(
            frame_width=config.FRAME_WIDTH,
            frame_height=config.FRAME_HEIGHT,
            meas_noise_px=config.KALMAN_MEAS_NOISE_PX,
            proc_noise=config.KALMAN_PROC_NOISE,
        )

        self._frame_cx = config.FRAME_WIDTH // 2
        self._frame_cy = config.FRAME_HEIGHT // 2
        
        self._sync_counter = 0
        
        self._sync_thread = None
        self._sync_result = None
        self._sync_lock = threading.Lock()

        self._SETTLE_FRAMES = 2
        self._settle = 0
        self._warmup_frames = 30
        self._lost_frames = 0
        self._track_streak = 0

        self._patrol_direction = -1  # 1 for right, -1 for left
        self._patrol_speed = 5       # Degrees per frame to pan
        self._patrol_max = config.PAN_MAX_ANGLE
        self._patrol_min = config.PAN_MIN_ANGLE

        self._last_frame = None
        self._last_target = None        # Kalman-smoothed aim point for stream overlay
        self._prev_vision_state = None  # track state transitions for kalman.reset()

        # Stream encoding state
        self._enable_stream = enable_stream
        self._stream_jpg = None
        self._stream_lock = threading.Lock()
        self._stream_frame_counter = 0
        self._STREAM_EVERY_N = 3
        self._STREAM_QUALITY = 40

        # Latency instrumentation. Pass latency_csv=None to disable.
        self.timer = StageTimer(latency_csv, summary_every=60) if latency_csv else None
        self._prev_err_x = 0.0
        self._prev_err_y = 0.0

    # ---- main control loop ----
    def _run_dnn_sync(self, frame_copy):
        """Runs the heavy neural network on a background thread."""
        bbox, area = self.vision._detect_human(frame_copy)
        
        # Safely hand the result back to the main thread
        with self._sync_lock:
            self._sync_result = (bbox, area)

    def step(self):
        if self.timer is not None:
            self.timer.frame_begin()

        # Warmup: let camera/AGC settle; don't drive servos.
        if self._warmup_frames > 0:
            self.vision.detect(timer=self.timer)
            self._warmup_frames -= 1
            if self.timer is not None:
                self.timer.note("phase", "warmup")
                self.timer.note("state", self.vision.state)
                self.timer.note("sensor_age_ms", f"{self.vision.sensor_age_ms():.2f}")
                self.timer.frame_end()
            return None

        # Settle: servo command was just issued; wait for gimbal to stop so
        # framediff isn't seeing ego-motion.  Still update the Kalman filter
        # (predict only) so it stays warm and velocity doesn't decay to zero.
        if self._settle > 0:
            frame, target, _ = self.vision.detect(timer=self.timer)
            self._last_frame = frame
            self._settle -= 1

            curr_state = self.vision.state
            if target is not None:
                # MOSSE managed a valid update through the motion — use it.
                raw_cx, raw_cy, area = target
                kx, ky, vx, vy = self.kalman.update(raw_cx, raw_cy)
                self._last_target = (kx, ky, area)
            else:
                # Dead-reckon so velocity stays alive.
                pred = self.kalman.predict()
                if pred is not None:
                    kx, ky, vx, vy = pred
                    self._last_target = (kx, ky, 0)

            self._handle_state_transition(curr_state)

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

        # ---- Active perception + control ----

        # --- THE FIX: Asynchronous DNN Course-Correction ---
        self._sync_counter += 1
        if self.vision.is_tracking and self._sync_counter > 30 and self._last_frame is not None:
            # Only start a new thread if the old one is finished
            if self._sync_thread is None or not self._sync_thread.is_alive():
                # MUST copy the frame so the main thread doesn't overwrite it while the DNN is reading
                frame_copy = self._last_frame.copy() 
                self._sync_thread = threading.Thread(target=self._run_dnn_sync, args=(frame_copy,))
                self._sync_thread.daemon = True
                self._sync_thread.start()
            self._sync_counter = 0 # Reset timer

        # Check if the background thread has finished and handed us a new bounding box
        with self._sync_lock:
            if self._sync_result is not None:
                bbox, area = self._sync_result
                self._sync_result = None # Clear the result so we don't apply it twice
                
                if bbox is not None and area > 2000:
                    if self.timer is not None: self.timer.mark("kcf_init")
                    # Re-initialize MOSSE with the perfect upper-body crop
                    x, y, w, h = bbox
                    track_w = min(int(w * 0.6), 150)
                    track_h = min(int(h * 0.4), 150)
                    track_x = max(0, min(x + (w // 2) - (track_w // 2), config.FRAME_WIDTH - track_w))
                    track_y = max(0, min(y + int(h * 0.1), config.FRAME_HEIGHT - track_h))
                    
                    self.vision._csrt = _cv2.legacy.TrackerMOSSE_create()
                    gray_frame = _cv2.cvtColor(self._last_frame, _cv2.COLOR_BGR2GRAY)
                    self.vision._csrt.init(gray_frame, (int(track_x), int(track_y), int(track_w), int(track_h)))
                    self.kalman.reset()
                    if self.timer is not None: self.timer.lap("kcf_init")

        # Normal vision step...
        frame, target, _ = self.vision.detect(timer=self.timer)
        self._last_frame = frame

        curr_state = self.vision.state
        self._handle_state_transition(curr_state)

        # --- Kalman update / predict ---
        vx, vy = 0.0, 0.0
        aim = None

        if target is not None:
            raw_cx, raw_cy, area = target
            if self.timer is not None: self.timer.mark("kalman")
            kx, ky, vx, vy = self.kalman.update(raw_cx, raw_cy)
            # Lead-compensated aim: where will the target be in KALMAN_LEAD_MS?
            speed = (vx**2 + vy**2) ** 0.5
            if speed > 30 and config.KALMAN_LEAD_MS > 0:
                lead = self.kalman.lead(config.KALMAN_LEAD_MS / 1000.0)
                aim_cx, aim_cy = lead if lead else (kx, ky)
            else:
                aim_cx, aim_cy = kx, ky
            if self.timer is not None: self.timer.lap("kalman")
            aim = (aim_cx, aim_cy, area)
        elif self.vision.is_tracking:
            # MOSSE is in tracking state but returned no target this frame
            # (e.g. brief occlusion). Dead-reckon rather than going blind.
            pred = self.kalman.predict()
            if pred is not None:
                kx, ky, vx, vy = pred
                aim = (kx, ky, 0)

        self._last_target = aim

        # --- Stream encode ---
        if self._enable_stream:
            if self.timer is not None: self.timer.mark("stream")
            self._maybe_encode_stream_frame()
            if self.timer is not None: self.timer.lap("stream")

        # --- Servo control ---
        result = None
        commanded = False

        if not self.vision.is_tracking or aim is None:
            self._lost_frames += 1
            self._track_streak = 0

            # If we've been lost for a bit, start patrolling
            if self._lost_frames > config.LOST_LOCK_FRAMES_TO_RECENTER:
                if self.timer is not None: self.timer.mark("control")
                
                # --- PATROL LOGIC ---
                pan, tilt = self.turret.position
                
                # Slowly pan back and forth
                next_pan = pan + (self._patrol_speed * self._patrol_direction)
                if next_pan > self._patrol_max:
                    self._patrol_direction = -1
                elif next_pan < self._patrol_min:
                    self._patrol_direction = 1
                    
                # Gently guide tilt back to center during patrol
                dtilt = 0
                if tilt < config.TILT_CENTER - 1: dtilt = 1
                elif tilt > config.TILT_CENTER + 1: dtilt = -1

                dpan = (self._patrol_speed * self._patrol_direction)
                
                self.turret.nudge(dpan, dtilt)
                # Give it a tiny settle time so the HOG detector doesn't get a blurry frame
                self._settle = 1 
                
                if self.timer is not None: self.timer.lap("control")
                commanded = True

        else:
            self._lost_frames = 0
            self._track_streak += 1
            cx, cy, _area = aim
            err_x = cx - self._frame_cx
            err_y = cy - self._frame_cy

            if abs(err_x) >= config.DEADZONE_PIXELS or abs(err_y) >= config.DEADZONE_PIXELS:
                
                # Calculate the rate of change (Derivative)
                d_err_x = err_x - self._prev_err_x
                d_err_y = err_y - self._prev_err_y

                # PD Control Math
                dpan  = (config.KP_PAN * err_x) + (config.KD_PAN * d_err_x)
                dtilt = -(config.KP_TILT * err_y) - (config.KD_TILT * d_err_y)
                
                # Save current error for next frame
                self._prev_err_x = err_x
                self._prev_err_y = err_y

                # Clamp to the new, higher max speed
                dpan  = max(-config.MAX_NUDGE_DEG, min(config.MAX_NUDGE_DEG, dpan))
                dtilt = max(-config.MAX_NUDGE_DEG, min(config.MAX_NUDGE_DEG, dtilt))

                if self.timer is not None: self.timer.mark("control")
                self.turret.nudge(dpan, dtilt)
                if self.timer is not None: self.timer.lap("control")
                commanded = True

                max_step = max(abs(dpan), abs(dtilt))
                self._settle = max(self._SETTLE_FRAMES, int(round(max_step / 3.0)))

                if self.timer is not None:
                    self.timer.note("dpan", f"{dpan:.2f}")
                    self.timer.note("dtilt", f"{dtilt:.2f}")
                    self.timer.note("settle_set", self._settle)

            result = aim

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
    def _handle_state_transition(self, curr_state: str):
        """Reset the Kalman filter and vision background when lock is lost."""
        if (curr_state in (MotionTracker.STATE_LOST, MotionTracker.STATE_SEARCHING)
                and self._prev_vision_state == MotionTracker.STATE_TRACKING):
            self.kalman.reset()
            self.vision.reset_background()
            
            # THE FIX: Force a settle period to blind the frame-differencer
            # while the physical gimbal mount stops vibrating from the sudden halt.
            self._settle = max(self._settle, self._SETTLE_FRAMES * 5) 
            
        self._prev_vision_state = curr_state

    def _recenter_step(self):
        pan, tilt = self.turret.position
        step = config.RECENTER_STEP_DEG

        dpan = 0.0
        if pan < config.PAN_CENTER - step:
            dpan = step
        elif pan > config.PAN_CENTER + step:
            dpan = -step

        dtilt = 0.0
        if tilt < config.TILT_CENTER - step:
            dtilt = step
        elif tilt > config.TILT_CENTER + step:
            dtilt = -step

        if dpan or dtilt:
            self.turret.nudge(dpan, dtilt)
            self._settle = self._SETTLE_FRAMES

    # ---- stream frame encoder (runs inline in main thread) ----

    def _maybe_encode_stream_frame(self):
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
            _cv2.circle(bgr, (tx, ty), 10, (0,0,255), 2)
            _cv2.arrowedLine(bgr, (cx0, cy0), (tx, ty),
                             (255,100,0), 1, tipLength=0.15)

        bbox = self.vision.last_bbox
        if bbox is not None and self.vision.is_tracking:
            bx, by, bw, bh = [int(v) for v in bbox]
            _cv2.rectangle(bgr, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)

        _cv2.putText(bgr, f"state: {self.vision.state}",
                     (10, 25), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        pan, tilt = self.turret.position
        _cv2.putText(bgr, f"pan={pan:5.1f}  tilt={tilt:5.1f}",
                     (10, 55), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        ok, jpg = _cv2.imencode(".jpg", bgr,
                                [_cv2.IMWRITE_JPEG_QUALITY, self._STREAM_QUALITY])
        if ok:
            with self._stream_lock:
                self._stream_jpg = jpg.tobytes()

    # ---- HTTP server ----

    def start_mjpeg_server(self, port=8080):
        tracker_self = self

        class StreamHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
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

            def log_message(self, *_):
                pass

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        srv = socketserver.ThreadingTCPServer(("0.0.0.0", port), StreamHandler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"MJPEG stream: http://<pi>:{port}/")

    # ---- run loop ----

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


if __name__ == "__main__":
    enable_stream = os.environ.get("STREAM", "1") != "0"
    duration = os.environ.get("DURATION")
    duration = float(duration) if duration else None

    t = Tracker(enable_stream=enable_stream)
    if enable_stream:
        t.start_mjpeg_server(8080)
    t.run(duration_sec=duration)
