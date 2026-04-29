# tracker.py — closed-loop P-controller driving the turret to follow a tracked target.

import threading
import http.server
import socketserver
import time
import cv2 as _cv2

import config
from servos import Turret
from vision import MotionTracker


class Tracker:
    def __init__(self):
        self.turret = Turret()
        self.vision = MotionTracker(mode="tracker")

        self._frame_cx = config.FRAME_WIDTH // 2
        self._frame_cy = config.FRAME_HEIGHT // 2

        self._SETTLE_FRAMES = 2
        self._settle = 0
        self._warmup_frames = 30
        self._lost_frames = 0

        self._last_frame = None
        self._last_target = None

        # Stream encoding state — produced by main loop, consumed by HTTP handler
        self._stream_jpg = None
        self._stream_lock = threading.Lock()
        self._stream_frame_counter = 0
        self._STREAM_EVERY_N = 3        # encode 1 of every N frames
        self._STREAM_QUALITY = 20

    # ---- main control loop ----

    def step(self):
        if self._warmup_frames > 0:
            self.vision.detect()
            self._warmup_frames -= 1
            return None

        if self._settle > 0:
            self.vision.detect()
            self._settle -= 1
            return None

        frame, target, _ = self.vision.detect()
        self._last_frame = frame
        self._last_target = target

        # Maintain stream encoding (cheap if no clients connected,
        # because no one will read self._stream_jpg)
        self._maybe_encode_stream_frame()

        if not self.vision.is_tracking or target is None:
            self._lost_frames += 1
            if self._lost_frames > config.LOST_LOCK_FRAMES_TO_RECENTER:
                self._recenter_step()
            return None

        self._lost_frames = 0

        cx, cy, _area = target
        err_x = cx - self._frame_cx
        err_y = cy - self._frame_cy

        if abs(err_x) < config.DEADZONE_PIXELS and \
           abs(err_y) < config.DEADZONE_PIXELS:
            return target

        dpan  =  config.KP_PAN  * err_x
        dtilt = -config.KP_TILT * err_y
        dpan  = max(-config.MAX_NUDGE_DEG, min(config.MAX_NUDGE_DEG, dpan))
        dtilt = max(-config.MAX_NUDGE_DEG, min(config.MAX_NUDGE_DEG, dtilt))

        self.turret.nudge(dpan, dtilt)
        max_step = max(abs(dpan), abs(dtilt))
        self._settle = max(self._SETTLE_FRAMES, int(round(max_step / 3.0)))
        return target

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

    # ---- stream frame encoder (runs in main thread) ----

    def _maybe_encode_stream_frame(self):
        self._stream_frame_counter += 1
        if self._stream_frame_counter % self._STREAM_EVERY_N != 0:
            return
        if self._last_frame is None:
            return

        bgr = _cv2.cvtColor(self._last_frame, _cv2.COLOR_RGB2BGR)
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
                        time.sleep(0.05)
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
        self.vision.close()
        self.turret.center()


if __name__ == "__main__":
    t = Tracker()
    t.start_mjpeg_server(8080)
    t.run()
