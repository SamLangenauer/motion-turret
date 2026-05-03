#!/usr/bin/env python3
# laptop_vision.py — laptop-side CV loop for the hybrid Pi/laptop architecture.
#
# Receives JPEG frames from the Pi via UDP, runs MobileNet-SSD + CSRT,
# sends (cx, cy, area, state) coord packets back.
#
# Usage:
#   python3 laptop_vision.py                    # uses config defaults
#   python3 laptop_vision.py --pi 192.168.1.42  # override Pi IP
#   python3 laptop_vision.py --preview          # show OpenCV debug window
#
# Requirements (laptop):
#   pip install opencv-contrib-python numpy
#   (No picamera2 / adafruit needed — pure CV only)

import argparse
import socket
import struct
import time
import threading
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Inline config — mirror the values from Pi's config.py.
# Change these to match your setup if not using the defaults.
# ---------------------------------------------------------------------------
DEFAULT_PI_IP        = "192.168.1.100"   # Pi's IP on your LAN
FRAME_RX_PORT        = 5005              # laptop listens for frames here
COORD_TX_PORT        = 5006              # laptop sends coords to this Pi port
MODEL_PROTOTXT       = "MobileNetSSD_deploy.prototxt"
MODEL_WEIGHTS        = "MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
FRAME_WIDTH          = 680
FRAME_HEIGHT         = 480
AIM_VERTICAL_BIAS    = 0.45

# CSRT bbox crop — must match Pi's tight upper-body crop
TRACK_W_RATIO = 0.6
TRACK_H_RATIO = 0.4
TRACK_Y_RATIO = 0.1
MAX_TRACK_DIM = 150

# Drop frames older than this before processing — keeps us on the latest view.
FRAME_QUEUE_MAX = 2

# ---------------------------------------------------------------------------
# Packet format (must match net_protocol.py on the Pi)
# ---------------------------------------------------------------------------
_FRAME_HDR  = struct.Struct("!IHH")     # frame_id, w, h
_COORD      = struct.Struct("!IBHHIf")  # frame_id, state, cx, cy, area, conf
STATE_SEARCHING, STATE_TRACKING, STATE_LOST = 0, 1, 2


def _make_tracker():
    try:
        return cv2.legacy.TrackerCSRT_create()
    except (AttributeError, cv2.error):
        pass
    try:
        return cv2.TrackerCSRT_create()
    except (AttributeError, cv2.error):
        return cv2.legacy.TrackerMOSSE_create()


# ---------------------------------------------------------------------------
# Frame receiver thread — fills a shared slot with the latest JPEG
# ---------------------------------------------------------------------------
class FrameReceiver(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(1.0)
        self._lock = threading.Lock()
        self._latest = None     # (frame_id, jpeg_bytes)
        self.running = True

    def run(self):
        while self.running:
            try:
                data, _ = self._sock.recvfrom(131072)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) <= _FRAME_HDR.size:
                continue
            frame_id, w, h = _FRAME_HDR.unpack_from(data)
            jpeg = data[_FRAME_HDR.size:]
            with self._lock:
                self._latest = (frame_id, jpeg)

    def get(self):
        with self._lock:
            val = self._latest
            self._latest = None
            return val

    def close(self):
        self.running = False
        self._sock.close()


# ---------------------------------------------------------------------------
# Coord sender — fire-and-forget UDP
# ---------------------------------------------------------------------------
class CoordSender:
    def __init__(self, pi_ip, port):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dest = (pi_ip, port)

    def send(self, frame_id, state, cx, cy, area, confidence):
        pkt = _COORD.pack(frame_id, state, cx, cy, area, confidence)
        try:
            self._sock.sendto(pkt, self._dest)
        except OSError:
            pass

    def close(self):
        self._sock.close()


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------
def run(pi_ip: str, show_preview: bool):
    print(f"[laptop_vision] Pi IP: {pi_ip}")
    print(f"[laptop_vision] Listening for frames on UDP :{FRAME_RX_PORT}")
    print(f"[laptop_vision] Sending coords to {pi_ip}:{COORD_TX_PORT}")

    # Load DNN
    net = cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_WEIGHTS)
    print("[laptop_vision] MobileNet-SSD loaded")

    receiver = FrameReceiver(FRAME_RX_PORT)
    receiver.start()
    sender   = CoordSender(pi_ip, COORD_TX_PORT)

    tracker      = None
    tracker_ok   = False
    relock_count = 0
    DNN_RELOCK   = 30   # re-run DNN every N tracked frames
    MAX_LOST     = 20   # CSRT failures before declaring lost
    lost_streak  = 0

    state     = STATE_SEARCHING
    last_send = time.monotonic()

    print("[laptop_vision] Waiting for frames from Pi...")

    try:
        while True:
            item = receiver.get()
            if item is None:
                time.sleep(0.005)
                # Send a searching heartbeat every 100 ms so Pi doesn't stale out
                if (time.monotonic() - last_send) * 1000 > 100:
                    sender.send(0, STATE_SEARCHING, 0, 0, 0, 0.0)
                    last_send = time.monotonic()
                continue

            frame_id, jpeg = item

            # Decode JPEG
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 0)

            cx, cy, area, conf = 0, 0, 0, 0.0
            target_found = False

            # ---- CSRT update ----
            if state == STATE_TRACKING and tracker is not None:
                relock_count += 1
                ok, bbox = tracker.update(gray)

                bx, by, bw, bh = [int(v) for v in bbox] if ok else (0,0,0,0)
                bbox_bad = (
                    not ok or bw < 20 or bh < 20 or
                    bw > w * 0.9 or bh > h * 0.9 or
                    bx < 0 or by < 0 or bx + bw > w or by + bh > h
                )

                if bbox_bad:
                    lost_streak += 1
                    if lost_streak >= MAX_LOST:
                        state   = STATE_LOST
                        tracker = None
                        lost_streak = 0
                else:
                    lost_streak = 0
                    cx   = int(bx + bw / 2)
                    cy   = int(by + bh * AIM_VERTICAL_BIAS)
                    area = bw * bh
                    conf = 1.0
                    target_found = True

                    if show_preview:
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0,255,255), 2)

                # Periodic DNN re-lock
                if relock_count >= DNN_RELOCK:
                    relock_count = 0
                    _run_dnn(net, frame, gray, tracker,
                             w, h, sender, frame_id)  # async would be ideal; sync ok on laptop

            # ---- DNN acquisition ----
            if state in (STATE_SEARCHING, STATE_LOST) or \
               (state == STATE_TRACKING and relock_count == 0 and not target_found):

                bbox_dnn, conf_dnn = _detect_human(net, frame)
                if bbox_dnn is not None:
                    x, y, bw, bh = bbox_dnn
                    tw = min(int(bw * TRACK_W_RATIO), MAX_TRACK_DIM)
                    th = min(int(bh * TRACK_H_RATIO), MAX_TRACK_DIM)
                    tx = max(0, min(x + bw//2 - tw//2, w - tw))
                    ty = max(0, min(y + int(bh * TRACK_Y_RATIO), h - th))
                    tight = (tx, ty, tw, th)

                    tracker = _make_tracker()
                    tracker.init(gray, tight)
                    state       = STATE_TRACKING
                    lost_streak = 0
                    relock_count = 0

                    cx   = int(tx + tw / 2)
                    cy   = int(ty + th * AIM_VERTICAL_BIAS)
                    area = tw * th
                    conf = conf_dnn
                    target_found = True

                    if show_preview:
                        cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (0,255,0), 2)

            # ---- Send coord packet ----
            send_state = STATE_TRACKING if target_found else \
                         STATE_LOST if state == STATE_LOST else STATE_SEARCHING
            sender.send(frame_id, send_state, cx, cy, area, conf)
            last_send = time.monotonic()

            # ---- Preview ----
            if show_preview:
                label_color = {
                    STATE_TRACKING:  (0,255,0),
                    STATE_SEARCHING: (200,200,200),
                    STATE_LOST:      (0,0,255),
                }.get(send_state, (255,255,255))
                state_labels = {STATE_TRACKING: "tracking",
                                STATE_SEARCHING: "searching",
                                STATE_LOST: "lost"}
                cv2.putText(frame, f"state: {state_labels[send_state]}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                if target_found:
                    cv2.circle(frame, (cx, cy), 8, (0,0,255), -1)
                cv2.imshow("laptop_vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        sender.close()
        if show_preview:
            cv2.destroyAllWindows()
        print("[laptop_vision] stopped")


def _detect_human(net, frame):
    """Run MobileNet-SSD, return (bbox, confidence) for best person or None."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    dets = net.forward()

    best_box, best_conf, max_area = None, 0.0, 0
    for i in np.arange(0, dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < CONFIDENCE_THRESHOLD:
            continue
        if int(dets[0, 0, i, 1]) != 15:   # class 15 = person
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        sx, sy, ex, ey = box.astype("int")
        sx, sy = max(0, sx), max(0, sy)
        ex, ey = min(w-1, ex), min(h-1, ey)
        bw, bh = ex - sx, ey - sy
        a = bw * bh
        if a > max_area:
            max_area = a
            best_box = (sx, sy, bw, bh)
            best_conf = conf

    return best_box, best_conf


def _run_dnn(net, frame, gray, tracker, w, h, sender, frame_id):
    """Inline DNN re-lock (laptop has plenty of CPU; can thread if needed)."""
    bbox_dnn, conf_dnn = _detect_human(net, frame)
    if bbox_dnn is None:
        return
    x, y, bw, bh = bbox_dnn
    tw = min(int(bw * TRACK_W_RATIO), MAX_TRACK_DIM)
    th = min(int(bh * TRACK_H_RATIO), MAX_TRACK_DIM)
    tx = max(0, min(x + bw//2 - tw//2, w - tw))
    ty = max(0, min(y + int(bh * TRACK_Y_RATIO), h - th))
    # Re-init the tracker in-place with the corrected bbox
    tracker.init(gray, (tx, ty, tw, th))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi",      default=DEFAULT_PI_IP,
                    help="Pi's IP address on your LAN")
    ap.add_argument("--preview", action="store_true",
                    help="Show OpenCV debug window")
    args = ap.parse_args()
    run(pi_ip=args.pi, show_preview=args.preview)
