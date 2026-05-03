# lidar.py — TF-Luna UART driver + non-blocking threaded reader.
#
# Two public classes:
#
#   TFLuna          — raw synchronous reader (one read() call per sample).
#                     Fine for diagnostics; blocks up to serial timeout on
#                     each call, so don't call it from the main tracking loop.
#
#   LidarReader     — thin thread wrapper around TFLuna.  Spawns a daemon
#                     thread that reads continuously and publishes the latest
#                     (distance_cm, strength) pair to a shared slot.  The
#                     main loop calls .get() which returns instantly.
#
# Usage in tracker.py:
#     from lidar import LidarReader
#     lidar = LidarReader()          # starts background thread
#     ...
#     reading = lidar.get()          # (dist_cm, strength) or None
#     if reading:
#         dist_cm, _ = reading

import threading
import time
import serial
import config


class TFLuna:
    """Minimal TF-Luna reader. Call read() to get distance in cm, or None."""

    FRAME_HEADER = b"\x59\x59"
    FRAME_SIZE = 9

    def __init__(self):
        self.ser = serial.Serial(
            config.LIDAR_SERIAL_PORT,
            config.LIDAR_BAUD,
            timeout=0.1,
        )

    def read(self):
        """
        Read one distance sample.
        Returns: (distance_cm, strength) or None if no valid frame.
        """
        # Hunt for frame header
        b1 = self.ser.read(1)
        if b1 != b"\x59":
            return None
        b2 = self.ser.read(1)
        if b2 != b"\x59":
            return None

        rest = self.ser.read(self.FRAME_SIZE - 2)
        if len(rest) != self.FRAME_SIZE - 2:
            return None

        frame = b1 + b2 + rest

        # Checksum: sum of first 8 bytes, low byte only
        if sum(frame[:8]) & 0xFF != frame[8]:
            return None

        distance = frame[2] | (frame[3] << 8)      # cm
        strength = frame[4] | (frame[5] << 8)

        if distance <= 0 or distance > config.LIDAR_MAX_RANGE_CM:
            return None
        return distance, strength

    def close(self):
        self.ser.close()


class LidarReader:
    """
    Non-blocking wrapper around TFLuna.

    A daemon thread reads from the sensor at full speed (~100 Hz) and keeps
    the freshest valid reading in a shared slot.  The main tracking loop
    calls .get() which returns that slot instantly — no serial I/O, no
    blocking, no added latency to the control loop.

    If the sensor is absent or produces no valid frames the slot stays None
    and the tracker degrades gracefully (no range gating).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None          # (distance_cm, strength) | None
        self._running = False
        self._thread = None
        self._luna = None

        try:
            self._luna = TFLuna()
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print("[lidar] TF-Luna reader started")
        except Exception as exc:
            print(f"[lidar] WARNING: could not open TF-Luna ({exc}). "
                  "Range gating disabled.")

    # ---- public API ----

    def get(self):
        """Return the most recent (distance_cm, strength) tuple, or None."""
        with self._lock:
            return self._latest

    @property
    def available(self):
        """True if the sensor opened successfully."""
        return self._luna is not None

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._luna is not None:
            self._luna.close()

    # ---- background thread ----

    def _loop(self):
        consecutive_failures = 0
        while self._running:
            try:
                reading = self._luna.read()
                if reading is not None:
                    consecutive_failures = 0
                    with self._lock:
                        self._latest = reading
                else:
                    consecutive_failures += 1
                    # After 50 consecutive misses (~0.5 s) clear the slot so the
                    # tracker doesn't keep trusting a stale distance.
                    if consecutive_failures >= 50:
                        with self._lock:
                            self._latest = None
            except Exception:
                # Serial glitch — clear slot, back off briefly, keep going.
                with self._lock:
                    self._latest = None
                time.sleep(0.05)
