# lidar.py — TF-Luna UART driver.

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
