# servos.py — thin wrapper around ServoKit with safety clamping

from adafruit_servokit import ServoKit
import config


class Turret:
    """Two-axis turret: pan + tilt. Clamps angles to configured safe range."""

    def __init__(self):
        self.kit = ServoKit(channels=16)

        # Configure pulse width range so ServoKit maps 0-180° correctly
        for ch in (config.PAN_CHANNEL, config.TILT_CHANNEL):
            self.kit.servo[ch].set_pulse_width_range(
                config.SERVO_MIN_PULSE, config.SERVO_MAX_PULSE
            )

        # Internal state — updated by set_pan/set_tilt
        self._pan = config.PAN_CENTER
        self._tilt = config.TILT_CENTER
        self.center()

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))

    def set_pan(self, angle):
        angle = self._clamp(angle, config.PAN_MIN_ANGLE, config.PAN_MAX_ANGLE)
        self._pan = angle
        self.kit.servo[config.PAN_CHANNEL].angle = angle

    def set_tilt(self, angle):
        angle = self._clamp(angle, config.TILT_MIN_ANGLE, config.TILT_MAX_ANGLE)
        self._tilt = angle
        self.kit.servo[config.TILT_CHANNEL].angle = angle

    def nudge(self, dpan, dtilt):
        """Incremental move — used by the tracker's control loop."""
        self.set_pan(self._pan + dpan)
        self.set_tilt(self._tilt + dtilt)

    def center(self):
        self.set_pan(config.PAN_CENTER)
        self.set_tilt(config.TILT_CENTER)

    @property
    def position(self):
        return (self._pan, self._tilt)
