# tracker.py — closed-loop P-controller that points the turret at detected motion.

import time
import config
from servos import Turret
from vision import MotionTracker


class Tracker:
    """
    Each tick: grab a frame, find target, compute pixel error from center,
    translate to servo-angle correction via proportional gain.
    """

    def __init__(self):
        self.turret = Turret()
        self.vision = MotionTracker()

        self._frame_cx = config.FRAME_WIDTH // 2
        self._frame_cy = config.FRAME_HEIGHT // 2

    def step(self):
        """Run one detect + correct cycle. Returns the target or None."""
        _, target, _ = self.vision.detect()
        if target is None:
            return None

        cx, cy, _ = target
        err_x = cx - self._frame_cx   # positive = target right of center
        err_y = cy - self._frame_cy   # positive = target below center

        # Deadzone prevents oscillation when the target is already centered
        if abs(err_x) < config.DEADZONE_PIXELS and \
           abs(err_y) < config.DEADZONE_PIXELS:
            return target

        # Note sign conventions depend on how your servos are physically mounted;
        # you may need to flip one or both signs during Phase 4 calibration.
        dpan = -config.KP_PAN * err_x    # target right → pan right (decrease? increase?)
        dtilt = config.KP_TILT * err_y   # target below → tilt down

        self.turret.nudge(dpan, dtilt)
        return target

    def run(self, duration_sec=None):
        """Continuous tracking loop. Ctrl-C to stop."""
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
    Tracker().run()
