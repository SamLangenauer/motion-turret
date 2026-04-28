# tracker.py — closed-loop P-controller that points the turret at detected motion.

import time
import config
from servos import Turret
from vision import MotionTracker


class Tracker:
    def __init__(self):
        self.turret = Turret()
        self.vision = MotionTracker(mode="framediff")

        self._frame_cx = config.FRAME_WIDTH // 2
        self._frame_cy = config.FRAME_HEIGHT // 2

        # Lock state
        self._locked = False
        self._lock_frames = 0
        self._LOCK_ACQUIRE = 2     # frames of consistent detection to acquire lock
        self._LOCK_RELEASE = 35    # frames of no detection to release lock

        # Self-motion suppression: the camera is mounted on the moving gimbal,
        # so every commanded servo move shows up as scene motion in the next
        # few frames. After a nudge, ignore detections until the platform
        # settles. Tune _SETTLE_FRAMES: lower = more responsive, higher = less
        # self-feedback.
        self._SETTLE_FRAMES = 4
        self._settle = 0
        self._warmup_frames = 30
        self._lost_frames = 0

    def step(self):
        if self._warmup_frames > 0:
            self.vision.detect()
            self._warmup_frames -= 1
            return None

        # Burn frames after a commanded move so we don't react to our own motion
        if self._settle > 0:
            self.vision.detect()
            self._settle -= 1
            if self._settle == 0:
                self.vision.reset_background()
            return None

        _, target, _ = self.vision.detect()
        if target is None:
            if self._locked:
                self._lock_frames -= 1
                if self._lock_frames <= 0:
                    self._locked = False
                    self._lock_frames = 0
            self._lost_frames += 1
            if self._lost_frames > config.LOST_LOCK_FRAMES_TO_RECENTER:
                self._recenter_step()
            return None

        self._lost_frames = 0

        cx, cy, area = target
        if not self._locked:
            self._lock_frames += 1
            if self._lock_frames >= self._LOCK_ACQUIRE:
                self._locked = True
                self._lock_frames = self._LOCK_RELEASE
            else:
                return target

        err_x = cx - self._frame_cx
        err_y = cy - self._frame_cy

        if abs(err_x) < config.DEADZONE_PIXELS and \
           abs(err_y) < config.DEADZONE_PIXELS:
            return target

        dpan  =  config.KP_PAN  * err_x
        dtilt = -config.KP_TILT * err_y

        # Slew rate limit
        dpan  = max(-config.MAX_NUDGE_DEG, min(config.MAX_NUDGE_DEG, dpan))
        dtilt = max(-config.MAX_NUDGE_DEG, min(config.MAX_NUDGE_DEG, dtilt))

        self.turret.nudge(dpan, dtilt)
        # Larger moves need more physical settle time; small ones almost none.
        max_step = max(abs(dpan), abs(dtilt))
        self._settle = max(3, int(round(max_step / 2.0)))   # 3..5 frames
        return target

    def _recenter_step(self):
        """Slowly slew toward configured center. Called when lock has been
        lost long enough to assume the target left the frame."""
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
            self._settle = self._SETTLE_FRAMES   # same suppression as a tracking move

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
