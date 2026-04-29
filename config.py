#http://100.70.102.124:8080/# config.py — tuning constants and hardware assignments

# --- Servo channels on PCA9685 ---
PAN_CHANNEL = 0
TILT_CHANNEL = 1

# --- Servo calibration ---
# MS18s are nominally 0-180°. Real travel varies per unit;
# refine these numbers after the bench test in Phase 2.
PAN_MIN_ANGLE = 0
PAN_MAX_ANGLE = 140
PAN_CENTER = 90

TILT_MIN_ANGLE = 58     # don't let tilt crash into the base
TILT_MAX_ANGLE = 177
TILT_CENTER = 130

# PCA9685 pulse width range in microseconds.
# 500-2500 is the standard range for 9g-class hobby servos.
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

# --- Camera ---
FRAME_WIDTH = 680
FRAME_HEIGHT = 480
FRAME_RATE = 30

# --- Motion detection ---
MOTION_THRESHOLD = 25       # pixel-difference threshold (0-255)
MIN_CONTOUR_AREA = 750      # ignore blobs smaller than this (pixels)

# Aim-point vertical bias: 0.5 = bbox center, >0.5 = aim lower (toward feet),
# <0.5 = aim higher (toward head). 0.6 approximates human center-of-mass when
# the motion blob captures the full silhouette.
AIM_VERTICAL_BIAS = 0.45

# --- Tracking control (Phase 4) ---
KP_PAN = 0.035             # proportional gain, pan axis
KP_TILT = 0.035            # proportional gain, tilt axis
DEADZONE_PIXELS = 35       # don't correct if target is within N px of center

# --- TF-Luna (Phase 5) ---
LIDAR_SERIAL_PORT = "/dev/serial0"
LIDAR_BAUD = 115200
LIDAR_MAX_RANGE_CM = 300   # ignore objects further than this

# Loss-of-lock recovery
LOST_LOCK_FRAMES_TO_RECENTER = 45   # ~1.5 seconds at 30fps
RECENTER_STEP_DEG = 1.5             # how aggressively to slew home (deg/frame)

# Slew rate limit: maximum servo correction per control cycle (degrees).
# Caps how aggressively the controller can chase a target. Smaller = smoother
# tracking, slower acquisition; larger = snappier, more overshoot risk.
MAX_NUDGE_DEG = 5
