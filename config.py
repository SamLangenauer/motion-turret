# config.py — tuning constants and hardware assignments

# --- Servo channels on PCA9685 ---
PAN_CHANNEL = 0
TILT_CHANNEL = 1

# --- Servo calibration ---
# MS18s are nominally 0-180°. Real travel varies per unit;
# refine these numbers after the bench test in Phase 2.
PAN_MIN_ANGLE = 0
PAN_MAX_ANGLE = 180
PAN_CENTER = 90

TILT_MIN_ANGLE = 30     # don't let tilt crash into the base
TILT_MAX_ANGLE = 150
TILT_CENTER = 90

# PCA9685 pulse width range in microseconds.
# 500-2500 is the standard range for 9g-class hobby servos.
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

# --- Camera ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30

# --- Motion detection ---
MOTION_THRESHOLD = 25       # pixel-difference threshold (0-255)
MIN_CONTOUR_AREA = 500      # ignore blobs smaller than this (pixels)

# --- Tracking control (Phase 4) ---
KP_PAN = 0.03              # proportional gain, pan axis
KP_TILT = 0.03             # proportional gain, tilt axis
DEADZONE_PIXELS = 15       # don't correct if target is within N px of center

# --- TF-Luna (Phase 5) ---
LIDAR_SERIAL_PORT = "/dev/serial0"
LIDAR_BAUD = 115200
LIDAR_MAX_RANGE_CM = 300   # ignore objects further than this
