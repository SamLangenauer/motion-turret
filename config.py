# config.py — tuning constants and hardware assignments

# --- Servo channels on PCA9685 ---
PAN_CHANNEL = 0
TILT_CHANNEL = 1

# --- Servo calibration ---
PAN_MIN_ANGLE = 0
PAN_MAX_ANGLE = 140
PAN_CENTER = 90

TILT_MIN_ANGLE = 58
TILT_MAX_ANGLE = 177
TILT_CENTER = 115

# PCA9685 pulse width range in microseconds.
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

# --- Camera ---
FRAME_WIDTH = 680
FRAME_HEIGHT = 480

# --- Motion detection ---
MOTION_THRESHOLD = 25
MIN_CONTOUR_AREA = 750

AIM_VERTICAL_BIAS = 0.45

# --- Tracking control ---
KP_PAN = 0.035
KP_TILT = 0.035
KD_PAN = 0.025
KD_TILT = 0.025
DEADZONE_PIXELS = 22.5

# --- Kalman filter (Phase 5) ---
# Measurement noise: how noisy are raw detections (pixels std dev).
#   Larger → filter trusts its own prediction more → smoother, slightly laggier.
#   Smaller → filter trusts raw detections more → snappier, noisier crosshair.
KALMAN_MEAS_NOISE_PX = 10.0

# Process noise: how fast can the target's velocity change (pixels/s²).
#   Larger → filter adapts quickly to direction changes.
#   Smaller → smoother velocity estimate, slower to react to sharp cuts.
KALMAN_PROC_NOISE = 60.0

# Lead compensation: aim this many milliseconds ahead of the current state
# estimate to pre-compensate for capture latency + servo response time.
# Start at 20 ms (roughly capture p50 + one servo step).
# Increase if the laser consistently trails a moving target.
# Decrease if aiming overshoots in front of the target.
KALMAN_LEAD_MS = 0.0

# --- TF-Luna (Phase 6) ---
LIDAR_SERIAL_PORT = "/dev/serial0"
LIDAR_BAUD = 115200
 
# Set False to disable LiDAR entirely (e.g. sensor not connected).
# When False the tracker behaves exactly as Phase 5 — no range gating at all.
LIDAR_ENABLED = True
 
# Reject targets beyond this distance.  Tune to your room / use-case.
# TF-Luna rated range: 0.2 – 8 m (indoor, white target).
LIDAR_MAX_RANGE_CM = 300
 
# Reject targets closer than this (suppresses close-range noise / ground returns).
# Anything under ~20 cm is usually sensor noise or a stray reflection.
LIDAR_MIN_RANGE_CM = 20

# Loss-of-lock recovery
LOST_LOCK_FRAMES_TO_RECENTER = 45
RECENTER_STEP_DEG = 1

# Slew rate limit
MAX_NUDGE_DEG = 15

REMOTE_VISION = True
LAPTOP_IP = "100.69.0.24"
# UDP ports.  Pi listens on COORD_RX_PORT; laptop listens on FRAME_TX_PORT.
FRAME_TX_PORT = 5005    # Pi → Laptop (JPEG frames)
COORD_RX_PORT = 5006    # Laptop → Pi (coord packets)
 
# JPEG quality for frame transmission.  Lower = smaller packet, more
# compression artefacts.  40 is a good starting point; drop to 30 if
# you're seeing UDP drops (check with --preview on the laptop).
REMOTE_FRAME_QUALITY = 40
