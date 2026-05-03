# Motion Tracking Laser Turret

A low-latency, 2-axis pan/tilt turret that detects and tracks humans using a Raspberry Pi camera, pointing a laser at the target. Built around a multi-stage perception pipeline: MobileNet-SSD acquires the target, MOSSE correlation tracking maintains lock at ~80 FPS, a Kalman filter smooths position and estimates velocity, and a PD controller drives the servos.

## Hardware

| Component | Part |
|---|---|
| SBC | Raspberry Pi 4B (Pi OS Bookworm 64-bit) |
| Camera | Arducam IMX219 8MP |
| Servo driver | PCA9685 16-channel PWM (I²C) |
| Servos | 2× GH-S37D 3.7g micro servos |
| Laser | 5V 5mW 650nm red laser diode |
| LiDAR | TF-Luna (driver ready, not yet integrated) |
| Power | Dedicated regulated 5V supply → PCA9685 terminal block |
| Mount | 3D-printed 2-axis pan/tilt gimbal |

## Software Architecture

```
Camera (80 FPS)
    │
    ├── [SEARCHING / LOST] → MobileNet-SSD (full-frame human detection)
    │       └── on detection: initialize MOSSE on tight head/chest crop
    │
    └── [TRACKING] → MOSSE correlation filter (grayscale, ~10ms)
            │
            ├── every ~30 frames: async MobileNet-SSD re-lock on background thread
            │       └── on detection: re-initialize MOSSE with corrected bbox
            │
            └── Kalman filter [cx, cy, vx, vy]
                    └── velocity-gated lead compensation
                            └── PD controller → PCA9685 → servos
```

**Key design decisions:**
- MOSSE runs on grayscale with a clamped 150×150px max bbox to keep `kcf_update` under 15ms
- MobileNet-SSD only runs on the background thread during tracking — it never blocks the main loop
- Kalman lead compensation is gated on `speed > 30 px/s` so a stationary target isn't pushed off by stale velocity
- On LOST transition: Kalman resets, frame-differencer background resets, and a forced settle period absorbs gimbal vibration before framediff restarts

## File Structure

```
motion-turret/
├── tracker.py          # Main control loop: PD controller, patrol, async DNN sync
├── vision.py           # Camera capture, MobileNet-SSD acquisition, MOSSE tracking
├── kalman.py           # Constant-velocity Kalman filter with lead compensation
├── servos.py           # PCA9685 servo wrapper with angle clamping
├── instrument.py       # Per-stage latency timer → CSV + p50/p95/p99 summaries
├── config.py           # All tuning constants (gains, noise, limits)
├── lidar.py            # TF-Luna UART driver (ready, not yet wired into tracker)
├── MobileNetSSD_deploy.prototxt   # Network architecture
├── MobileNetSSD_deploy.caffemodel # Pre-trained weights (not in repo — see below)
└── tests/
    ├── test_servos.py   # Keyboard-driven manual servo control
    ├── test_direct.py   # Raw PCA9685 channel sweep (hardware diagnosis)
    ├── test_tracker.py  # Live OpenCV window for tracker visualization
    └── test_vision.py   # Live motion detection preview
```

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/SamLangenauer/motion-turret.git
cd motion-turret
python3 -m venv venv
source venv/bin/activate
pip install picamera2 opencv-contrib-python adafruit-circuitpython-servokit numpy pyserial
```

### 2. Download the model weights

The `.caffemodel` file is excluded from the repo (~23MB binary). Download it:

```bash
wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel \
     -O MobileNetSSD_deploy.caffemodel
```

### 3. Enable camera and I²C

```bash
sudo raspi-config
# Interface Options → Camera → Enable
# Interface Options → I2C → Enable
```

### 4. Verify hardware

```bash
# Check camera
rpicam-hello --list-cameras

# Check PCA9685
i2cdetect -y 1   # should show 0x40

# Manual servo test
python3 tests/test_servos.py
```

## Running

```bash
source venv/bin/activate

# Full tracking with MJPEG stream on :8080
python3 tracker.py

# Disable stream for maximum tracking performance
STREAM=0 python3 tracker.py

# Timed benchmark run
DURATION=60 python3 tracker.py
```

View the live stream at `http://<pi-ip>:8080/`.

## Tuning

All tuning constants are in `config.py`.

| Constant | Effect |
|---|---|
| `KP_PAN / KP_TILT` | Proportional gain — higher = snappier, more overshoot |
| `KD_PAN / KD_TILT` | Derivative gain — higher = more damping |
| `DEADZONE_PIXELS` | Error threshold before any servo moves |
| `MAX_NUDGE_DEG` | Slew rate cap per frame |
| `KALMAN_MEAS_NOISE_PX` | Higher = smoother crosshair, slightly more lag |
| `KALMAN_PROC_NOISE` | Higher = faster velocity adaptation |
| `KALMAN_LEAD_MS` | Lead compensation (0 = disabled, tune after gains are set) |

## Measured Latency (p50, active tracking)

| Stage | p50 |
|---|---|
| capture | 2.6 ms |
| preproc | 2.4 ms |
| kcf_update | ~10 ms |
| kalman | 0.5 ms |
| control | 1.4 ms |
| **total** | **~17 ms** |

## Branches

| Branch | Description |
|---|---|
| `main` | Current stable build |
| `kalman-experiment` | Kalman lead compensation work-in-progress |

## Roadmap

- [ ] TF-Luna range-gating (reject detections beyond set distance)
- [ ] Laser-on-lock logic (fire only when aim error < deadzone)
- [ ] Tune Kalman lead compensation with dynamic `sensor_age_ms` feedback
- [ ] Performance characterization table (accuracy vs. distance)
