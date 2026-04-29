# Motion Tracking Laser Turret

A high-speed, 2-axis pan/tilt turret that tracks motion using a Raspberry Pi camera
and points a laser at the detected target. Features an 80Hz hardware-accelerated 
computer vision pipeline, MOSSE correlation tracking, and an optional TF-Luna LiDAR 
for range-gating.

## Hardware
- Raspberry Pi 4B (running Pi OS Bookworm 64-bit)
- Arducam IMX219 (8MP) camera module
- PCA9685 16-channel PWM driver (I²C)
- 2× GH-S37D 3.7g micro servos
- 5V 5mW 650nm red laser diode
- TF-Luna LiDAR (Phase 5)
- Dedicated 5V 5A power supply (wired to PCA9685 terminal block for servo isolation)
- 3D-printed pan/tilt gimbal

## Software phases
1. ~~OS + environment setup~~ ✅
2. ~~Servo control (`servos.py`, `test_servos.py`)~~ ✅
3. ~~Camera + motion detection (`vision.py`)~~ ✅ (Hardware ISP bypass, Framediff -> MOSSE)
4. ~~Closed-loop tracking (`tracker.py`)~~ ✅ (80Hz P-controller with MJPEG stream)
5. TF-Luna integration (`lidar.py`) — awaiting LiDAR
6. Polish: lase-on-lock logic, HUD, calibration routine

## Features
- **Hardware Acceleration:** Bypasses CPU color-space conversion using libcamera/Picamera2 ISP.
- **Two-Stage Vision:** Uses lightweight frame differencing to acquire targets, then hands off to an ultra-fast MOSSE correlation filter for 80 FPS tracking.
- **Time-Invariant P-Controller:** Closed-loop servo driving mathematically smoothed to reject high-frequency bounding box noise.
- **Latency Instrumentation:** Built-in `StageTimer` that logs per-frame phase durations to CSV and outputs rolling p50/p95/p99 latency metrics.
- **Asynchronous Web HUD:** Built-in MJPEG streaming server for remote monitoring without blocking the main control loop.

## Running
```bash
cd ~/turret
source venv/bin/activate

# Manual servo calibration/test
python3 test_servos.py    

# Autonomous tracking (Default: Stream enabled on port 8080)
python3 tracker.py        

# Run without MJPEG stream (maximizes tracking performance)
STREAM=0 python3 tracker.py

# Run for a specific benchmark duration (e.g., 60 seconds)
DURATION=60 python3 tracker.py
