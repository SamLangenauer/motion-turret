# Motion Tracking Laser Turret

A 2-axis pan/tilt turret that tracks motion with a Raspberry Pi camera
and points a laser at the detected target. Optional TF-Luna LiDAR for
range-gating and distance display.

## Hardware
- Raspberry Pi 4B (running Pi OS Bookworm 64-bit)
- Arducam IMX219 (8MP) camera module
- PCA9685 16-channel PWM driver (I²C)
- 2× Miuzei MS18 9g metal-gear servos
- 5V 5mW 650nm red laser diode
- TF-Luna LiDAR (Phase 5)
- Dedicated 5V 5A supply for the servo rail
- 3D-printed pan/tilt gimbal

## Software phases
1. ~~OS + environment setup~~ ✅
2. ~~Servo control (`servos.py`, `test_servos.py`)~~ ✅ (code done, awaiting hardware)
3. Camera + motion detection (`vision.py`) — awaiting camera
4. Closed-loop tracking (`tracker.py`) — awaiting full hardware
5. TF-Luna integration (`lidar.py`) — awaiting LiDAR
6. Polish: lase-on-lock logic, HUD, calibration routine

## Running
```bash
cd ~/turret
source venv/bin/activate
python3 test_servos.py    # manual control
python3 tracker.py        # autonomous tracking
```

## Project layout
- `config.py` — tuning constants, pin/channel assignments
- `servos.py` — pan/tilt abstraction with safety clamping
- `vision.py` — camera capture + motion detection
- `tracker.py` — closed-loop P-controller tying vision to servos
- `lidar.py` — TF-Luna UART driver
