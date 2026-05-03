# test_direct.py — bypasses config/servos.py, hits PCA9685 directly
# Tests every channel 0-5 so we can find which one your servos are on

from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

# Set pulse range for all test channels
for ch in range(6):
    kit.servo[ch].set_pulse_width_range(500, 2500)

print("Testing channels 0-5 one at a time.")
print("Watch for ANY servo movement.\n")

while True:
    for ch in range(6):
        print(f"Channel {ch} → 60°", end="  ")
        kit.servo[ch].angle = 60
        time.sleep(0.8)
        print(f"→ 120°")
        kit.servo[ch].angle = 120
        time.sleep(0.8)
        kit.servo[ch].angle = 90
        time.sleep(0.3)
    print("--- loop ---")

