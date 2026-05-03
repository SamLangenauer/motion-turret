# test_tracker.py — live preview of the closed-loop tracker.
# Shows the camera feed, frame center crosshair, CSRT bbox when tracking,
# state label, and current servo angles.

import cv2
from tracker import Tracker
import config


def main():
    t = Tracker()
    print("Press 'q' in the video window to quit.")

    try:
        while True:
            t.step()

            # Re-detect for visualization purposes (cheap; tracker.step already ran)
            # We want the latest frame and target without re-driving servos.
            frame = t._last_frame
            target = t._last_target
            if frame is None:
                continue
            h, w = frame.shape[:2]

            # Convert RGB (picamera2 default) to BGR for OpenCV display
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Frame center crosshair
            cx0, cy0 = w // 2, h // 2
            cv2.line(display, (cx0 - 20, cy0), (cx0 + 20, cy0), (0, 255, 0), 1)
            cv2.line(display, (cx0, cy0 - 20), (cx0, cy0 + 20), (0, 255, 0), 1)

            # Deadzone box
            dz = config.DEADZONE_PIXELS
            cv2.rectangle(display,
                          (cx0 - dz, cy0 - dz), (cx0 + dz, cy0 + dz),
                          (0, 200, 0), 1)

            # Target marker
            if target is not None:
                tx, ty, area = target
                cv2.circle(display, (tx, ty), 10, (0, 0, 255), 2)
                # Draw an error vector from frame center to target
                cv2.arrowedLine(display, (cx0, cy0), (tx, ty),
                                (255, 100, 0), 1, tipLength=0.15)

            # State label
            state = t.vision.state
            color = {
                "tracking": (0, 255, 0),
                "acquiring": (0, 255, 255),
                "searching": (200, 200, 200),
                "lost": (0, 0, 255),
            }.get(state, (255, 255, 255))
            cv2.putText(display, f"state: {state}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Servo readout
            pan, tilt = t.turret.position
            cv2.putText(display, f"pan={pan:5.1f}  tilt={tilt:5.1f}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("turret tracker", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        t.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
