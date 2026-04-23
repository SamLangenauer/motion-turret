# test_vision.py — live window showing detected motion.
# Needs a display. If running headless over SSH, see notes below.

import cv2
from vision import MotionTracker


def main():
    tracker = MotionTracker()
    print("Press 'q' in the video window to quit.")
    try:
        while True:
            frame, target, mask = tracker.detect()

            # Draw crosshairs at frame center
            h, w = frame.shape[:2]
            cx0, cy0 = w // 2, h // 2
            cv2.line(frame, (cx0 - 15, cy0), (cx0 + 15, cy0), (0, 255, 0), 1)
            cv2.line(frame, (cx0, cy0 - 15), (cx0, cy0 + 15), (0, 255, 0), 1)

            # Mark target if one was found
            if target is not None:
                cx, cy, area = target
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), 2)
                cv2.putText(frame, f"area={int(area)}", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("turret view", frame)
            cv2.imshow("motion mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
