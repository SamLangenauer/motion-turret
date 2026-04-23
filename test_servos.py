# test_servos.py — manual keyboard control
# Arrow keys nudge, 'c' centers, 'q' quits.

import sys
import tty
import termios
from servos import Turret


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # escape — arrow key sequence follows
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main():
    t = Turret()
    print("Arrow keys to move, 'c' to center, 'q' to quit.")
    STEP = 2  # degrees per keypress
    while True:
        k = getch()
        if k == "q":
            break
        elif k == "c":
            t.center()
        elif k == "\x1b[A":  # up
            t.nudge(0, -STEP)
        elif k == "\x1b[B":  # down
            t.nudge(0, STEP)
        elif k == "\x1b[C":  # right
            t.nudge(STEP, 0)
        elif k == "\x1b[D":  # left
            t.nudge(-STEP, 0)
        pan, tilt = t.position
        print(f"pan={pan:5.1f}  tilt={tilt:5.1f}   ", end="\r")
    print()


if __name__ == "__main__":
    main()
