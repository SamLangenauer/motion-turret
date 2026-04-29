# instrument.py — per-stage latency timer with CSV log and periodic summaries.
#
# Usage in a frame loop:
#     timer = StageTimer("latency.csv")
#     while running:
#         timer.frame_begin()
#         timer.mark("capture"); ... ; timer.lap("capture")
#         timer.mark("detect");  ... ; timer.lap("detect")
#         timer.note("state", "tracking")
#         timer.frame_end()        # writes a CSV row, prints summary every N frames
#
# Each frame_end() writes one CSV row and updates rolling history. Every
# `summary_every` frames, a p50/p95/p99/max table is printed for each stage
# over the last `history_len` frames.

import time
import csv
import os
from collections import defaultdict, deque


class StageTimer:
    def __init__(self, csv_path="latency.csv", summary_every=60, history_len=600):
        self._csv_path = csv_path
        self._summary_every = summary_every
        self._history = defaultdict(lambda: deque(maxlen=history_len))

        # Per-frame transient state
        self._marks = {}       # stage_name -> start ns
        self._stages = {}      # stage_name -> duration ms (current frame only)
        self._notes = {}       # extra fields for current frame
        self._frame_start_ns = None
        self._frame_idx = 0

        # Stable column ordering for CSV output (extended as new keys appear)
        self._known_stages = []
        self._known_notes = []

        new_file = not os.path.exists(csv_path)
        self._csv_f = open(csv_path, "a", buffering=1, newline="")
        self._csv_writer = csv.writer(self._csv_f)
        self._header_written = not new_file

    # ---- per-frame API ----

    def frame_begin(self):
        self._stages = {}
        self._notes = {}
        self._marks = {}
        self._frame_start_ns = time.monotonic_ns()

    def mark(self, name):
        """Start a stage timer."""
        self._marks[name] = time.monotonic_ns()

    def lap(self, name):
        """End the named stage; record duration in ms and return it."""
        now = time.monotonic_ns()
        start = self._marks.pop(name, now)
        dur_ms = (now - start) / 1e6
        self._stages[name] = dur_ms
        return dur_ms

    def note(self, key, value):
        """Attach a non-timing field to the current frame's row."""
        self._notes[key] = value

    def frame_end(self):
        if self._frame_start_ns is not None:
            self._stages["total"] = (time.monotonic_ns() - self._frame_start_ns) / 1e6

        for n, d in self._stages.items():
            if n not in self._known_stages:
                self._known_stages.append(n)
            self._history[n].append(d)
        for k in self._notes:
            if k not in self._known_notes:
                self._known_notes.append(k)

        if not self._header_written:
            header = ["frame_idx", "wall_t"]
            header += [f"ms_{n}" for n in self._known_stages]
            header += list(self._known_notes)
            self._csv_writer.writerow(header)
            self._header_written = True

        row = [self._frame_idx, f"{time.time():.6f}"]
        for n in self._known_stages:
            v = self._stages.get(n)
            row.append(f"{v:.3f}" if v is not None else "")
        for k in self._known_notes:
            row.append(self._notes.get(k, ""))
        self._csv_writer.writerow(row)

        self._frame_idx += 1
        if self._summary_every and self._frame_idx % self._summary_every == 0:
            self.print_summary()

    # ---- output ----

    def print_summary(self):
        n_window = len(self._history.get("total", []))
        print(f"\n--- latency @ frame {self._frame_idx} "
              f"(window: last {n_window} frames) ---")
        print(f"  {'stage':<18} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}  (ms)")
        for n in self._known_stages:
            samples = sorted(self._history[n])
            if not samples:
                continue
            sn = len(samples)
            p50 = samples[sn // 2]
            p95 = samples[min(sn - 1, int(sn * 0.95))]
            p99 = samples[min(sn - 1, int(sn * 0.99))]
            mx = samples[-1]
            print(f"  {n:<18} {p50:>8.2f} {p95:>8.2f} {p99:>8.2f} {mx:>8.2f}")
        print()

    def close(self):
        try:
            self.print_summary()
        finally:
            self._csv_f.close()
