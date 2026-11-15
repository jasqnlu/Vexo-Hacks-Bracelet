"""Stillness watcher: "you've been still a while -> stretch?" from the band's motion only.

Pure Python, no band, no Pipecat: feed it motion samples, it says when to nudge and when the wrist moved again.

  w = StillnessWatcher(still_secs=15)
  w.armed = True                      # only after the check-in card is saved (holding still for the dial is normal)
  ev = w.feed(ax, ay, az, gx, gy, gz, t)   # -> "nudge" | "moved" | None ; t in seconds

Stillness = |a| (acceleration magnitude, in g) barely changing over a short sliding window, and the gyro (if the
band sends it) barely turning. Not the wrist angle: a wrist resting tilted on a desk is still too.
"""
import math
from collections import deque

WINDOW_SECS = 1.0          # sliding window for the "still right now?" test
STILL_STD_G = 0.01         # std of |a| below this = still (a band on a desk: ~0.002-0.005 g of sensor noise)
STILL_GYRO_DPS = 5.0       # mean |gyro| below this = still (covers a few deg/s of gyro bias)
MOVE_STD_G = 0.06          # "moved clearly": std of |a| above this ...
MOVE_GYRO_DPS = 60.0       # ... or any |gyro| sample above this (a raised arm / shoulder roll is far beyond)
MAX_NUDGES = 3             # per run


class StillnessWatcher:
    def __init__(self, still_secs: float = 15.0, max_nudges: int = MAX_NUDGES):
        self.still_secs, self.max_nudges = still_secs, max_nudges
        self.armed = False             # set True after the check-in card is saved
        self.nudges = 0                # stretch reminders given
        self.stretches = 0             # times the wrist moved clearly after a reminder
        self.waiting = False           # a nudge was given; waiting for the wrist to move
        self._win: deque = deque()     # (t, |a|, |gyro| or None)
        self._still_since = None

    def _stats(self):
        a = [x[1] for x in self._win]
        mean = sum(a) / len(a)
        std = math.sqrt(sum((x - mean) ** 2 for x in a) / len(a))
        g = [x[2] for x in self._win if x[2] is not None]
        return std, (sum(g) / len(g) if g else 0.0), (max(g) if g else 0.0)

    def feed(self, ax, ay, az, gx, gy, gz, t: float):
        if ax is None or ay is None or az is None:
            return None
        amag = math.sqrt(ax * ax + ay * ay + az * az)
        gmag = math.sqrt(gx * gx + gy * gy + gz * gz) if None not in (gx, gy, gz) else None
        self._win.append((t, amag, gmag))
        while self._win and t - self._win[0][0] > WINDOW_SECS:
            self._win.popleft()
        if not self.armed:
            self._still_since = None
            return None
        if t - self._win[0][0] < WINDOW_SECS * 0.8:      # not enough samples yet for a fair test
            return None
        std, g_mean, g_max = self._stats()

        if self.waiting:                                  # after a nudge: only look for a clear movement
            if std > MOVE_STD_G or g_max > MOVE_GYRO_DPS:
                self.mark_moved()
                return "moved"
            return None

        if std < STILL_STD_G and g_mean < STILL_GYRO_DPS:
            if self._still_since is None:
                self._still_since = self._win[0][0]       # still since the start of this window
            if t - self._still_since >= self.still_secs and self.nudges < self.max_nudges:
                self.mark_nudged()
                return "nudge"
        else:
            self._still_since = None
        return None

    def mark_nudged(self):
        """A nudge was given (by the timer, or the HUD's T key): wait for the wrist to move."""
        self.nudges += 1
        self.waiting, self._still_since = True, None

    def mark_moved(self):
        """The wrist moved after a nudge: count a stretch and re-arm (another FULL still period is needed)."""
        self.stretches += 1
        self.waiting, self._still_since = False, None
        self._win.clear()
