"""Stillness -> stretch nudge, from synthetic motion. Run: python tests/test_stillness.py"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillness import StillnessWatcher  # noqa: E402

HZ = 25
fails = 0


def check(name, ok):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}")


def still(t0, secs, rnd, gyro=True):
    """A band resting (tilted) on a desk: tiny sensor noise, a small gyro bias."""
    for i in range(int(secs * HZ)):
        g = (0.8 + rnd.gauss(0, 0.3), rnd.gauss(0, 0.3), -0.5) if gyro else (None, None, None)
        yield (0.31 + rnd.gauss(0, .003), -0.42 + rnd.gauss(0, .003), 0.85 + rnd.gauss(0, .003), *g, t0 + i / HZ)


def moving(t0, secs, rnd):
    """Raising / circling the arm: |a| swings by ~0.3 g, turning ~150 deg/s."""
    for i in range(int(secs * HZ)):
        s = i / HZ
        yield (0.3 * math.sin(4 * s), 0.2 * math.cos(4 * s), 1 + 0.3 * math.sin(6 * s),
               150 * math.cos(3 * s), 80 * math.sin(3 * s), 20.0, t0 + s)


def sim_swing(t0, secs):
    """The kit's pretend band left alone (vexo-open-kit sim.py): a gentle swing every 2 s."""
    for i in range(int(secs * HZ)):
        a = 2 * math.pi * (t0 + i / HZ) / 2
        yield (.12 * math.sin(a), .12 * math.cos(a), 1 + .04 * math.sin(3 * a),
               30 * math.cos(a), -30 * math.sin(a), 10 * math.sin(2 * a), t0 + i / HZ)


def run(w, stream):
    return [(s[-1], e) for s in stream if (e := w.feed(*s))]


rnd = random.Random(1)

# 1. Not armed (check-in still going): no nudge, however long it is still
w = StillnessWatcher(still_secs=8)
check("not armed: no nudge in 30 s of stillness", run(w, still(0, 30, rnd)) == [])

# 2. Armed: one nudge at ~still_secs, not before, not repeated while still
w = StillnessWatcher(still_secs=8); w.armed = True
ev = run(w, still(0, 30, rnd))
check(f"armed: exactly one nudge in 30 s ({ev})", [e for _, e in ev] == ["nudge"])
check(f"nudge fires at ~8 s (t={ev[0][0] if ev else None})", ev and 7.5 <= ev[0][0] <= 8.5)

# 3. Moving after the nudge -> "moved", counted once
ev = run(w, moving(30, 3, rnd))
check(f"moved after nudge ({ev})", [e for _, e in ev] == ["moved"])
check("counts: 1 nudge, 1 stretch", (w.nudges, w.stretches) == (1, 1))

# 4. Re-arm needs another FULL still period
ev = run(w, still(33, 5, rnd))
check("re-arm: no nudge after only 5 s still", ev == [])
ev = run(w, still(38, 5, rnd))
check(f"re-arm: nudge after another full still period ({ev})", [e for _, e in ev] == ["nudge"])

# 5. Max 3 nudges per run
w = StillnessWatcher(still_secs=2); w.armed = True
t, kinds = 0.0, []
for _ in range(5):
    kinds += [e for _, e in run(w, still(t, 4, rnd))]; t += 4
    kinds += [e for _, e in run(w, moving(t, 2, rnd))]; t += 2
check(f"max 3 nudges ({kinds.count('nudge')} nudges, {w.stretches} stretches)",
      kinds.count("nudge") == 3 and w.nudges == 3)

# 6. Moving / fidgeting resets the still timer
w = StillnessWatcher(still_secs=8); w.armed = True
ev = run(w, still(0, 6, rnd)) + run(w, moving(6, 1, rnd)) + run(w, still(7, 6, rnd))
check("movement midway resets the timer (6 s + 6 s still: no nudge)", ev == [])

# 7. The pretend band's own swing never counts as still
w = StillnessWatcher(still_secs=5); w.armed = True
check("pretend band swing: no nudge in 30 s", run(w, sim_swing(0, 30)) == [])

# 8. No gyro (accel-only band): still works on |a| alone
w = StillnessWatcher(still_secs=5); w.armed = True
ev = run(w, still(0, 10, rnd, gyro=False))
check(f"accel only: nudge ({ev})", [e for _, e in ev] == ["nudge"])

# 9. HUD T key path: mark_nudged() then movement -> moved
w = StillnessWatcher(still_secs=60); w.armed = True
w.mark_nudged()
ev = run(w, still(0, 3, rnd)) + run(w, moving(3, 2, rnd))
check(f"T key: forced nudge, then moved ({ev})", [e for _, e in ev] == ["moved"] and w.stretches == 1)

print("PASS" if not fails else f"{fails} FAILED"); sys.exit(fails)
