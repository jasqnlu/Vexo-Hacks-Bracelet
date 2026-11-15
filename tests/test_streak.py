"""Check-in streak = consecutive calendar days with a check-in. Run: python tests/test_streak.py"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LOG_LEVEL", "ERROR")
import hud_bot as hb  # noqa: E402

T = date(2026, 11, 14)
def tl(*days): return [{"created": f"2026-11-{d:02d}T09:00:00"} for d in days]

CASES = [  # (timeline, include_today, expected)
    (tl(8, 9, 10, 11, 12, 13), True, 7),      # yesterday back 6 days + today's check-in = day 7
    (tl(8, 9, 10, 11, 12, 13, 14), True, 7),  # already checked in today: today isn't counted twice
    (tl(10, 11, 13), True, 2),                # gap on the 12th: only 13 + today
    (tl(), True, 1),                          # first ever
    (tl(13, 13, 12), True, 3),                # two check-ins one day count once
    (tl(12, 13, 14), False, 3),               # after saving: count what's in history
    (tl(11, 12), False, 0),                   # nothing yesterday or today
]
fails = 0
for timeline, inc, want in CASES:
    got = hb.streak_days(timeline, today=T, include_today=inc)
    fails += got != want
    print(f"{'ok  ' if got == want else 'FAIL'} {[t['created'][8:10] for t in timeline]} today={inc} -> {got} (want {want})")
print("PASS" if not fails else f"{fails} FAILED"); sys.exit(fails)
