"""Voice answers and corrections update the HUD right away (no Gemini call). Run: python tests/test_voice_levels.py"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LOG_LEVEL", "ERROR")
import hud_bot as hb  # noqa: E402


class Hud:
    def __init__(self): self.msgs = []
    async def send(self, m, sticky=True): self.msgs.append(m)


CASES = [  # (topic the bot is asking about, already chosen levels, wearer said, expected levels after)
    ("appetite", {"appetite": "more_than_usual"}, "Actually can you change my appetite to normal", {"appetite": "normal"}),
    ("sleep", {}, "I slept pretty badly", {"sleep": "poor"}),
    ("sleep", {}, "It was good, I slept well", {"sleep": "good"}),
    ("mood", {}, "Honestly not good", {}),                                # negated: don't guess
    ("stress", {}, "Stress is pretty high this week", {"stress": "high"}),
    ("appetite", {}, "I've been eating less than usual", {"appetite": "less_than_usual"}),
    ("mood", {"sleep": "poor"}, "Oh and my sleep was fair, not poor", {"sleep": "fair"}),  # correction of an earlier topic
    ("mood", {}, "My sister called me", {}),                              # nothing to match
    (None, {}, "good", {}),                                               # no topic yet: ignore
]


async def main():
    fails = 0
    for topic, before, said, expect in CASES:
        s = hb.Session(Hud(), dwell=0)
        s.topic, s.levels = topic, dict(before)
        s.dial_on = topic is not None and topic not in before
        await s.heard(said)
        got = {k: v for k, v in s.levels.items() if before.get(k) != v}
        ok = got == expect
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {said!r:55} -> {got}" + ("" if ok else f"   expected {expect}"))
    if s.dial_on is not None:
        pass
    print("PASS" if not fails else f"{fails} FAILED")
    return fails


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
