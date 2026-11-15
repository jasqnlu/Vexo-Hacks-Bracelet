"""Saying "I'd rather type" opens the text box; talking again before Tend's reply plays drops that reply.
Run: python tests/test_typing_and_turns.py"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hud_bot as hb  # noqa: E402


class FakeHud:
    def __init__(self): self.sent = []
    async def send(self, m, sticky=True): self.sent.append(m)


class FakeInterrupter:
    def __init__(self): self.n = 0
    async def broadcast_interruption(self): self.n += 1


async def main():
    ok = True
    # 1. spoken "I'd prefer to type" while the tell-me-more menu is up -> text box opens for that topic
    s = hb.Session(FakeHud()); s.loop = asyncio.get_running_loop(); s.interrupter = FakeInterrupter()
    s.topic, s.menu, s.more_for, s.dial_on = "sleep", "more", ("sleep", "fair"), True
    await s.heard("I would prefer to type my reasons")
    boxes = [m for m in s.hud.sent if m.get("type") == "typebox" and m.get("open")]
    ok &= bool(boxes) and boxes[0]["topic"] == "sleep" and s.typing and s.menu is None and not s.dial_on
    print("voice 'type' opens the box:", bool(boxes), s.typing)
    # 2. same with no menu (e.g. at "anything else?")
    s2 = hb.Session(FakeHud()); s2.loop = asyncio.get_running_loop()
    await s2.heard("can I type it instead")
    ok &= any(m.get("type") == "typebox" and m.get("open") for m in s2.hud.sent)
    # 3. talking again before the reply plays -> the pending reply is dropped
    s3 = hb.Session(FakeHud()); s3.loop = asyncio.get_running_loop(); s3.interrupter = FakeInterrupter()
    s3.reply_pending = True
    await s3.speaking("user", True)
    ok &= s3.interrupter.n == 1 and not s3.reply_pending
    # 4. ...but not while Tend is already talking (mic is muted then anyway)
    s4 = hb.Session(FakeHud()); s4.loop = asyncio.get_running_loop(); s4.interrupter = FakeInterrupter()
    s4.reply_pending, s4.bot_talking = True, True
    await s4.speaking("user", True)
    ok &= s4.interrupter.n == 0
    print("stale reply dropped on new turn:", s3.interrupter.n == 1, "| not while speaking:", s4.interrupter.n == 0)
    print("PASS" if ok else "FAIL")
    return ok

sys.exit(0 if asyncio.run(main()) else 1)
