"""HUD server: the bridge between the browser HUD (camera + overlays) and the bot.

  python3 hud_server.py            standalone test: open http://localhost:8765, allow the camera

The browser owns the camera. It sends a small JPEG over a WebSocket about once a second; this
server keeps ONLY the latest frame, in memory (never written to disk), so the bot can attach it
to Gemini when the wearer shows something. The server pushes JSON messages back to every open
HUD tab: check-in state, cards, wrist pose. Bound to 127.0.0.1: nothing outside this Mac can connect.
"""
import asyncio
import json
import time
from pathlib import Path

from aiohttp import WSMsgType, web
from loguru import logger

HERE = Path(__file__).resolve().parent
MAX_FRAME_BYTES = 2_000_000          # a 640 px JPEG is ~50 KB; anything huge is refused


class HudServer:
    def __init__(self, host="127.0.0.1", port=8765):
        self.host, self.port = host, port
        self._clients: set[web.WebSocketResponse] = set()
        self._frame: bytes | None = None
        self._frame_at = 0.0
        self.frames_received = 0
        self._sticky: dict[str, dict] = {}   # last message of each type, replayed to a tab that (re)connects
        self._runner: web.AppRunner | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---------------------------------------------------------------- lifecycle
    async def start(self):
        self._loop = asyncio.get_running_loop()
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()
        logger.info(f"HUD: open http://localhost:{self.port}")

    async def stop(self):
        for ws in list(self._clients):
            await ws.close()
        if self._runner:
            await self._runner.cleanup()

    # ---------------------------------------------------------------- frames
    def latest_frame(self, max_age=5.0):
        """The newest JPEG if it is fresh enough, else None. Never stored anywhere else."""
        if self._frame is None or time.time() - self._frame_at > max_age:
            return None
        return self._frame

    def frame_age(self):
        return None if self._frame is None else time.time() - self._frame_at

    # ---------------------------------------------------------------- outgoing
    async def send(self, msg: dict, sticky=True):
        """Send a JSON message to every HUD tab. sticky: a newly opened tab gets the latest one too."""
        if sticky:
            self._sticky[msg.get("type", "")] = msg
        data = json.dumps(msg)
        for ws in list(self._clients):
            try:
                await ws.send_str(data)
            except Exception:                                # noqa: BLE001  tab closed mid-send
                self._clients.discard(ws)

    def send_threadsafe(self, msg: dict, sticky=True):
        """For band callbacks, which run on the band library's thread."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(self.send(msg, sticky), self._loop)

    # ---------------------------------------------------------------- routes
    async def _index(self, request):
        return web.FileResponse(HERE / "hud.html", headers={"Cache-Control": "no-store"})

    async def _ws(self, request):
        ws = web.WebSocketResponse(max_msg_size=MAX_FRAME_BYTES, heartbeat=20)
        await ws.prepare(request)
        self._clients.add(ws)
        logger.info(f"HUD tab connected ({len(self._clients)} open)")
        for msg in self._sticky.values():
            await ws.send_str(json.dumps(msg))
        try:
            async for m in ws:
                if m.type == WSMsgType.BINARY:
                    if m.data[:2] == b"\xff\xd8":            # JPEG magic bytes: ignore anything else
                        self._frame, self._frame_at = bytes(m.data), time.time()
                        self.frames_received += 1
                elif m.type == WSMsgType.TEXT:
                    logger.debug(f"HUD says: {m.data[:200]}")
        finally:
            self._clients.discard(ws)
            logger.info(f"HUD tab closed ({len(self._clients)} open)")
        return ws


# ---------------------------------------------------------------- standalone test
async def _demo():
    hud = HudServer()
    await hud.start()
    await hud.send({"type": "state", "topic": "sleep", "listening": False,
                    "topics": {"sleep": None, "mood": None, "stress": None, "appetite": None}})
    await hud.send({"type": "caption", "who": "bot", "text": "HUD test: if you see this, the server works."},
                   sticky=False)
    n = 0
    try:
        while True:
            await asyncio.sleep(2)
            n += 1
            age = hud.frame_age()
            logger.info(f"frames received: {hud.frames_received}, latest age: "
                        f"{'none' if age is None else f'{age:.1f}s'}")
            if n == 3:
                await hud.send({"type": "card", "id": "test", "title": "Test card",
                                "body": "Cards from the bot appear here.", "kind": "info"}, sticky=False)
    finally:
        await hud.stop()


if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass
