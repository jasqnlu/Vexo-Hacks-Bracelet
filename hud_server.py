"""HUD server: the bridge between the browser HUD (camera + overlays) and the bot.

  python3 hud_server.py            standalone test: open http://localhost:8765, allow the camera
                                   (also serves the clinician dashboard: http://localhost:8765/dashboard)

The browser owns the camera. It sends a small JPEG over a WebSocket about once a second; this
server keeps ONLY the latest frame, in memory (never written to disk), so the bot can attach it
to Gemini when the wearer shows something. The server pushes JSON messages back to every open
HUD tab: check-in state, cards, wrist pose. Bound to 127.0.0.1: nothing outside this Mac can connect.
"""
import asyncio
import json
import os
import re
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
        self.on_message = None
        self._recap_cache = None             # (history version, mp3 bytes)
        self._recap_lock = asyncio.Lock()               # async fn(dict): keyboard fallbacks from the HUD (show, dial, confirm)

    # ---------------------------------------------------------------- lifecycle
    async def start(self):
        self._loop = asyncio.get_running_loop()
        allowed = {f"localhost:{self.port}", f"127.0.0.1:{self.port}"}

        @web.middleware
        async def host_guard(request, handler):
            # DNS rebinding: a website can point its own domain at 127.0.0.1, and the browser then lets
            # that page read our responses. Its requests carry ITS hostname in Host, so allow only ours.
            if request.headers.get("Host", "").lower() not in allowed:
                logger.warning(f"HUD: refused request with Host {request.headers.get('Host')!r}")
                raise web.HTTPForbidden()
            return await handler(request)

        app = web.Application(middlewares=[host_guard])
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws)
        # Clinician dashboard. Only these exact files are served, never the folder (.env lives there).
        app.router.add_get("/dashboard", self._dashboard)
        app.router.add_get("/cards/{name}", self._card_file)
        app.router.add_get("/provenance.jsonl", self._provenance)
        app.router.add_get("/recap.mp3", self._recap)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()
        logger.info(f"HUD: open http://localhost:{self.port}   ·   dashboard: http://localhost:{self.port}/dashboard")

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

    async def _dashboard(self, request):
        return web.FileResponse(HERE / "card.html", headers={"Cache-Control": "no-store"})

    async def _card_file(self, request):
        name = request.match_info["name"]
        f = HERE / "cards" / name
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}\.json", name) or not f.is_file():
            raise web.HTTPNotFound()          # card.html treats a missing file as "no data yet"
        return web.FileResponse(f, headers={"Cache-Control": "no-store"})

    async def _provenance(self, request):
        f = HERE / "provenance.jsonl"
        if not f.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(f, headers={"Cache-Control": "no-store"})

    async def _recap(self, request):
        """The 7-day overview read aloud in Tend's voice (ElevenLabs). Only the overview lines leave the Mac:
        they are counts of reported levels, no names or quotes, and each is re-checked for diagnostic words.
        One ElevenLabs call per history version (cached in memory), so repeated clicks cost nothing."""
        hist_file = HERE / "cards" / "history.json"
        if not hist_file.is_file():
            return web.json_response({"error": "No check-in history yet."}, status=409)
        hist = json.loads(hist_file.read_text())
        version = hist.get("generated", "")
        async with self._recap_lock:
            if self._recap_cache and self._recap_cache[0] == version:
                return web.Response(body=self._recap_cache[1], content_type="audio/mpeg",
                                    headers={"Cache-Control": "no-store"})
            import guardrails
            lines = [l["text"] for l in hist.get("overview", {}).get("lines", [])
                     if not guardrails._DIAG_RE.search(l.get("text", ""))]
            n = len(hist.get("timeline", []))
            text = (f"Here's a summary of the last {n} check-ins. " + " ".join(lines) if lines else
                    f"There's nothing to flag across the last {n} check-ins.")
            text += " These are patterns in what was reported, not a diagnosis."
            if not os.getenv("ELEVENLABS_API_KEY"):
                from dotenv import load_dotenv
                load_dotenv(HERE / ".env")
            key, voice = os.getenv("ELEVENLABS_API_KEY"), os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
            if not key:
                return web.json_response({"error": "No ElevenLabs key in .env."}, status=503)
            import aiohttp
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128"
            audio, err = None, ""
            async with aiohttp.ClientSession() as http:
                for model in dict.fromkeys([os.getenv("ELEVENLABS_MODEL", "eleven_v4_turbo"), "eleven_flash_v2_5"]):
                    async with http.post(url, json={"text": text, "model_id": model},
                                         headers={"xi-api-key": key, "accept": "audio/mpeg"}) as r:
                        if r.status == 200:
                            audio = await r.read()
                            break
                        err = f"ElevenLabs {r.status} on {model}"
                        logger.warning(f"recap: {err}")
            if audio is None:
                return web.json_response({"error": err or "ElevenLabs failed."}, status=502)
            import guardrails as g
            g.Provenance(HERE / "provenance.jsonl").log("dashboard->elevenlabs", "7-day overview text (counts only)",
                                                        "ElevenLabs (cloud)", model=model)
            self._recap_cache = (version, audio)
            return web.Response(body=audio, content_type="audio/mpeg", headers={"Cache-Control": "no-store"})

    async def _ws(self, request):
        # Browsers don't apply same-origin rules to WebSockets: without this check any website open in
        # the browser could read the check-in or inject camera frames. Browsers always send Origin;
        # a local script (no Origin) is already on this Mac, so it is allowed.
        origin = request.headers.get("Origin")
        if origin is not None and origin not in (f"http://localhost:{self.port}", f"http://127.0.0.1:{self.port}"):
            logger.warning(f"HUD: refused WebSocket from {origin}")
            raise web.HTTPForbidden()
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
                    if self.on_message:
                        try:
                            msg = json.loads(m.data)
                            if isinstance(msg, dict):
                                await self.on_message(msg)
                        except Exception as e:               # noqa: BLE001  a bad message must not drop the tab
                            logger.warning(f"HUD message: {e}")
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
