"""Tend with a camera HUD: talk, tilt, or show it something.

  python3 hud_bot.py --sim --max-minutes 2          pretend band, then open http://localhost:8765
  python3 hud_bot.py 1 --max-minutes 3              real band 1 (double-tap it first)

Same check-in as neurokind_bot.py (which stays unchanged as the fallback demo), plus:
  - hud.html shows the camera, the topic list, captions, a wrist-tilt dial and answer cards
  - showing something attaches ONE camera frame to Gemini: say "here's my lunch" / "look at this",
    or press P on the HUD (no band tap: a flick reads as a tap). Frames live only in memory; nothing is saved.
  - the dial: tilt the wrist to pick an answer, HOLD it there --dwell 1.2 s to confirm (a bar fills);
    double-tap (or Enter on the HUD) confirms at once
  - the card gets an optional `shown` line per topic (objects only, never people), safety-gated
HUD keys: P show camera · ← → move the dial · Enter confirm · M mirror · C captions.
"""
import argparse
import asyncio
import copy
import json
import math
import os
import re
import time
from datetime import datetime

import neurokind_bot as nk                     # reuses band input, redaction, prompt, provenance, logging
import guardrails
from hud_server import HudServer
from neurokind_bot import HERE, KIT, SAMPLE_RATE, logger, prov

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame,  # noqa: E402
                                   UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                                   Frame, LLMFullResponseEndFrame, LLMRunFrame, LLMTextFrame,
                                   TranscriptionFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (LLMContextAggregatorPair,
                                                                    LLMUserAggregatorParams)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

TOPICS = guardrails.DOMAINS                     # sleep, mood, stress, appetite
# Dial order, left to right on screen (the HUD shows them in this order).
DIAL = {"sleep": ["poor", "fair", "good"], "mood": ["low", "okay", "good"],
        "stress": ["low", "medium", "high"], "appetite": ["less_than_usual", "normal", "more_than_usual"]}
# The dial shows SKIP first (tilt all the way left), then the three answers. A resting wrist = the middle answer.
SKIP = "skip"
REST = 2                                        # index of the middle answer in [skip, a, b, c]
EDGES = (-36.0, -14.0, 14.0)                    # tilt (deg) between skip|a, a|b, b|c
HYST = 4.0                                      # cross an edge by this much to move; so in at 18°, out at 10°


def dial_options(topic: str) -> list[str]:
    return [SKIP] + DIAL[topic]
TOPIC_RE = {"sleep": r"\b(sleep|slept|rest(ed)?|dormi\w*|duerm\w*|sueño|descans\w*)\b",
            "mood": r"\b(mood|feeling|ánimo|animo|sientes|humor)\b",
            "stress": r"\b(stress(ed|ful)?|estr[eé]s\w*)\b",
            "appetite": r"\b(appetite|eat(ing)?|ate|food|meals?|hungry|apetito|comer|comido|comida)\b"}
# Answer words the wearer might SAY, per topic -> level. Longest phrases first so "less than usual" isn't
# read as "usual". Only for the live HUD; Gemini's summary at the end is what goes on the card.
_SAID = {
    "sleep": [("good", r"good|well|great"), ("fair", r"fair|okay|ok|alright|so-so"),
              ("poor", r"poor|bad|badly|terrible|awful|barely")],
    "mood": [("good", r"good|great|happy"), ("okay", r"okay|ok|fine|alright"), ("low", r"low|down|sad|bad")],
    "stress": [("high", r"very stressed|high|a lot"), ("medium", r"medium|moderate"), ("low", r"low|calm|relaxed")],
    "appetite": [("less_than_usual", r"less than usual|less|smaller"), ("more_than_usual", r"more than usual|more|hungrier"),
                 ("normal", r"normal|usual")],
}
SAID_RE = {t: re.compile("|".join(f"(?P<{lvl}>\\b(?:{rx})\\b)" for lvl, rx in opts), re.I) for t, opts in _SAID.items()}
NEGATED_RE = re.compile(r"\b(not|never|no)\s+(\w+\s+)?$|n't\s+(\w+\s+)?$", re.I)

SHOW_RE = re.compile(r"\b(here'?s|here is|look(ing)? at|take a look|see this|this is my|show(ing)? you|"
                     r"my (lunch|dinner|breakfast|meal|plate|food|snack))\b", re.I)
PEOPLE_RE = re.compile(r"\b(person|people|man|woman|men|women|face|faces|child|kid|boy|girl|someone|"
                       r"he|she|his|her|they look)\b", re.I)
IMAGE_NOTE = ("(Camera frame: the wearer is showing you this right now. Describe only the objects you can see, "
              "in a few neutral words. Never describe people or faces. Never guess about health from it.)")

LANG_EXTRA = {
    "en": "",
    "es": """

LANGUAGE: the wearer chose Spanish. Speak ONLY natural, warm Spanish with them (use "tú"), every turn, including
the greeting, follow-ups, "¿Hay algo más que quieras añadir?" and the goodbye. Name the topics plainly in Spanish
(sueño, ánimo, estrés, apetito) so the screen can follow. For self-harm, say they can call or text 988 any time
(it has Spanish support). BUT write every submit_summary text field (reported, other_notes, shown) in ENGLISH for
the clinician; a `quote` keeps the wearer's exact Spanish words.""",
}

HUD_EXTRA = """

This check-in also has a camera screen. Name each topic plainly when you ask about it (sleep, mood,
stress, appetite) so the screen can follow along.

Sometimes a message includes a camera image: the wearer is showing you something, often their meal while
you talk about appetite. Mention what you see in a few words, then continue. When you submit, put a short
neutral phrase naming the visible objects in that topic's `shown` field (for example "sandwich and an apple,
about half eaten"). Never describe people, faces, or bodies, and never guess about health from an image.
If the image is not about the check-in, answer in one sentence and return to the check-in.

Sometimes a message says the wearer chose an answer with a wrist tilt: treat it as their answer for that
topic, acknowledge in a few words, and go on to the next topic. If they chose to skip a topic, say that's
fine, don't ask about it again, and submit it as not_discussed with reported "Wearer chose to skip this topic".
After a wrist-tilt answer, don't ask an open follow-up: the wearer may not use speech. Just go on.
If a message says the wearer double-tapped to move on, they have nothing to add: go to the next step.
When you ask "anything else?", add "or double-tap if that's all"."""


# ------------------------------------------------------------------ shared check-in state
class Session:
    """What the HUD shows. Touched from the event loop; band callbacks hop onto it via the loop."""

    def __init__(self, hud: HudServer, flip_dial=False, dwell=1.2):
        self.hud, self.flip = hud, flip_dial
        self.dwell = dwell                                # seconds to hold a tilt before it confirms (0 = off)
        self.dial_armed = False                           # dwell counts only after the wrist has moved once
        self._dwell_task = None
        self._last_move_on = 0.0
        self._avatar = None
        self.lang = "en"
        self.lang_chosen: asyncio.Future = asyncio.get_event_loop().create_future()
        self.loop = None
        self.task = None
        self.context = None
        self.topic = None
        self.levels: dict[str, str] = {}
        self.shown: dict[str, bool] = {}
        self.dial_index = REST
        self.dial_on = False
        self.last_attach = 0.0
        self.last_roll = 0.0
        self.roll_zero = None                             # set when a dial appears (see on_roll)

    async def avatar(self, state: str, force: bool = False):
        """Sprout's state on the HUD: idle | listening | thinking | speaking | concern (concern is final)."""
        if (self._avatar == "concern" and not force) or state == self._avatar:
            return
        self._avatar = state
        await self.hud.send({"type": "avatar", "state": state})   # sticky: a reloaded tab gets it back

    async def push_state(self, listening=True):
        await self.hud.send({"type": "state", "topic": self.topic, "listening": listening,
                             "topics": {t: self.levels.get(t) for t in TOPICS}, "shown": self.shown})

    async def push_dial(self):
        if self.topic:
            await self.hud.send({"type": "dial", "active": self.dial_on, "topic": self.topic,
                                 "options": dial_options(self.topic), "index": self.dial_index,
                                 "dwell": self.dwell if (self.dial_armed and self.dial_on) else 0,
                                 "question": f"{self.topic.capitalize()}: tilt to choose"})

    async def bot_said(self, text: str):
        """Follow the conversation: the last topic the bot names is the one it is asking about."""
        await self.hud.send({"type": "caption", "who": "bot", "text": text}, sticky=False)
        hits = [(m.start(), t) for t, rx in TOPIC_RE.items() for m in re.finditer(rx, text, re.I)]
        if not hits:
            return
        new = max(hits)[1]
        if new != self.topic and new not in self.levels:
            if self.topic and self.topic not in self.levels:
                self.levels[self.topic] = "said"          # answered by voice; the level arrives with the summary
            self.topic, self.dial_index, self.dial_on = new, REST, True
            self.roll_zero = None                         # re-centre on the wrist's next reading
            self.dial_armed = False                       # resting wrist = middle, so don't start a hold yet
            self._cancel_dwell()
            await self.push_state()
            await self.push_dial()

    # -------------------------------------------------------------- wrist dial
    def on_roll(self, roll: float):                       # band thread
        self.last_roll = roll
        if not self.dial_on or self.loop is None:
            return
        if self.roll_zero is None:                        # however the wrist rests when the dial appears = middle
            self.roll_zero = roll
        r = (roll - self.roll_zero + 180) % 360 - 180     # tilt relative to that, wrapped to -180..180
        r = -r if self.flip else r
        idx = self._pick(r)
        if idx != self.dial_index:
            asyncio.run_coroutine_threadsafe(self.set_index(idx), self.loop)

    def _pick(self, r: float) -> int:
        """Tilt angle -> option index, with hysteresis: to cross an edge you must pass it by HYST degrees,
        so hand shake near an edge doesn't flip between options (and restart the hold).
        Answers: in at ±18°, out at ±10°. Skip: in past -40°, out at -32°."""
        c = self.dial_index
        return sum(r > (e + HYST if c <= k else e - HYST) for k, e in enumerate(EDGES))

    async def set_index(self, idx: int):
        if not self.dial_on or idx == self.dial_index:
            return
        self.dial_index, self.dial_armed = idx, True
        self._cancel_dwell()
        if self.dwell > 0:
            self._dwell_task = asyncio.create_task(self._hold(self.topic, idx))
        await self.push_dial()

    def _cancel_dwell(self):
        if self._dwell_task:
            self._dwell_task.cancel()
            self._dwell_task = None

    async def _hold(self, topic: str, idx: int):
        await asyncio.sleep(self.dwell)
        if self.dial_on and self.topic == topic and self.dial_index == idx:
            self._dwell_task = None                       # we are that task: don't let confirm cancel us
            await self.confirm_dial(how="hold")

    async def move_dial(self, delta: int):
        if self.dial_on:
            await self.set_index(max(0, min(len(EDGES), self.dial_index + delta)))

    async def heard(self, text: str):
        """The wearer's (redacted) words: if they give or change an answer by voice, show it on the HUD now.
        "change my appetite to normal" updates appetite even after it was answered. Negated words
        ("not good") are skipped rather than guessed."""
        named = [(m.start(), t) for t, rx in TOPIC_RE.items() for m in re.finditer(rx, text, re.I)]
        topic = max(named)[1] if named else self.topic
        if topic not in SAID_RE:
            return
        level = None
        for m in SAID_RE[topic].finditer(text):
            if not NEGATED_RE.search(text[:m.start()]):
                level = m.lastgroup                       # the last answer word wins: corrections come last
        if level is None or self.levels.get(topic) == level:
            return
        self.levels[topic] = level
        prov.log("transcript->hud", "answer level (one word)", "HUD on this Mac", topic=topic)
        if topic == self.topic and self.dial_on:          # answered by voice: the dial isn't needed any more
            self._cancel_dwell()
            self.dial_on = self.dial_armed = False
            await self.push_dial()
        await self.push_state()
        await self.hud.send({"type": "card", "id": topic, "title": topic, "body": level.replace("_", " "),
                             "meta": "said by voice"}, sticky=False)

    async def double_tap(self):
        """Double-tap: confirm the dial if one is showing; otherwise 'nothing to add, move on'
        (answers an open question without speech)."""
        if self.dial_on:
            return await self.confirm_dial()
        if not self.task or time.time() - self._last_move_on < 3:
            return
        self._last_move_on = time.time()
        prov.log("wrist->llm", "double-tap: nothing to add", "Gemini (Google)")
        await self.hud.send({"type": "gesture", "text": "✓ nothing to add"}, sticky=False)
        self.context.add_message({"role": "user", "content": "(The wearer double-tapped to move on: nothing to add.)"})
        await self.avatar("thinking")
        await self.task.queue_frames([LLMRunFrame()])

    async def confirm_dial(self, how: str = "double-tap"):
        if not (self.dial_on and self.topic and self.task):
            return
        self._cancel_dwell()
        topic, level = self.topic, dial_options(self.topic)[self.dial_index]
        skipped = level == SKIP
        self.levels[topic], self.dial_on, self.dial_armed = ("skipped" if skipped else level), False, False
        prov.log("wrist->llm", "dial answer (one word)", "Gemini (Google)", topic=topic, how=how)
        await self.hud.send({"type": "dial", "active": False, "topic": topic, "chosen": self.dial_index},
                            sticky=False)                 # HUD flashes the chosen option, then hides the dial
        await self.push_state()
        await self.hud.send({"type": "card", "id": topic, "title": topic,
                             "body": "skipped" if skipped else level.replace("_", " "),
                             "meta": f"chosen with a wrist tilt ({'held' if how == 'hold' else how})"},
                            sticky=False)
        self.context.add_message({"role": "user", "content":
                                  f"(With a wrist tilt, the wearer chose to skip {topic}.)" if skipped else
                                  f"(With a wrist tilt, the wearer chose '{level}' for {topic}.)"})
        await self.avatar("thinking")
        await self.task.queue_frames([LLMRunFrame()])

    # -------------------------------------------------------------- camera
    async def attach_frame(self, reason: str) -> bool:
        """Put the newest camera frame into Gemini's context. Returns False if there is no fresh frame."""
        if time.time() - self.last_attach < 8:            # one frame per "show", even if STT sends several finals
            return False
        jpeg = self.hud.latest_frame()
        if jpeg is None:
            await self.hud.send({"type": "caption", "who": "bot",
                                 "text": "(No camera frame: is the HUD open with the camera on?)"}, sticky=False)
            return False
        self.last_attach = time.time()
        await self.context.add_image_frame_message(format="image/jpeg", size=(640, 480), image=jpeg,
                                                   text=IMAGE_NOTE)
        prov.log("camera->llm", "one camera frame (jpeg, memory only, not stored)", "Gemini (Google)",
                 trigger=reason, topic=self.topic)
        if self.topic:
            self.shown[self.topic] = True
        await self.hud.send({"type": "flash"}, sticky=False)
        await self.hud.send({"type": "card", "id": f"cam-{self.last_attach:.0f}", "title": "Shown",
                             "body": "Gemini is looking…", "kind": "camera",
                             "meta": "📷 one frame → Gemini · not stored"}, sticky=False)
        await self.push_state()
        logger.info(f"Camera frame attached ({reason}, {len(jpeg) // 1024} KB)")
        return True

    async def show_now(self, reason: str):
        """Tap / P key: attach a frame and let the bot respond to it right away."""
        if self.task and await self.attach_frame(reason):
            self.context.add_message({"role": "user", "content": "(The wearer is showing you this.)"})
            await self.avatar("thinking")
            await self.task.queue_frames([LLMRunFrame()])

    async def on_hud_message(self, msg: dict):
        kind = msg.get("type")
        if kind == "begin":
            lang = msg.get("lang")
            if lang in LANG_EXTRA and not self.lang_chosen.done():
                self.lang_chosen.set_result(lang)
        elif kind == "show":
            await self.show_now("HUD key")
        elif kind == "dial_move":
            await self.move_dial(-1 if msg.get("delta", 0) < 0 else 1)
        elif kind == "confirm":
            await self.double_tap()


# ------------------------------------------------------------------ wrist flick = "I'm done talking"
from pipecat.frames.frames import VADUserStartedSpeakingFrame  # noqa: E402
from pipecat.turns.user_stop import BaseUserTurnStopStrategy  # noqa: E402


class FlickUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """Ends the wearer's turn the moment they flick their wrist, instead of waiting for silence.

    Used first in a FIRST-wins list, so Smart Turn still ends turns when nobody flicks. If the flick
    lands before Deepgram's last words arrive, the turn ends as soon as that text comes in (within
    PENDING_SECS), so Gemini never gets an empty turn.
    """
    PENDING_SECS = 3.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._text = ""
        self._open = False
        self._pending_at = 0.0

    async def handle_user_turn_started(self):
        self._text, self._open, self._pending_at = "", True, 0.0

    async def handle_user_turn_stopped(self):
        self._text, self._open, self._pending_at = "", False, 0.0

    async def process_frame(self, frame: Frame):
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._text += frame.text
            if self._pending_at and time.time() - self._pending_at < self.PENDING_SECS:
                await self._end("flick (text arrived)")
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._pending_at = 0.0                        # they kept talking after the flick: let it go

    async def flick(self) -> bool:
        """Called on the event loop when the band sees a flick. Returns True if it ended a turn."""
        if not self._open:
            return False
        if self._text:
            await self._end("flick")
            return True
        self._pending_at = time.time()                    # words still on their way from Deepgram
        return True

    async def _end(self, why: str):
        self._pending_at = 0.0
        logger.info(f"Turn ended by wrist {why}")
        await self.report_user_turn_stopped()


# ------------------------------------------------------------------ band: mic (inherited) + motion + gestures
class HudBandInput(nk.VexoBandInput):
    def __init__(self, session: Session, *args, gestures=True, flick_dps=350.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._s, self._gestures = session, gestures
        self._roll = self._pitch = 0.0
        self._sent = 0.0
        self._flick_dps, self._last_flick = flick_dps, 0.0
        self.flick_strategy = None                        # set by main(); see FlickUserTurnStopStrategy

    def _connect(self):
        super()._connect()                                # mic on, exactly as in the check-in bot
        b = self._band
        b.on("motion", self._on_motion)
        if self._gestures:                                # the pretend band taps by itself: ignore it in --sim
            # No single-tap action: the band reports a wrist flick's sharp stop as a "tap", so tap = camera
            # fired on every flick in the first real run. Camera = voice ("here's my lunch") or the HUD's P key.
            b.on("double-tap", lambda e: self._hop(self._s.double_tap()))
        b.configure(motion=25, events=True)               # low rate: motion shares the Bluetooth link with audio
        prov.log("band->mac", "wrist motion 25/s (tilt only)", "this Mac (Bluetooth)")

    def _hop(self, coro):
        if self._s.loop:
            asyncio.run_coroutine_threadsafe(coro, self._s.loop)

    async def _on_flick(self, spin: float):
        if await self.flick_strategy.flick():
            logger.info(f"Wrist flick ({spin:.0f}°/s): ending your turn")
            await self._s.hud.send({"type": "gesture", "text": "↯ done: answering"}, sticky=False)

    def _on_motion(self, m):
        if self._gestures and self.flick_strategy and m.gx is not None:
            spin = math.sqrt(m.gx * m.gx + m.gy * m.gy + m.gz * m.gz)   # degrees/second, any direction
            now = time.time()
            if spin > self._flick_dps and now - self._last_flick > 1.0:
                self._last_flick = now
                self._hop(self._on_flick(spin))
        if m.ax is None:
            return
        roll = math.degrees(math.atan2(m.ay, m.az))      # gravity only: tilt is reliable, turning is not (no compass)
        pitch = math.degrees(math.atan2(-m.ax, math.hypot(m.ay, m.az)))
        self._roll += 0.25 * (roll - self._roll)          # smooth out hand shake
        self._pitch += 0.25 * (pitch - self._pitch)
        self._s.on_roll(self._roll)
        now = time.time()
        if now - self._sent > 0.1:                        # 10 updates a second is plenty for the meter
            self._sent = now
            self._s.hud.send_threadsafe({"type": "wrist", "roll": self._roll, "pitch": self._pitch}, sticky=False)


# ------------------------------------------------------------------ pipeline processors
class CameraAndCaptions(FrameProcessor):
    """After redaction: shows the wearer's words on the HUD and attaches a frame when they show something."""

    def __init__(self, session: Session):
        super().__init__()
        self._s = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            await self._s.hud.send({"type": "caption", "who": "you", "text": frame.text}, sticky=False)
            await self._s.heard(frame.text)
            if SHOW_RE.search(frame.text):
                await self._s.attach_frame("spoken")
        await self.push_frame(frame, direction)


class AvatarState(FrameProcessor):
    """Pipecat's turn frames -> Sprout's state on the HUD. Placed at the end of the pipeline so it sees the
    wearer's turn (from the user aggregator) and the bot's audio start/stop (from the speaker output)."""

    STATES = {UserStartedSpeakingFrame: "listening", UserStoppedSpeakingFrame: "thinking",
              BotStartedSpeakingFrame: "speaking", BotStoppedSpeakingFrame: "idle"}

    def __init__(self, session: Session):
        super().__init__()
        self._s = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        state = self.STATES.get(type(frame))
        if state:
            await self._s.avatar(state)
        await self.push_frame(frame, direction)


class BotCaptions(FrameProcessor):
    """After the LLM: collects each reply and sends it to the HUD as one caption."""

    def __init__(self, session: Session):
        super().__init__()
        self._s, self._buf = session, []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self._buf.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame) and self._buf:
            text, self._buf = "".join(self._buf).strip(), []
            if text:
                await self._s.bot_said(text)
        await self.push_frame(frame, direction)


# ------------------------------------------------------------------ summary tool with `shown`
def summary_schema():
    s = copy.deepcopy(guardrails.SUMMARY_SCHEMA)          # the shared schema stays untouched for neurokind_bot.py
    for t in TOPICS:
        s["properties"][t]["properties"]["shown"] = {
            "type": "string",
            "description": "Only if the wearer showed a camera image for this topic: a few neutral words naming "
                           "the visible objects. Never people or faces. No health interpretation."}
    return s


def make_submit(session: Session):
    async def submit_summary(params):
        summary = dict(params.arguments)
        ok, issues, escalate = guardrails.check_summary(summary)
        for t in TOPICS:                                  # the camera line gets the same gate, plus no people
            shown = str((summary.get(t) or {}).get("shown", ""))
            if m := guardrails._DIAG_RE.search(shown):
                issues.append(f'diagnostic language in {t}.shown: "{m.group(0)}"')
            if m := PEOPLE_RE.search(shown):
                issues.append(f'{t}.shown describes a person ("{m.group(0)}"): name objects only')
        ok = not issues
        prov.log("llm->safety_gate", "summary json", "local safety gate", ok=ok, issues=issues)
        if not ok:
            logger.warning(f"Safety gate BLOCKED: {issues}")
            await params.result_callback({"status": "blocked", "fix": issues,
                                          "rule": "Report only what the wearer said or showed. No conditions, "
                                                  "no interpretation, no people."})
            return
        card = {"session": prov.session, "created": datetime.now().isoformat(timespec="seconds"),
                "non_diagnostic": True, "escalate": escalate, "source": "synthetic demo data",
                "camera_hud": True, **summary}
        out = HERE / "cards" / f"{prov.session}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(card, indent=2))
        (HERE / "cards" / "latest.json").write_text(json.dumps(card, indent=2))
        prov.log("safety_gate->card", "approved summary", f"cards/{out.name}", escalate=escalate)
        try:
            import trends
            trends.build()
        except Exception as e:                            # noqa: BLE001
            logger.warning(f"trends: {e}")
        # HUD: final levels + one card per topic
        session._cancel_dwell()
        session.topic, session.dial_on = None, False
        for t in TOPICS:
            v = summary.get(t) or {}
            session.levels[t] = v.get("level", "not_discussed")
            meta = f"📷 shown: {v['shown']}" if v.get("shown") else (f"“{v['quote']}”" if v.get("quote") else "")
            await session.hud.send({"type": "card", "id": t, "title": t, "body": v.get("reported", ""),
                                    "meta": meta, "kind": "camera" if v.get("shown") else "info"}, sticky=False)
        await session.push_dial()
        await session.push_state()
        if escalate:
            await session.avatar("concern", force=True)
            await session.hud.send({"type": "card", "id": "escalate", "title": "Flagged for care team",
                                    "body": "988 Suicide & Crisis Lifeline: call or text 988.", "kind": "alert"},
                                   sticky=False)
        logger.success(f"Card saved: {out}" + ("  ** ESCALATION FLAG **" if escalate else ""))
        await params.result_callback({"status": "saved"})
    return submit_summary


# ------------------------------------------------------------------ main
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("number", nargs="?", type=int)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--replay", default=str(KIT / "dashboard/speech/sample-speech.opus"))
    ap.add_argument("--stt", choices=["deepgram", "whisper"], default="deepgram")
    ap.add_argument("--max-minutes", type=float, default=4, help="hard stop so a forgotten run can't burn credits")
    ap.add_argument("--no-speaker", action="store_true")
    ap.add_argument("--flip-dial", action="store_true", help="if tilting right moves the dial left")
    ap.add_argument("--dwell", type=float, default=1.2,
                    help="seconds to hold a tilt before it confirms (0 = double-tap / Enter only)")
    ap.add_argument("--lang", choices=["ask", "en", "es"], default="ask",
                    help="check-in language; 'ask' = pick it on the HUD start screen (nothing speaks until then)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--turn-silence", type=float, default=1.5,
                    help="max seconds of silence before the bot answers (Pipecat default 3)")
    ap.add_argument("--flick-dps", type=float, default=350,
                    help="wrist turning speed (degrees/second) that counts as an 'I'm done' flick")
    a = ap.parse_args()
    if not a.sim and a.number is None:
        ap.error("give your band number, or --sim")

    hud = HudServer(port=a.port)
    await hud.start()
    session = Session(hud, flip_dial=a.flip_dial, dwell=a.dwell)
    session.loop = asyncio.get_running_loop()
    hud.on_message = session.on_hud_message

    if a.lang == "ask":
        logger.info(f"Open http://localhost:{a.port} and pick a language: nothing starts until you do.")
        await hud.send({"type": "await_lang"})
        session.lang = await session.lang_chosen
    else:
        session.lang = a.lang
    logger.info(f"Language: {session.lang}")

    band_in = HudBandInput(session, a.number, sim=a.sim, replay=a.replay, gestures=not a.sim, flick_dps=a.flick_dps)

    if a.stt == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService
        stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"], sample_rate=SAMPLE_RATE,
                                 settings=DeepgramSTTService.Settings(
                                     redact=["pci", "ssn", "numbers"], smart_format=True,
                                     language=session.lang,
                                     keyterm=(["appetite", "insomnia", "anxious", "stressed"]
                                              if session.lang == "en" else ["apetito", "estrés", "ansiedad"])))
        stt_dest = "Deepgram (cloud, redact=pci/ssn/numbers)"
        prov.log("mac->deepgram", "raw audio stream", "Deepgram (cloud)", note="cloud mode")
    else:
        from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
        stt = WhisperSTTServiceMLX(settings=WhisperSTTServiceMLX.Settings(model=MLXModel.LARGE_V3_TURBO_Q4))
        stt_dest = "local MLX Whisper (on device)"

    from pipecat.services.google.llm import GoogleLLMService
    llm = GoogleLLMService(api_key=os.environ["GOOGLE_API_KEY"],
                           settings=GoogleLLMService.Settings(
                               model=os.getenv("GEMINI_MODEL") or "gemini-4-argon",
                               thinking=GoogleLLMService.ThinkingConfig(
                                   thinking_level=os.getenv("GEMINI_THINKING", "low"))))
    llm.register_function("submit_summary", make_submit(session))

    voice = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    tts_model = os.getenv("ELEVENLABS_MODEL", "eleven_v4_turbo")
    if tts_model.startswith(("eleven_v3", "eleven_v4")):
        from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService as TTS
    else:
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService as TTS
    tts = TTS(api_key=os.environ["ELEVENLABS_API_KEY"], settings=TTS.Settings(model=tts_model, voice=voice))

    p = summary_schema()
    tool = FunctionSchema(name="submit_summary",
                          description="Submit the structured, non-diagnostic check-in summary for the clinician card.",
                          properties=p["properties"], required=p["required"])
    context = LLMContext(messages=[{"role": "system", "content": nk.SYSTEM + HUD_EXTRA + LANG_EXTRA[session.lang]}],
                         tools=ToolsSchema(standard_tools=[tool]))
    session.context = context
    # Smart Turn (local model) decides when the wearer is done. If it thinks they're mid-sentence it waits for
    # this much silence before answering anyway (Pipecat's default is 3 s; short check-in answers hit that often).
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import StrategyMode, UserTurnStrategies, UserTurnStrategiesParams
    flick = FlickUserTurnStopStrategy()
    band_in.flick_strategy = flick
    turns = UserTurnStrategies(start=nk.calm_turn_start(), stop=[flick, TurnAnalyzerUserTurnStopStrategy(
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=a.turn_silence)))])
    aggs = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(), user_turn_strategies=turns,
        user_mute_strategies=[nk.MuteWhileBotSpeaks()],    # no self-interruptions from the speakers / room
        # Pipecat's default needs EVERY stop strategy to agree; we want whichever decides first (flick or silence).
        user_turn_strategies_params=UserTurnStrategiesParams(stop_mode=StrategyMode.FIRST)))

    procs = [band_in, stt, nk.Redactor(stt_dest), CameraAndCaptions(session), aggs.user(), llm,
             BotCaptions(session), tts]
    if not a.no_speaker:
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
        procs.append(LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True)).output())
    procs.append(AvatarState(session))
    procs.append(aggs.assistant())

    task = PipelineTask(Pipeline(procs), params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, enable_metrics=True))
    session.task = task
    prov.log("llm->tts", "agent reply text", "ElevenLabs (cloud)")
    await session.push_state()

    context.add_message({"role": "user", "content": "(The wearer just started a check-in. Greet them briefly and ask how their day was.)"})
    await session.avatar("thinking")
    await task.queue_frames([LLMRunFrame()])

    async def hard_stop():
        await asyncio.sleep(a.max_minutes * 60)
        logger.warning(f"--max-minutes {a.max_minutes} reached: stopping (saves API credits)")
        await task.cancel()
    stopper = asyncio.create_task(hard_stop())
    try:
        await PipelineRunner().run(task)
    finally:
        stopper.cancel()
        await session.push_state(listening=False)
        await hud.stop()


if __name__ == "__main__":
    asyncio.run(main())
