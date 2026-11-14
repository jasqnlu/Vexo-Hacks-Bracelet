"""NeuroKind check-in bot: Vexo band -> Pipecat -> (Deepgram | local Whisper) -> redaction -> Gemini -> ElevenLabs.

  python3 neurokind_bot.py --sim                         pretend band speaking the kit's sample
  python3 neurokind_bot.py 42                            real band 42 (double-tap it first)
  python3 neurokind_bot.py 42 --stt whisper              private mode: raw audio + transcript stay on the Mac
  python3 neurokind_bot.py 42 --wait-tap                 check-in starts on a double-tap

Keys in .env (from a teammate's account): DEEPGRAM_API_KEY, GOOGLE_API_KEY, ELEVENLABS_API_KEY,
optional ELEVENLABS_VOICE_ID, GEMINI_MODEL (ask DeepMind at the event for the Gemini 4 id).

Outputs: cards/<session>.json (the clinician card), provenance.jsonl (every hop).
"""
import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

HERE = Path(__file__).resolve().parent
KIT = HERE / "vexo-open-kit"
sys.path.insert(0, str(KIT / "python"))
from vexo_open import audio as vexo_audio            # noqa: E402
from vexo_open.audio import PcmDecoder               # noqa: E402
from vexo_open.sync import Band                      # noqa: E402

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema       # noqa: E402
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema             # noqa: E402
from pipecat.audio.vad.silero import SileroVADAnalyzer                    # noqa: E402
from pipecat.frames.frames import (EndFrame, Frame, InputAudioRawFrame,   # noqa: E402
                                   LLMRunFrame, StartFrame, TranscriptionFrame, TTSSpeakFrame)
from pipecat.pipeline.pipeline import Pipeline                            # noqa: E402
from pipecat.pipeline.runner import PipelineRunner                        # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask            # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext         # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (       # noqa: E402
    LLMContextAggregatorPair, LLMUserAggregatorParams)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.transports.base_input import BaseInputTransport              # noqa: E402
from pipecat.transports.base_transport import TransportParams             # noqa: E402

import guardrails                                                         # noqa: E402

load_dotenv(HERE / ".env")
logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))   # LOG_LEVEL=DEBUG for Pipecat's full output
SAMPLE_RATE = 16000
prov = guardrails.Provenance(HERE / "provenance.jsonl")


# ------------------------------------------------------------------ band input
class VexoBandInput(BaseInputTransport):
    """Turns the band's Opus packets into Pipecat InputAudioRawFrames (16 kHz mono PCM)."""

    def __init__(self, band_number=None, sim=False, replay=None, wait_tap=False, **kwargs):
        super().__init__(TransportParams(audio_in_enabled=True, audio_in_sample_rate=SAMPLE_RATE,
                                         audio_in_channels=1), **kwargs)
        self._number, self._sim, self._replay, self._wait_tap = band_number, sim, replay, wait_tap
        self._band = None
        self._decoder = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        loop = self.get_event_loop()
        await loop.run_in_executor(None, self._connect)
        await self.set_transport_ready(frame)

    def _connect(self):
        logger.info("Loading Opus decoder...")
        self._decoder = PcmDecoder()
        self._replay_left = None
        if self._sim:
            packets = vexo_audio.ogg_packets(self._replay)
            self._replay_left = len(packets)               # the pretend band loops: play the file once, then silence
            self._band = Band.connect(42, transport="sim", audio_packets=packets)
        else:
            self._band = Band.connect(self._number)
        logger.info(f"Connected to {self._band.name}")
        if self._wait_tap:
            tapped = threading.Event()
            self._band.on("double-tap", lambda e: tapped.set())
            self._band.configure(events=True)
            logger.info("Double-tap the band to start the check-in...")
            tapped.wait()
        loop = self.get_event_loop()

        def on_audio(a):
            if self._replay_left is not None:
                if self._replay_left <= 0:
                    pcm = bytes(640)                       # 20 ms of silence: the "wearer" waits for the bot
                    f = InputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1)
                    asyncio.run_coroutine_threadsafe(self.push_audio_frame(f), loop)
                    return
                self._replay_left -= 1
            pcm = self._decoder.feed(a)
            if pcm:
                f = InputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1)
                asyncio.run_coroutine_threadsafe(self.push_audio_frame(f), loop)

        self._band.on("audio", on_audio)
        self._band.configure(audio=True)                       # band light: solid blue = mic on
        prov.log("band->mac", "opus audio 16kHz (raw)", "this Mac (Bluetooth)", band=self._band.name)

    async def _close_band(self):
        if self._band:
            band, self._band = self._band, None
            def close():
                try:
                    band.configure(audio=False)
                    band.disconnect()
                except Exception as e:                         # noqa: BLE001
                    logger.warning(f"band close: {e}")
            await self.get_event_loop().run_in_executor(None, close)

    async def stop(self, frame: EndFrame):
        await self._close_band()
        await super().stop(frame)

    async def cancel(self, frame):
        await self._close_band()
        await super().cancel(frame)


# ------------------------------------------------------------------ redaction
class Redactor(FrameProcessor):
    """Sits between STT and the LLM: names/phones/addresses/dates never reach Gemini."""

    def __init__(self, stt_dest: str):
        super().__init__()
        self.stt_dest = stt_dest

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            clean, spans = guardrails.redact(frame.text)
            prov.log("audio->transcript", "speech-to-text", self.stt_dest)
            prov.log("transcript->llm", "redacted transcript", "Gemini (Google)",
                     redactions=[s["type"] for s in spans])
            if spans:
                logger.info(f"Redacted {len(spans)} item(s): {clean}")
            frame.text = clean
        await self.push_frame(frame, direction)


# ------------------------------------------------------------------ prompt + tool
SYSTEM = """You are NeuroKind, a calm voice companion running a short daily wellbeing check-in
through a wrist band. Your reply is spoken aloud, so keep each turn to one or two short sentences,
plain words, no lists, no emoji.

Ask about these four, one at a time, in this order: sleep, mood, stress, appetite. Ask one gentle
follow-up only if an answer is very short. Accept "skip". Never give medical advice, never diagnose,
never name a condition, never interpret what they said. You only listen and record.

Some words in the transcript look like [NAME] or [PHONE]: they were removed for privacy. Do not ask
about them or repeat them.

When all four are covered, ask if they want a clinician to follow up, then call submit_summary.
If submit_summary returns blocked, fix exactly the listed issues and call it again. When it returns
saved, thank them in one sentence and say goodbye.

If the wearer mentions wanting to hurt themselves, respond with warmth, tell them you are flagging
this for their care team now and that they can call or text 988 any time, then call submit_summary."""


def make_summary_tool():
    p = guardrails.SUMMARY_SCHEMA
    return FunctionSchema(name="submit_summary",
                          description="Submit the structured, non-diagnostic check-in summary for the clinician card.",
                          properties=p["properties"], required=p["required"])


async def submit_summary(params):
    summary = dict(params.arguments)
    ok, issues, escalate = guardrails.check_summary(summary)
    prov.log("llm->safety_gate", "summary json", "local safety gate", ok=ok, issues=issues)
    if not ok:
        logger.warning(f"Safety gate BLOCKED: {issues}")
        await params.result_callback({"status": "blocked", "fix": issues,
                                      "rule": "Report only what the wearer said. No conditions, no interpretation."})
        return
    from datetime import datetime
    card = {"session": prov.session, "created": datetime.now().isoformat(timespec="seconds"),
            "non_diagnostic": True, "escalate": escalate, "source": "synthetic demo data", **summary}
    out = HERE / "cards" / f"{prov.session}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(card, indent=2))
    (HERE / "cards" / "latest.json").write_text(json.dumps(card, indent=2))
    prov.log("safety_gate->card", "approved summary", f"cards/{out.name}", escalate=escalate)
    try:
        import trends
        trends.build()                                   # refresh history + overview for the card page
    except Exception as e:                               # noqa: BLE001
        logger.warning(f"trends: {e}")
    logger.success(f"Card saved: {out}" + ("  ** ESCALATION FLAG **" if escalate else ""))
    await params.result_callback({"status": "saved"})


# ------------------------------------------------------------------ main
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("number", nargs="?", type=int)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--replay", default=str(KIT / "dashboard/speech/sample-speech.opus"))
    ap.add_argument("--stt", choices=["deepgram", "whisper"], default="deepgram")
    ap.add_argument("--wait-tap", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=4, help="hard stop so a forgotten run can't burn credits")
    ap.add_argument("--no-speaker", action="store_true", help="skip local audio output (no pyaudio needed)")
    a = ap.parse_args()
    if not a.sim and a.number is None:
        ap.error("give your band number, or --sim")

    band_in = VexoBandInput(a.number, sim=a.sim, replay=a.replay, wait_tap=a.wait_tap)

    if a.stt == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService
        stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"], sample_rate=SAMPLE_RATE,
                                 settings=DeepgramSTTService.Settings(
                                     redact=["pci", "ssn", "numbers"], smart_format=True,
                                     keyterm=["appetite", "insomnia", "anxious", "stressed"]))
        stt_dest = "Deepgram (cloud, redact=pci/ssn/numbers)"
        prov.log("mac->deepgram", "raw audio stream", "Deepgram (cloud)", note="cloud mode")
    else:
        from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
        stt = WhisperSTTServiceMLX(settings=WhisperSTTServiceMLX.Settings(model=MLXModel.LARGE_V3_TURBO_Q4))
        stt_dest = "local MLX Whisper (on device)"

    from pipecat.services.google.llm import GoogleLLMService
    # Voice needs fast replies: ask for low thinking explicitly (Gemini 4 defaults to medium).
    llm = GoogleLLMService(api_key=os.environ["GOOGLE_API_KEY"],
                           settings=GoogleLLMService.Settings(
                               model=os.getenv("GEMINI_MODEL") or "gemini-4-argon",
                               thinking=GoogleLLMService.ThinkingConfig(
                                   thinking_level=os.getenv("GEMINI_THINKING", "low"))))
    llm.register_function("submit_summary", submit_summary)

    voice = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    tts_model = os.getenv("ELEVENLABS_MODEL", "eleven_v4_turbo")
    if tts_model.startswith(("eleven_v3", "eleven_v4")):
        # v3/v4 are only served by the Text-to-Dialogue WebSocket.
        from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService as TTS
    else:
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService as TTS   # flash / turbo v2.5
    tts = TTS(api_key=os.environ["ELEVENLABS_API_KEY"], settings=TTS.Settings(model=tts_model, voice=voice))
    logger.info(f"TTS: {tts_model}")

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM}],
                         tools=ToolsSchema(standard_tools=[make_summary_tool()]))
    aggs = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))

    procs = [band_in, stt, Redactor(stt_dest), aggs.user(), llm, tts]
    if not a.no_speaker:
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
        speaker = LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True))
        procs.append(speaker.output())
    procs.append(aggs.assistant())

    task = PipelineTask(Pipeline(procs), params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, enable_metrics=True))
    prov.log("llm->tts", "agent reply text", "ElevenLabs (cloud)")

    # Greet first: the bot opens the check-in.
    context.add_message({"role": "user", "content": "(The wearer just started a check-in. Greet them briefly and ask about sleep.)"})
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


if __name__ == "__main__":
    asyncio.run(main())
