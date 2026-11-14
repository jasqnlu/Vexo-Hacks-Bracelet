"""NeuroKind on-device audio layer: Vexo band -> local PCM -> quality checks -> local Whisper.

Raw audio never leaves this machine. Only (redacted) text is handed to the agent.

  python3 band_capture.py 42                 real band 42 (double-tap it first)
  python3 band_capture.py --sim              pretend band speaking the kit's sample file
  python3 band_capture.py --sim --replay my.ogg
  python3 band_capture.py 42 --no-whisper    capture + checks only (no mlx-whisper needed)

Flow: double-tap starts a check-in turn -> mic on (band glows blue) -> record until
~1.2 s of silence after speech (or --max-seconds) -> mic off -> quality checks ->
transcribe locally -> print JSON {ok, reason, text, audio_stats}.
"""
import argparse, json, sys, threading, time, wave
from pathlib import Path

KIT = Path(__file__).resolve().parent / "vexo-open-kit"
sys.path.insert(0, str(KIT / "python"))
from vexo_open.sync import Band            # noqa: E402
from vexo_open import audio                # noqa: E402
from vexo_open.audio import PcmDecoder     # noqa: E402

import numpy as np                         # noqa: E402
from loguru import logger                  # noqa: E402
logger.remove(); logger.add(sys.stderr, level="WARNING")   # hide Pipecat's debug lines

FRAME_S = 0.02                             # one band packet = 20 ms


def frame_db(pcm: bytes) -> float:
    return audio.level_db(pcm)


class Turn:
    """Collects PCM for one spoken turn. Speech detection = Silero VAD (robust in a noisy room),
    not loudness: at a hackathon the background can be almost as loud as the voice."""

    def __init__(self, max_seconds=20.0, end_silence=1.2, threshold=0.5):
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        self.vad = SileroVADAnalyzer()
        self.vad.set_sample_rate(16000)                          # must be set this way, or the model never loads its rate
        self.chunk = self.vad.num_frames_required() * 2          # bytes per Silero window (512 samples = 32 ms)
        self.threshold = threshold
        self.frames, self.dbs, self.speech_flags = [], [], []
        self._buf = b""
        self.max_frames = int(max_seconds / FRAME_S)
        self.end_windows = int(end_silence / 0.032)
        self.speech_windows = 0
        self.speech_started = False
        self.silent_run = 0
        self.noise_db = None
        self.margin_db = 0.0
        self.done = threading.Event()

    def add(self, pcm: bytes):
        if self.done.is_set() or not pcm:
            return
        self.frames.append(pcm); self.dbs.append(frame_db(pcm))
        self._buf += pcm
        while len(self._buf) >= self.chunk:
            win, self._buf = self._buf[:self.chunk], self._buf[self.chunk:]
            speech = float(np.asarray(self.vad.voice_confidence(win)).squeeze()) >= self.threshold
            self.speech_flags.append(speech)
            if speech:
                self.speech_windows += 1
                self.speech_started, self.silent_run = True, 0
            elif self.speech_started:
                self.silent_run += 1
                if self.silent_run >= self.end_windows:
                    self.done.set()
        if len(self.frames) >= self.max_frames:
            self.done.set()

    @property
    def pcm(self) -> bytes:
        return b"".join(self.frames)


def quality(turn: Turn, lost: int, min_speech_s=0.8):
    """Return (ok, reason, stats). reason drives the re-prompt the agent speaks."""
    pcm = turn.pcm
    x = audio.to_float(pcm) if pcm else np.zeros(0, np.float32)
    flags = turn.speech_flags
    quiet = [d for d, i in zip(turn.dbs, range(len(turn.dbs))) if not flags[min(len(flags) - 1, int(i * FRAME_S / 0.032))]] if flags else turn.dbs
    noise = float(np.percentile(quiet, 50)) if quiet else -100.0
    speech_seconds = turn.speech_windows * 0.032
    stats = {
        "seconds": round(len(x) / 16000, 2),
        "speech_seconds": round(speech_seconds, 2),
        "noise_floor_db": round(noise, 1),
        "speech_level_db": round(float(np.percentile(turn.dbs, 95)), 1) if turn.dbs else -100.0,
        "clipped_fraction": round(float(np.mean(np.abs(x) > 0.99)), 4) if len(x) else 0.0,
        "packets_lost": lost,
    }
    if stats["speech_seconds"] < min_speech_s:
        return False, "too_short", stats
    if stats["speech_level_db"] < -45:
        return False, "too_quiet", stats
    if stats["clipped_fraction"] > 0.01:
        return False, "clipping", stats
    if lost > 0.1 * max(1, len(turn.frames)):
        return False, "bad_link", stats
    return True, "ok", stats


REPROMPT = {
    "too_short": "Sorry, I didn't catch that. Could you say a bit more?",
    "too_quiet": "That was a little quiet. Could you bring the band closer and try again?",
    "clipping": "That was a bit loud. Could you hold the band a little further away?",
    "bad_link": "The connection dropped for a moment. Could you say that again?",
}


def transcribe(pcm: bytes, model: str) -> str:
    import mlx_whisper                                   # Mac only
    x = audio.to_float(audio.normalise(pcm))
    return mlx_whisper.transcribe(x, path_or_hf_repo=model, language="en")["text"].strip()


def save_wav(path, pcm):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)


def capture_turn(band, decoder, max_seconds):
    turn = Turn(max_seconds=max_seconds)
    lost0 = decoder.lost
    handler = band.on("audio", lambda a: turn.add(decoder.feed(a)))
    band.configure(audio=True)                           # band light: solid blue
    turn.done.wait(max_seconds + 2)
    band.configure(audio=False)
    band.off("audio", handler)
    return turn, decoder.lost - lost0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("number", nargs="?", type=int)
    p.add_argument("--sim", action="store_true")
    p.add_argument("--replay", default=str(KIT / "dashboard/speech/sample-speech.opus"))
    p.add_argument("--max-seconds", type=float, default=20)
    p.add_argument("--model", default="mlx-community/whisper-small-mlx")
    p.add_argument("--no-whisper", action="store_true")
    p.add_argument("--wait-tap", action="store_true", help="start the turn on a double-tap")
    a = p.parse_args()

    print("Loading Opus decoder...", file=sys.stderr)
    decoder = PcmDecoder()
    if a.sim:
        band = Band.connect(42, transport="sim", audio_packets=audio.ogg_packets(a.replay))
    else:
        band = Band.connect(a.number)

    if a.wait_tap:
        tapped = threading.Event()
        band.on("double-tap", lambda e: tapped.set())
        band.configure(events=True)
        print("Double-tap the band to start your check-in...", file=sys.stderr)
        tapped.wait()

    t0 = time.time()
    turn, lost = capture_turn(band, decoder, a.max_seconds)
    band.disconnect()
    ok, reason, stats = quality(turn, lost)
    save_wav("last_turn.wav", turn.pcm)                  # stays on device; for debugging only
    out = {"ok": ok, "reason": reason, "reprompt": REPROMPT.get(reason), "audio_stats": stats,
           "capture_s": round(time.time() - t0, 2), "text": None}
    if ok and not a.no_whisper:
        t1 = time.time()
        out["text"] = transcribe(turn.pcm, a.model)
        out["asr_s"] = round(time.time() - t1, 2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
