"""Generate the HUD's small sound effects ONCE with the ElevenLabs Sound Effects API.

    .venv/bin/python make_sounds.py          # makes only missing files in sounds/
    .venv/bin/python make_sounds.py --force  # regenerate all (costs credits)
    .venv/bin/python make_sounds.py --local  # synthesize with numpy instead (no API, no credits)

If the API answers 401/403 (e.g. the key lacks the sound_generation permission),
the script falls back to --local automatically.

Reads ELEVENLABS_API_KEY from .env; never prints it.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
OUT = HERE / "sounds"

SOUNDS = {
    "confirm": ("Soft gentle two-note wooden marimba chime, rising, warm and calm, "
                "clean, no reverb tail, UI confirmation sound", 0.6),
    "skip": ("Soft muted single bubble pop, gentle and quiet, warm, UI sound", 0.5),
    "streak": ("Gentle sparkling shimmer, soft warm chimes, calm and celebratory, "
               "delicate, wellbeing app reward sound", 1.2),
    "stop": ("Soft short descending airy whoosh, gentle and quiet, UI sound", 0.5),
}
# note: the API's minimum duration is 0.5 s, so "skip" uses 0.5 instead of 0.4


SR = 44100


def _note(freq, dur, start, total, decay=6.0, partials=((1, 1.0), (4, 0.25), (10, 0.06))):
    """A soft marimba-like note: a few sine partials, exponential decay, 5 ms attack."""
    t = np.arange(int(dur * SR)) / SR
    env = np.exp(-decay * t) * np.minimum(1, t / 0.005)
    tone = sum(a * np.exp(-decay * (k ** 0.5 - 1) * t) * np.sin(2 * np.pi * freq * k * t) for k, a in partials)
    out = np.zeros(int(total * SR))
    i = int(start * SR)
    seg = (tone * env)[: len(out) - i]
    out[i:i + len(seg)] += seg
    return out


def synth(name):
    if name == "confirm":                     # two rising notes, E5 -> A5
        total = 0.6
        x = _note(659.25, 0.45, 0.0, total, 9) + _note(880.0, 0.5, 0.12, total, 8)
    elif name == "skip":                      # soft low pop with a quick pitch drop
        total = 0.4
        t = np.arange(int(total * SR)) / SR
        f = 180 + 320 * np.exp(-t / 0.025)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.05) * np.minimum(1, t / 0.003)
    elif name == "streak":                    # 5 quick rising bell notes + light shimmer
        total = 1.2
        notes = [783.99, 987.77, 1174.66, 1318.51, 1567.98]   # G5 B5 D6 E6 G6
        x = sum(_note(f, 0.9, i * 0.09, total, 4.5, ((1, 1.0), (2.76, 0.2), (5.4, 0.05)))
                for i, f in enumerate(notes))
        t = np.arange(len(x)) / SR
        x *= 1 + 0.15 * np.sin(2 * np.pi * 7 * t)              # gentle tremolo shimmer
    elif name == "stop":                      # soft descending sweep: sine + lowpassed noise
        total = 0.5
        t = np.arange(int(total * SR)) / SR
        f = 900 * np.exp(-t * 3.2)                              # 900 -> ~180 Hz
        sine = np.sin(2 * np.pi * np.cumsum(f) / SR)
        noise = np.random.default_rng(7).standard_normal(len(t))
        a = 0.04                                                # one-pole lowpass ~300 Hz
        lp = np.zeros_like(noise); acc = 0.0
        for i, v in enumerate(noise):
            acc += a * (v - acc); lp[i] = acc
        lp /= np.abs(lp).max()
        env = np.sin(np.pi * np.minimum(1, t / total)) ** 1.5
        x = (0.6 * sine + 0.5 * lp) * env
    else:
        raise ValueError(name)
    n = len(x)
    fade_in, fade_out = int(0.004 * SR), int(0.03 * SR)
    x[:fade_in] *= np.linspace(0, 1, fade_in)
    x[-fade_out:] *= np.linspace(1, 0, fade_out)
    x = x / np.abs(x).max() * 10 ** (-6 / 20)                   # peak -6 dBFS
    return (x * 32767).astype(np.int16)


def encode_mp3(pcm, path):
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "x.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav),
                            "-codec:a", "libmp3lame", "-b:a", "96k", str(path)], check=True)
            return
    import av                                  # fallback: PyAV
    out = av.open(str(path), "w", format="mp3")
    st = out.add_stream("mp3", rate=SR)
    st.layout = "mono"
    frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = SR
    for pkt in st.encode(frame):
        out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()


def make_local(name, path):
    encode_mp3(synth(name), path)
    print(f"wrote {path.relative_to(HERE)} (local synth, {path.stat().st_size} B)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate files that already exist")
    ap.add_argument("--local", action="store_true", help="synthesize with numpy, no API call")
    args = ap.parse_args()

    load_dotenv(HERE / ".env")
    key = os.environ.get("ELEVENLABS_API_KEY")
    local = args.local
    if not key and not local:
        print("ELEVENLABS_API_KEY missing from .env: using --local")
        local = True
    OUT.mkdir(exist_ok=True)

    for name, (prompt, dur) in SOUNDS.items():
        path = OUT / f"{name}.mp3"
        if path.exists() and not args.force:
            print(f"skip {name}: exists ({path.stat().st_size} B)")
            continue
        if local:
            make_local(name, path)
            continue
        r = requests.post(
            "https://api.elevenlabs.io/v1/sound-generation",
            headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
            json={"text": prompt, "duration_seconds": dur, "prompt_influence": 0.5},
            timeout=60,
        )
        if r.status_code in (401, 403):
            print(f"{name}: API refused ({r.status_code}), falling back to --local for the rest")
            local = True
            make_local(name, path)
            continue
        if r.status_code != 200:
            sys.exit(f"{name}: API error {r.status_code}: {r.text[:300]}")
        path.write_bytes(r.content)
        print(f"wrote {path.relative_to(HERE)} ({len(r.content)} B)")


if __name__ == "__main__":
    main()
