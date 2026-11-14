"""Pinch probe: can the band feel a thumb-index pinch? A 1-minute guided test, separate from the bot.

  python pinch_probe.py 1            real band 1 (double-tap it first; close the HUD bot / kit dashboard)
  python pinch_probe.py 1 --no-mic   same, with the mic off (to see if audio + fast motion fight for Bluetooth)
  python pinch_probe.py --sim        just checks the script runs (the pretend band can't pinch)

Streams the accelerometer at 416 samples/s with the mic on (the Bluetooth-safe rate with audio), then
walks you through three phases: hold still, pinch 10 times, then do everything that is NOT a pinch
(wave, tap the table, type, talk). It measures "jerk" (how sharply acceleration changes between two
samples: a pinch is a tiny sharp click, arm movement is slow) and prints whether pinches stand out.
Raw samples go to probe_runs/ (gitignored) so a threshold can be tuned later. Nothing else is saved.
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KIT = HERE / "vexo-open-kit"
sys.path.insert(0, str(KIT / "python"))
from vexo_open.sync import Band  # noqa: E402

RATE = 416
PHASES = [("still", 5, "Hold your wrist STILL, resting on the table or your lap."),
          ("pinch", 15, "PINCH thumb + index finger firmly, 10 times, about once a second. Arm still."),
          ("other", 15, "Now NOT pinching: wave your arm, tap the table, type, clap once, talk.")]
REFRACTORY = 0.20                                   # one pinch = one event, even if it rings for a bit

samples = deque()                                   # (t_s, n, ax, ay, az) filled by the band thread
audio_seq = {"count": 0, "gaps": 0, "last": None}


def on_motion(m):                                   # runs 416x a second: collect only
    if m.ax is not None:
        samples.append((m.t / 1000.0, m.n, m.ax, m.ay, m.az))


def on_audio(a):
    s = getattr(a, "seq", None)
    audio_seq["count"] += 1
    if s is not None and audio_seq["last"] is not None and (s - audio_seq["last"]) % 65536 != 1:
        audio_seq["gaps"] += 1
    audio_seq["last"] = s


def events(t, jerk, thr):
    """Times where jerk crosses thr, at most one per REFRACTORY seconds."""
    out, last = [], -1e9
    for ti, j in zip(t, jerk):
        if j > thr and ti - last > REFRACTORY:
            out.append(ti)
            last = ti
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("number", nargs="?", type=int)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-mic", action="store_true")
    a = ap.parse_args()
    if not a.sim and a.number is None:
        ap.error("give your band number, or --sim")

    band = Band.connect(42, transport="sim") if a.sim else Band.connect(a.number)
    info = band.info
    print(f"Band firmware {info.firmware}  features {list(info.features or [])}")
    lim = band.motion_limits()
    print(f"Fast motion limits on this link: accel-only {lim.accel}/s, both {lim.both}/s ({lim.transport})")
    if not lim.accel or lim.accel < RATE:
        print("This band can't stream fast motion (needs firmware 1.1). Pinch detection is not possible: stop here.")
        band.disconnect()
        return

    band.set_motion(accel_range=4)                  # 4 g: a table tap doesn't clip, still 1 mg steps
    band.on("motion", on_motion)
    band.on("audio", on_audio)
    if not a.no_mic:
        band.configure(audio=True)                  # same load as the real bot: mic + motion together
    band.configure(motion=RATE, sensors="accel")
    band.wait(1.0)                                  # let the stream settle
    samples.clear()

    marks = []                                      # (phase, t_start, t_end) in band time
    try:
        for name, secs, say in PHASES:
            print(f"\n=== {name.upper()} ({secs} s) === {say}")
            for i in (3, 2, 1):
                print(f"  starting in {i}…", end="\r", flush=True)
                time.sleep(1)
            print("  GO" + " " * 20)
            t0 = samples[-1][0] if samples else 0.0
            end = time.time() + secs
            while time.time() < end:
                time.sleep(0.1)
                recent = list(samples)[-60:]         # ~150 ms: a live meter so you can see it react
                if len(recent) > 2:
                    acc = np.array([r[2:] for r in recent])
                    j = np.linalg.norm(np.diff(acc, axis=0), axis=1).max() * 1000   # milli-g per sample
                    bar = "#" * min(60, int(j / 5))
                    print(f"  jerk {j:7.1f} mg |{bar:<60}| {end - time.time():4.1f}s", end="\r", flush=True)
            print()
            marks.append((name, t0, samples[-1][0] if samples else t0))
    finally:
        band.configure(motion=False, audio=False)
        band.disconnect()

    # ------------------------------------------------------------------ analysis
    arr = np.array(samples)
    if len(arr) < 100:
        print("Too few samples arrived. Is the band close by? Try again.")
        return
    t, n, acc = arr[:, 0], arr[:, 1], arr[:, 2:]
    jerk = np.r_[0.0, np.linalg.norm(np.diff(acc, axis=0), axis=1) * 1000]
    lost = int(np.sum(np.clip(np.diff(n) - 1, 0, None)))
    dur = t[-1] - t[0]
    print(f"\nGot {len(arr)} samples in {dur:.1f} s ({len(arr) / max(dur, 1e-9):.0f}/s), lost {lost}. "
          f"Audio packets {audio_seq['count']}, audio gaps {audio_seq['gaps']}.")

    out = HERE / "probe_runs"
    out.mkdir(exist_ok=True)
    f = out / f"pinch_{time.strftime('%H%M%S')}.npz"
    np.savez(f, t=t, n=n, acc=acc, marks=np.array([(m[1], m[2]) for m in marks]),
             names=np.array([m[0] for m in marks]))
    print(f"Raw samples saved: {f}")

    ph = {name: (t >= t0) & (t <= t1) for name, t0, t1 in marks}
    still, pinch, other = (jerk[ph[k]] for k in ("still", "pinch", "other"))
    print(f"\njerk (mg)        median    99th pct   max")
    for k, v in (("still", still), ("pinch", pinch), ("other", other)):
        if len(v):
            print(f"  {k:<8}  {np.median(v):9.1f} {np.percentile(v, 99):10.1f} {v.max():8.1f}")

    # Two candidate limits: just above everything in "other" (zero false triggers in this test),
    # and halfway (on a log scale) between the still floor and the pinch peaks.
    floor = np.percentile(still, 99.5) if len(still) else 1.0
    cands = {"no false triggers": other.max() * 1.1 if len(other) else floor * 3,
             "middle": float(np.sqrt(floor * np.sort(pinch)[-10])) if len(pinch) >= 10 else floor * 3}
    print("\nlimit                      pinches found (of 10)   false triggers in 'other'")
    best = None
    for label, thr in cands.items():
        p = len(events(t[ph["pinch"]], jerk[ph["pinch"]], thr))
        o = len(events(t[ph["other"]], jerk[ph["other"]], thr))
        print(f"  {label:<18} {thr:6.1f} mg   {p:>3}                     {o:>3}")
        if best is None or (p - 3 * o) > best[1]:
            best = (label, p - 3 * o, thr, p, o)

    label, _, thr, p, o = best
    print("\nVERDICT:", end=" ")
    if p >= 8 and o == 0:
        print(f"YES. Pinches stand out ({p}/10, no false triggers) at {thr:.0f} mg. Worth adding as a confirm.")
    elif p >= 7 and o <= 2:
        print(f"MAYBE. {p}/10 pinches, {o} false triggers at {thr:.0f} mg. Usable only behind a flag, "
              f"with hold-to-confirm kept as the main way.")
    else:
        print(f"NO. Best was {p}/10 pinches with {o} false triggers. Keep hold-to-confirm; skip pinch.")
    if lost or audio_seq["gaps"]:
        print(f"Note: Bluetooth dropped {lost} motion samples and {audio_seq['gaps']} audio gaps. "
              f"If audio gaps > 0, fast motion would hurt transcription in the bot.")


if __name__ == "__main__":
    main()
