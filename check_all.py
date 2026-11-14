"""One-shot check: are the three keys valid, and which Gemini 4 model is faster?
   python check_all.py            (prints no keys)"""
import json, os, subprocess, sys, time, urllib.request, urllib.error
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))


def call(url, headers, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={**headers, "Content-Type": "application/json"},
                                 method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:                       # noqa: BLE001
        return None, str(e).encode()


ok = True
print("== Deepgram")
s, b = call("https://api.deepgram.com/v1/projects", {"Authorization": "Token " + os.getenv("DEEPGRAM_API_KEY", "")})
print("  key OK" if s == 200 else f"  PROBLEM ({s}): {b[:200]!r}"); ok &= s == 200

print("== ElevenLabs")
voice = os.getenv("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"
t0 = time.time()
s, b = call(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?model_id=eleven_flash_v2_5",
            {"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")}, {"text": "Hi."})
if s == 200:
    print(f"  key OK, Text to Speech works ({len(b)} bytes of audio in {time.time()-t0:.1f}s)")
else:
    print(f"  PROBLEM ({s}): {b[:300]!r}"); ok = False
print("  (eleven_v4_turbo itself is checked when you run the bot: watch for a 401/403 on 'text-to-dialogue')")

print("== Gemini 4 (3 runs each, low thinking)")
subprocess.run([sys.executable, os.path.join(HERE, "bench_gemini.py")] + sys.argv[1:])
print("\nAll keys OK" if ok else "\nFix the PROBLEM lines above first")
