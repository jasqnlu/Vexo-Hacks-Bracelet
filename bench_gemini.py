"""Time a check-in-sized Gemini turn on each model, with your key from .env.
   python bench_gemini.py                      (argon vs hydrogen, 3 runs each)
   python bench_gemini.py gemini-4-carbon      (any model ids)"""
import os, sys, time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
models = sys.argv[1:] or ["gemini-4-argon", "gemini-4-hydrogen"]
SYSTEM = "You are a calm voice check-in companion. Reply in one short spoken sentence, then ask about mood."
USER = "I slept maybe four hours, kept waking up around three."

for m in models:
    times = []
    for i in range(3):
        t0 = time.time(); first = None; text = ""
        try:
            for chunk in client.models.generate_content_stream(
                    model=m, contents=USER,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=2048,
                                                       thinking_config=types.ThinkingConfig(thinking_level="low"))):
                if chunk.text:
                    first = first or time.time() - t0
                    text += chunk.text
            times.append(first or float("nan"))
        except Exception as e:
            print(f"{m}: ERROR {str(e)[:150]}"); break
    if times:
        print(f"{m}: first words after {', '.join(f'{t:.2f}s' for t in times)}  ->  {text.strip()[:90]}")
