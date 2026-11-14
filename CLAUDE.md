# NeuroKind — VexoHacks build context

Read this whole file before changing anything. It is the shared context for this repo.

## The event
- **VexoHacks**: one-day AI wearables hackathon by Vexo at Y Combinator, San Francisco. Sat **Nov 14 2026, 9 AM–9 PM PT**.
- **Feature freeze ~5 PM**, then rehearse the demo and record a backup video. 30-minute rule: if stuck on something for 30 min, cut it or fall back.
- Prizes: Best/2nd/3rd overall + sponsor tracks. We are targeting:
  - **Deepgram — Best Voice Use Case on Pipecat** (main sponsor target)
  - **Google DeepMind — Best Use of Gemini 4**
  - **ElevenLabs — Best ElevenLabs Project**
  - (Stripe "Best Economic Agent" deliberately skipped: doesn't fit.)

## Team
- **Jason** (me): owns everything between the wearer's voice and the agent, plus the guardrails on the output: band audio, speech detection, quality checks, transcription, redaction, provenance log, safety gate, trends, clinician dashboard. High school junior; ML/CV background. Under 18, so API keys come from teammates' accounts.
- **Alyshah**: project owner (company: NeuroKind, clinical AI screening). Owns agent prompts and the clinician summary card design.
- Machine: MacBook (Apple Silicon M5, 16 GB). Python 3.12 venv in `.venv` (Homebrew Python 3.14 is too new for some deps: always `source .venv/bin/activate`).

## The product
A **voice-first daily wellbeing check-in on the Vexo wrist band**. The wearer double-taps the band, a voice agent asks about **sleep, mood, stress, appetite** one at a time, and the output is a **structured, non-diagnostic summary card** a clinician can scan in seconds, plus a **provenance record** of where every piece of data went.

Hard rules (do not break these):
- **Non-diagnostic.** Report only what the wearer said. Never name a condition, never interpret, never give medical advice. The safety gate enforces this.
- **Synthetic data only** for the demo.
- **Privacy:** names/phones/addresses/dates are redacted before text reaches Gemini. Raw audio is never written to disk by the bot. Private mode (`--stt whisper`) keeps audio and transcription fully on the Mac.
- If the wearer mentions self-harm: the agent responds warmly, mentions 988, and the card gets `escalate: true` (red banner on the dashboard). The gate escalates; it does not block the wearer's own words.

## Hardware: the Vexo band (Vexo Open kit)
- Kit repo: `vexo-open-kit/` (cloned from github.com/VexoAI-Org/vexo-open-kit; reference only, never edit it).
- Band has: **mic** (Opus packets, 50/s, 20 ms each, 16 kHz mono, with a `seq` counter), accel/gyro, gestures (tap, double-tap, tilt...), one RGB LED, battery, temperature, tiny key-value storage. **No speaker, no on-device AI model, no heart rate.**
- **Our band is number 1** → BLE name `Vexo-HACK-01`. Connect with `Band.connect(1)` (Python lib `vexo_open.sync`, uses `bleak`).
- Gotchas:
  - **Double-tap before every connect** (blue blink = discoverable for 60 s).
  - **One connection at a time.** Close the kit dashboard tab before running Python.
  - If "not-found-nearby": press RESET once (twice = bootloader), double-tap, retry. Also check macOS System Settings → Privacy & Security → Bluetooth → Terminal enabled.
  - While the mic is on the LED is solid blue and LED commands are refused.
- Decode Opus → PCM with `vexo_open.audio.PcmDecoder` (needs `av`).
- The agent's voice plays from the **Mac** (use earbuds for the demo so the bot doesn't hear itself).

## Architecture (Pipecat pipeline, `neurokind_bot.py`)
```
Vexo band (Opus over BLE)
  → VexoBandInput (custom Pipecat BaseInputTransport: PcmDecoder → InputAudioRawFrame 16 kHz)
  → STT: DeepgramSTTService (cloud, redact=pci/ssn/numbers, keyterms)  |  WhisperSTTServiceMLX (--stt whisper, on device)
  → Redactor (FrameProcessor on TranscriptionFrame: guardrails.redact)
  → LLMContextAggregatorPair.user() with SileroVADAnalyzer (turn detection)
  → GoogleLLMService (Gemini 4 Hydrogen, thinking=low) with tool submit_summary
  → ElevenLabs TTS (eleven_v4_turbo via ElevenLabsDialogueTTSService; flash_v2_5 via ElevenLabsTTSService)
  → LocalAudioTransport output (Mac speakers / earbuds)
  → assistant aggregator
submit_summary(args) → guardrails.check_summary → blocked: return issues so Gemini rewrites
                                               → ok: cards/<session>.json + cards/latest.json → trends.build() → cards/history.json
Every hop → provenance.jsonl
card.html polls cards/latest.json, cards/history.json, provenance.jsonl every 3 s.
```
Pipecat version: **1.16.0**. Its API changes often: check installed source before using a class. Notes from inspecting 1.16:
- VAD goes in `LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())`, not the transport.
- Schemas: `pipecat.adapters.llm.schemas.function_schema.FunctionSchema`, `...tools_schema.ToolsSchema` (old `pipecat.adapters.schemas.*` paths are deprecated).
- Services take `settings=Service.Settings(...)` (passing `voice_id=` / `model=` directly is deprecated).
- ElevenLabs **v3/v4 models are not served by the standard TTS WebSocket**: use `pipecat.services.elevenlabs.dialogue.tts.ElevenLabsDialogueTTSService`. The code picks automatically from `ELEVENLABS_MODEL`.
- `GoogleLLMService.ThinkingConfig(thinking_level="low")`: set explicitly; Pipecat only auto-lowers thinking for models it knows.
- `SileroVADAnalyzer` used standalone: construct with no args, then `set_sample_rate(16000)`; `voice_confidence()` returns an ndarray (use `float(np.asarray(x).squeeze())`).
- Function handlers receive `FunctionCallParams` (`.arguments`, `await .result_callback(dict)`).

## Files
| File | What it does |
|---|---|
| `neurokind_bot.py` | Main Pipecat bot. Flags: `<band#>` or `--sim`, `--replay file.ogg`, `--stt deepgram\|whisper`, `--wait-tap`, `--no-speaker`, `--max-minutes N` (default 4, hard stop to save credits). `--sim` plays the replay file **once** then silence (the kit's sim band loops forever otherwise). Log level via `LOG_LEVEL` env (default INFO). |
| `band_capture.py` | Standalone single-turn capture: band → PCM → **Silero** speech detection → quality checks (`too_short`, `too_quiet`, `clipping`, `bad_link` + re-prompt text) → mlx-whisper → JSON. Saves `last_turn.wav` for debugging. Flags: `--sim`, `--no-whisper`, `--wait-tap`, `--max-seconds`. |
| `guardrails.py` | `redact(text)` (regex: PHONE, EMAIL, SSN, ADDRESS, DATE, NAME after "I'm/my name is/my sister…", plus `KNOWN_NAMES`), `Provenance` (JSONL, never stores content), `SCALES`/`WATCH`, `SUMMARY_SCHEMA` (the tool schema), `check_summary()` (schema + diagnostic-language regex; quotes exempt; escalation regex). |
| `trends.py` | Rule-based overview over the last 7 check-ins (streaks, counts, "topics to ask about"). `python trends.py --seed` writes 6 days of synthetic history (`cards/seed*.json`). Output also passes the diagnostic-language check. |
| `card.html` | Clinician dashboard (pastel tiles per topic with 7-day mini bars, overview + topic chips, quotes + follow-up bar, 7-day timeline grid, data-flow stepper). Plain HTML/JS, no build step. |
| `check_all.py` | Validates Deepgram + ElevenLabs keys and runs `bench_gemini.py`. Never prints keys. |
| `bench_gemini.py` | Times first-token latency per Gemini model. |
| `hud_bot.py` | **Camera HUD** version of the check-in ("talk, tilt, or show it something"). Imports `neurokind_bot` (unchanged). Adds wrist motion (roll → tilt dial, tap = show camera, double-tap = confirm), attaches ONE camera frame to Gemini on "here's…/look at…", tap, or P key; optional `shown` per topic, gated for diagnostic words + people. Flags as the bot plus `--port`, `--flip-dial`. Open http://localhost:8765. |
| `hud_server.py` | aiohttp on 127.0.0.1: serves `hud.html`, receives JPEG frames over WebSocket (newest kept in memory only, never disk), pushes state/captions/cards/wrist/dial. Refuses foreign Origins. `python hud_server.py` = standalone camera test. |
| `hud.html` | Full-screen camera HUD. Keys: P show camera · ← → dial · Enter confirm · M mirror · C captions. |
| `env.example` | Template for `.env`. |
| `requirements.txt` | `pipecat-ai[deepgram,google,elevenlabs,silero,local,mlx-whisper]`, bleak, av, numpy, python-dotenv. |

## Card schema (the hand-off to Alyshah's card)
Each topic: `{reported: neutral paraphrase, quote: wearer's words ≤15 words (optional), level}`.
- sleep: `good | fair | poor | not_discussed`
- mood: `good | okay | low | not_discussed`
- stress: `low | medium | high | not_discussed`
- appetite: `normal | less_than_usual | more_than_usual | not_discussed`
Plus `other_notes`, `wearer_requested_followup` (bool). The bot adds `session`, `created`, `non_diagnostic`, `escalate`, `source`.
"Worth asking about" (red) = sleep poor, mood low, stress high, appetite less/more than usual.

## Config (`.env`, never commit, never print)
```
DEEPGRAM_API_KEY=      GOOGLE_API_KEY=       ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=   (must be a default/own voice: library voices return 402 on free plans)
ELEVENLABS_MODEL=eleven_v4_turbo   (fallback: eleven_flash_v2_5)
GEMINI_MODEL=gemini-4-hydrogen     (fallback: gemini-4-argon)
GEMINI_THINKING=low
```
Gemini choice: measured first-token latency on our key: **Hydrogen ~1.8–2.1 s**, Argon 3.6–13 s. Hydrogen is 2 days old and may 429 under load → switch to Argon in `.env`.

## Commands
```bash
cd ~/Desktop/neurokind && source .venv/bin/activate
python check_all.py                                   # keys + Gemini latency
python neurokind_bot.py --sim --max-minutes 2         # full loop, no band
python neurokind_bot.py 1 --max-minutes 3             # real band (double-tap first)
python neurokind_bot.py 1 --stt whisper               # private mode
python band_capture.py 1 --max-seconds 10             # single-turn mic test
python trends.py --seed                               # synthetic history
python3 -m http.server 8000 --bind 127.0.0.1          # then open http://localhost:8000/card.html
python vexo-open-kit/examples/python/audio_to_text.py 1 --seconds 40 --text no --out demo_checkin.ogg   # record demo input
```

## Status (as of ~1:10 PM)
Done and tested:
- Band connects (#1); mic capture works; Silero speech detection works in the noisy room (old loudness-based detector failed there); local Whisper transcribes correctly.
- Pipecat pipeline runs end to end on the sim band (Deepgram transcribing, Gemini replying ~1.3 s TTFB).
- Deepgram key OK. Gemini key OK. Safety gate blocks "shows signs of insomnia" and Gemini gets the fix list. Dashboard renders with synthetic history.
Update ~2:10 PM: ElevenLabs fixed (voice was still a library voice → now default "Sarah" `EXAVITQu4vr4xnSDxMaL`, v4 Turbo works, TTFB ~0.18 s; the API key cannot list voices). Camera HUD (`hud_bot.py`) runs end to end on `--sim`: Gemini described a test image ("a plate with a sandwich and a red piece of fruit").
Open / next:
1. ~~ElevenLabs voice~~ done.
2. **First full real-band check-in** producing a card (`neurokind_bot.py 1`). This is the milestone.
3. Record `demo_checkin.ogg` (synthetic script incl. a name + phone number so redaction shows).
4. Sync card schema with Alyshah.
5. Rehearse demo; record backup video before 5 PM.
Stretch: Deepgram Flux (`DeepgramFluxSTTService`) if Deepgram judges prefer it; double-tap-to-start inside the bot (`--wait-tap` exists).

## Demo story (2 minutes)
1. Double-tap the band → bot greets through earbuds → short check-in.
2. Dashboard updates live: tiles, "Today" column, topics to ask about.
3. Show **Where the data went**: raw audio on device → Deepgram (with redaction) → redacted text to Gemini → safety gate → card. Point at the `blocked` step if the gate fired.
4. Mention private mode (`--stt whisper`): nothing but redacted text leaves the Mac.
5. Sponsors: Pipecat pipeline with a custom wearable input (Deepgram), Gemini 4 Hydrogen tool-calling with self-correction (DeepMind), Eleven v4 Turbo voice (ElevenLabs).

## Working rules for any AI assistant in this repo
- Never print or commit `.env` or keys. `.gitignore` covers `.env`, `vexo-open-kit/`, `cards/`, `provenance.jsonl`, `last_turn.wav`.
- Every bot run should use `--max-minutes` (APIs cost credits).
- Keep the non-diagnostic rule in any new text the system generates; run new summary text through `guardrails._DIAG_RE`.
- Prefer small, tested changes; it's a one-day hackathon. Test against `--sim` before the real band.
