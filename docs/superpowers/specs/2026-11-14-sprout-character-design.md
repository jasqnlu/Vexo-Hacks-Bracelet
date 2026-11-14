# Sprout: Tend's on-screen character (HUD)

**Intent.** Give Tend a face on the camera HUD so judges see, from across the room, that the check-in is
voice-first: the character visibly listens, thinks, and talks. Audience: demo judges (chosen by Jason).

**Concept (approved).** "Sprout": a small round seedling with two leaf "ears". Ties to the name (you *tend*
a plant). Warm and calm rather than bouncy: Tend talks about sleep, stress, mood, and sometimes self-harm.

**Approaches considered.**
- A. Inline SVG + CSS animation in `hud.html`, driven by state messages from the bot. **Picked**: no new
  files or libraries, ~40 min, reliable on stage. Mouth animates on a loop while speaking (TTS plays from
  Python, so the browser can't measure the voice's loudness).
- B. Lottie/Rive or generated art: needs art sourced, licensed and wired up; no time before freeze.
- C. Glowing orb: safe but not a character, and judges have seen it many times.

**Message contract (bot -> HUD, over the existing WebSocket).**
`{"type": "avatar", "state": S}` with S one of:
| state | when | Sprout does |
|---|---|---|
| `idle` | waiting for the wearer (after the bot finished speaking) | slow breathing, occasional blink |
| `listening` | the wearer is speaking (user turn started) | leaves perk up, leans toward the captions |
| `thinking` | wearer finished, or a dial/double-tap answer was sent; until the bot speaks | slow sway, three dots |
| `speaking` | bot audio playing (BotStartedSpeaking .. BotStoppedSpeaking) | gentle bounce, mouth moves |
| `concern` | the safety gate escalated (self-harm path) | still, soft glow, no bounce; stays until reload |
The HUD falls back from `thinking` to `idle` after 12 s with no new state (e.g. a silent tool call).

**Constraints.** Plain HTML/CSS/JS in `hud.html` (no build step, no CDN). Readable on a projector.
Doesn't cover the dial (centre), captions (bottom centre), topic rail (left) or cards (right).
Respects `prefers-reduced-motion`. Keyboard key to cycle states for rehearsal without the band.

**Split.** HUD visuals: `hud.html` only. Bot signals: `hud_bot.py` only.
