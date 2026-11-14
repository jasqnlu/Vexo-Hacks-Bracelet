# Camera HUD extension: design (short)

**Intent.** Extend the NeuroKind check-in so the wearer can *show* as well as *tell*. A camera view
on the Mac (or iPhone via Continuity Camera) acts as a HUD; the band's mic and motion drive it.
The pitch: "a hands-free daily check-in on your wrist: talk, tilt, or show it something."

**Goals.** HUD page with live camera + check-in progress + answer cards; a camera frame attached to
Gemini when the wearer shows something ("here's my lunch"); neutral one-line description on the card
(`shown` field per topic); wrist tilt scrolls/answers, double-tap confirms.

**Non-goals.** Real glasses; recording bystanders; storing images; any interpretation of images.

**Anti-goals.** Silent always-on capture of other people (CA two-party consent); describing faces or people;
diagnostic language in image descriptions.

**Constraints.** Ship by the 5 PM freeze. One BLE connection: motion + mic + gestures share the
band connection in `VexoBandInput`. NeuroKind files stay working as the fallback demo.

**Approaches considered.**
- Python owns the camera (OpenCV): extra install, no live view for free. Rejected.
- Separate HUD repo: loses the shared guardrails/provenance. Rejected.
- **Chosen:** browser owns the camera (`hud.html`, getUserMedia); sends downscaled JPEGs over a
  localhost WebSocket to `hud_server.py` (aiohttp, inside the bot process); the server keeps only the
  latest frame in memory and pushes state/cards/wrist back.

**Decided.** Push-to-talk / visible listening indicator; image never stored; Gemini told not to describe
people; provenance step `camera frame -> Gemini (not stored)`; safety gate checks image descriptions.

**Left to the builder.** When to attach a frame (trigger words or a recent double-tap); exact HUD layout;
wrist mapping thresholds.

**Open with Alyshah.** Optional `shown` field per topic on the card schema.
