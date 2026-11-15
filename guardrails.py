"""Tend guardrails: local redaction, provenance log, safety gate. No network, no extra installs."""
import json
import re
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------- redaction
# Runs on the transcript BEFORE it reaches the LLM. Synthetic demo data only.
_PATTERNS = [
    ("PHONE", r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("ADDRESS", r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct)\b\.?"),
    ("DATE", r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b"),
    ("DATE", r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"),
    ("NAME", r"\b(?:[Mm]y name is|I'm|I am|[Cc]all me|[Mm]y (?:wife|husband|son|daughter|mom|dad|mother|father|sister|brother|friend|partner|boss|doctor)|[Mm]e llamo|[Ss]oy|[Mm]i (?:esposa|esposo|hijo|hija|mamá|papá|madre|padre|hermana|hermano|amiga|amigo|pareja|jefe|jefa|novia|novio|doctora?))\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+)?)"),
]
# Names you know will appear in the synthetic demo script (add to this list).
KNOWN_NAMES = ["Alyshah", "Jason", "Maya", "Daniel", "Priya", "Sam"]
# "I'm tired" must not become "I'm [NAME]": the NAME rule only fires on a capitalised word that isn't a feeling.
_NOT_NAMES = {"Tired", "Fine", "Okay", "Good", "Bad", "Stressed", "Anxious", "Sad", "Happy", "Not", "Feeling", "Really", "So", "Just", "Sleeping",
              "Bien", "Mal", "Feliz", "Triste", "Cansado", "Cansada", "Muy", "Un", "Una", "No", "Estresado", "Estresada"}


def redact(text: str):
    """Returns (redacted_text, spans). spans = [{"type", "original_len"}]: never the original value."""
    spans = []

    def sub(kind, pattern, s, group=0):
        def repl(m):
            if group:
                word = m.group(group)
                if word.split()[0] in _NOT_NAMES:
                    return m.group(0)
                spans.append({"type": kind, "original_len": len(word)})
                return m.group(0).replace(word, f"[{kind}]")
            spans.append({"type": kind, "original_len": len(m.group(0))})
            return f"[{kind}]"
        return re.sub(pattern, repl, s)

    out = text
    for kind, pat in _PATTERNS:
        out = sub(kind, pat, out, group=1 if kind == "NAME" else 0)
    if KNOWN_NAMES:
        out = sub("NAME", r"\b(?:%s)\b" % "|".join(map(re.escape, KNOWN_NAMES)), out)
    return out, spans


# ---------------------------------------------------------------- provenance
class Provenance:
    """Append-only JSONL: one line per hop. What data, from where, to where, when. Never the content itself."""

    def __init__(self, path="provenance.jsonl"):
        self.path = Path(path)
        self.session = uuid.uuid4().hex[:8]

    def log(self, hop: str, data: str, dest: str, **extra):
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": self.session,
               "hop": hop, "data": data, "dest": dest, **extra}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec


# ---------------------------------------------------------------- safety gate
DOMAINS = ["sleep", "mood", "stress", "appetite"]

# Diagnostic / clinical-judgement language the card must never contain.
_DIAGNOSTIC = [
    r"\bdiagnos\w*", r"\bdisorder\w*", r"\bsyndrome\b", r"\bclinical(?:ly)?\s+(?:depress|anxi)\w*",
    r"\b(?:major\s+)?depressi(?:on|ve)\b", r"\bdepressed\b", r"\banxiety\b", r"\binsomnia\b", r"\bbipolar\b",
    r"\bptsd\b", r"\badhd\b", r"\bocd\b", r"\banorexi\w*", r"\bbulimi\w*", r"\bmania\b", r"\bmanic\b",
    r"\bpsychos\w*", r"\bsuffers? from\b", r"\b(?:has|have|shows?|exhibits?|indicat\w*|consistent with)\s+(?:signs|symptoms)\s+of\b",
    r"\blikely has\b", r"\bat risk (?:of|for)\b", r"\bprescri\w*", r"\bmedicat\w*", r"\btreatment\b",
]
_DIAG_RE = re.compile("|".join(_DIAGNOSTIC), re.I)

# A safety concern in what the wearer SAID is not blocked: it is escalated to the clinician.
_ESCALATE_RE = re.compile(r"\b(?:hurt (?:myself|himself|herself)|kill (?:myself|himself|herself)|suicid\w*|end (?:my|it all)|self[- ]harm)\b", re.I)

# Each topic uses the wearer's own words on a natural scale ("high" sleep would be ambiguous).
SCALES = {
    "sleep": ["good", "fair", "poor"],
    "mood": ["good", "okay", "low"],
    "stress": ["low", "medium", "high"],
    "appetite": ["normal", "less_than_usual", "more_than_usual"],
}
# Which answers the trend summary counts as "worth asking about". A pattern in reports, never a finding.
WATCH = {"sleep": {"poor"}, "mood": {"low"}, "stress": {"high"}, "appetite": {"less_than_usual", "more_than_usual"}}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        **{d: {"type": "object", "description": f"What the wearer said about {d}.",
               "properties": {"reported": {"type": "string", "description": "Short neutral paraphrase of what they said. No interpretation."},
                              "quote": {"type": "string", "description": "Their own words, verbatim, max 15 words."},
                              "level": {"type": "string", "enum": SCALES[d] + ["not_discussed"],
                                        "description": f"How the WEARER described their {d}, on this scale: "
                                                       + ", ".join(SCALES[d]) + ". not_discussed if skipped."}},
               "required": ["reported", "level"]} for d in DOMAINS},
        "other_notes": {"type": "string", "description": "Anything else the wearer raised, paraphrased neutrally."},
        "wearer_requested_followup": {"type": "boolean"},
    },
    "required": DOMAINS + ["wearer_requested_followup"],
}


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def check_summary(summary: dict):
    """Returns (ok, issues, escalate). Quotes are the wearer's own words, so they are exempt
    from the diagnostic check but are scanned for safety escalation."""
    issues = []
    for d in DOMAINS:
        v = summary.get(d)
        if not isinstance(v, dict):
            issues.append(f"missing domain: {d}")
            continue
        if v.get("level") not in SCALES[d] + ["not_discussed"]:
            issues.append(f"{d}.level invalid")
        if not isinstance(v.get("reported"), str):
            issues.append(f"{d}.reported missing")
        if len(str(v.get("quote", "")).split()) > 15:
            issues.append(f"{d}.quote longer than 15 words")
    if not isinstance(summary.get("wearer_requested_followup"), bool):
        issues.append("wearer_requested_followup must be true/false")

    # Diagnostic language: everything except the verbatim quotes.
    for d in DOMAINS:
        v = summary.get(d) if isinstance(summary.get(d), dict) else {}
        for field in ("reported",):
            m = _DIAG_RE.search(str(v.get(field, "")))
            if m:
                issues.append(f'diagnostic language in {d}.{field}: "{m.group(0)}"')
    m = _DIAG_RE.search(str(summary.get("other_notes", "")))
    if m:
        issues.append(f'diagnostic language in other_notes: "{m.group(0)}"')

    escalate = any(_ESCALATE_RE.search(s) for s in _strings(summary))
    return (not issues), issues, escalate


if __name__ == "__main__":
    print(redact("Hi, I'm Maya, call me at 415-555-0134. I live at 12 Oak Street and my appointment is March 3rd."))
    print(redact("I'm tired and I'm stressed"))
    bad = {d: {"reported": "Wearer has symptoms of insomnia", "level": "high"} for d in DOMAINS}
    bad["wearer_requested_followup"] = False
    print(check_summary(bad))
    good = {d: {"reported": "Slept about 4 hours, woke up several times", "quote": "I barely slept", "level": "not_discussed"} for d in DOMAINS}
    good["sleep"]["level"] = "poor"
    good["wearer_requested_followup"] = True
    print(check_summary(good))
