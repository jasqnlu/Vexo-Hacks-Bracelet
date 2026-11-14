"""Check-in history + rule-based overview for the clinician card. No LLM, no network.

  python trends.py --seed     write 6 days of synthetic history before today, then rebuild
  python trends.py            rebuild cards/history.json from cards/*.json

The overview reports PATTERNS IN WHAT THE WEARER SAID and topics a clinician may want to ask about.
It never names a condition; every line also passes the same diagnostic-language check as the card.
"""
import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import guardrails as g

HERE = Path(__file__).resolve().parent
CARDS = HERE / "cards"
WINDOW = 7                       # check-ins the overview looks at

PHRASE = {"sleep": {"poor": "poor sleep", "fair": "fair sleep", "good": "good sleep"},
          "mood": {"low": "low mood", "okay": "okay mood", "good": "good mood"},
          "stress": {"high": "high stress", "medium": "medium stress", "low": "low stress"},
          "appetite": {"less_than_usual": "eating less than usual", "more_than_usual": "eating more than usual",
                       "normal": "normal appetite"}}


def load_cards():
    cards = []
    for p in CARDS.glob("*.json"):
        if p.name in ("latest.json", "history.json"):
            continue
        try:
            c = json.loads(p.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        if c.get("created"):
            cards.append(c)
    return sorted(cards, key=lambda c: c["created"])


def overview(cards):
    recent = cards[-WINDOW:]
    n = len(recent)
    lines, ask = [], []
    if not n:
        return {"lines": [], "ask_about": [], "window": 0}
    for d in g.DOMAINS:
        levels = [c.get(d, {}).get("level", "not_discussed") for c in recent]
        flagged = [lv in g.WATCH[d] for lv in levels]
        hits = sum(flagged)
        streak = 0
        for f in reversed(flagged):
            if not f:
                break
            streak += 1
        if hits == 0:
            continue
        worst = next(lv for lv in reversed(levels) if lv in g.WATCH[d])
        what = PHRASE[d][worst]
        if streak >= 2:
            lines.append({"domain": d, "text": f"Reported {what} in the last {streak} check-ins in a row ({hits} of {n} overall)."})
        else:
            lines.append({"domain": d, "text": f"Reported {what} in {hits} of the last {n} check-ins."})
        if streak >= 2 or hits >= max(2, n // 2):
            ask.append(d)
    fu = sum(1 for c in recent if c.get("wearer_requested_followup"))
    if fu:
        lines.append({"domain": "followup", "text": f"Asked for a clinician follow-up in {fu} of the last {n} check-ins."})
    if any(c.get("escalate") for c in recent):
        lines.insert(0, {"domain": "safety", "text": "Safety language appeared in the wearer's own words: see the flagged check-in."})
    # Belt and braces: the overview must pass the same gate as the card.
    for ln in lines:
        if g._DIAG_RE.search(ln["text"]):
            raise ValueError(f"overview produced diagnostic language: {ln['text']}")
    return {"lines": lines, "ask_about": ask, "window": n}


def build():
    cards = load_cards()
    hist = {"generated": datetime.now().isoformat(timespec="seconds"),
            "overview": overview(cards),
            "timeline": [{"created": c["created"], "session": c.get("session"), "synthetic_history": c.get("synthetic_history", False),
                          **{d: c.get(d, {}).get("level", "not_discussed") for d in g.DOMAINS},
                          "followup": bool(c.get("wearer_requested_followup")), "escalate": bool(c.get("escalate"))}
                         for c in cards[-14:]]}
    CARDS.mkdir(exist_ok=True)
    (CARDS / "history.json").write_text(json.dumps(hist, indent=2))
    return hist


SEED = [  # 6 synthetic days, oldest first: sleep and stress drift worse, appetite dips late
    ("good", "good", "medium", "normal", "Slept about seven hours.", "Feeling pretty good."),
    ("fair", "good", "medium", "normal", "Woke up once, around six hours.", "Good, busy day."),
    ("fair", "okay", "high", "normal", "About six hours, restless.", "Okay, a bit tense."),
    ("poor", "okay", "high", "normal", "Maybe five hours, kept waking up.", "Okay, tired."),
    ("poor", "low", "high", "less_than_usual", "Four or five hours.", "Pretty flat today."),
    ("poor", "okay", "high", "less_than_usual", "Woke up around three and stayed up.", "Okay, just tired."),
]


def seed():
    CARDS.mkdir(exist_ok=True)
    today = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    for i, (sl, mo, st, ap, sl_txt, mo_txt) in enumerate(SEED):
        day = today - timedelta(days=len(SEED) - i)
        sid = f"seed{i}"
        card = {"session": sid, "created": day.isoformat(timespec="seconds"), "non_diagnostic": True,
                "escalate": False, "source": "synthetic demo data", "synthetic_history": True,
                "sleep": {"reported": sl_txt, "level": sl},
                "mood": {"reported": mo_txt, "level": mo},
                "stress": {"reported": {"low": "Calm week.", "medium": "Some deadlines.", "high": "Work has been a lot."}[st], "level": st},
                "appetite": {"reported": {"normal": "Eating normally.", "less_than_usual": "Not very hungry."}[ap], "level": ap},
                "other_notes": "", "wearer_requested_followup": i == 4}
        ok, issues, _ = g.check_summary(card)
        assert ok, issues
        (CARDS / f"{sid}.json").write_text(json.dumps(card, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true")
    a = ap.parse_args()
    if a.seed:
        seed()
    h = build()
    print(json.dumps(h["overview"], indent=2))
