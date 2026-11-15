"""Card quotes must be the wearer's own words. Run: python tests/test_quotes.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hud_bot import quote_was_said  # noqa: E402

SAID = ["I slept okay I guess, woke up twice", "honestly work was really stressful today",
        "my sister [NAME] was up late", "dormí muy mal anoche"]
CASES = [
    ("woke up twice", True),                        # exact
    ("I slept okay, I guess", True),                # punctuation differs
    ("Work was really stressful today.", True),     # partial sentence, case
    ("my sister [NAME] was up late", True),         # redacted tokens match
    ("Dormí muy mal anoche", True),                 # Spanish
    ("My appetite is fine, just normal.", False),   # invented
    ("I feel great about my sleep", False),         # invented
    ("", True),                                     # no quote = fine
]
bad = [(q, want) for q, want in CASES if quote_was_said(q, SAID) != want]
for q, want in bad:
    print(f"FAIL: {q!r} expected {want}")
print("PASS" if not bad else f"{len(bad)} failed")
sys.exit(1 if bad else 0)
