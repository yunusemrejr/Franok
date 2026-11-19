"""Checks of the held-out rule and of the line-break tasks: `python -m training.fifth.test_filter`."""

import json
import tempfile
from pathlib import Path

from .filter import TEST_FILES, check, prepare, run

TESTS = prepare(["Hi there! What should I call you?", "Describe the main reasons why the Roman legions were so effective in battle across Europe"])
CASES = [
    ("exact short match", [dict(user="Hi there, what should I call you", assistant="Call me Franok.")], "test_exact"),
    ("short, not identical", [dict(user="what should I call you today", assistant="Call me Franok.")], None),
    ("long Jaccard overlap", [dict(user="Why were the Roman legions so effective in battle, and what main reasons explain it across Europe", assistant="They trained hard.")], "test_jaccard"),
]

POEM = "The rain taps the glass,\nThe kettle starts to sing,\nThe slow hours pass,\nA good and quiet thing."
CODE = "print('Hello')\nThis line prints the word Hello on the screen."


def one(reply, lang):
    return [dict(user="write me a short thing" if lang == "en" else "bana kısa bir şey yaz", assistant=reply)]


# (name, reply, language, task, expected reason)
TASK_CASES = [
    ("poem with four lines", POEM, "en", "create", None),
    ("poem with a full stop on every line is not a sentence overflow", POEM.replace("\n", ".\n") + ".", "en", "create", None),
    ("the same poem under another task", POEM, "en", None, "markdown_or_emoji"),
    ("four sentences on one line under another task", "One is here. Two is here. Three is here. Four is here.", "en", "facts", "overlong_reply"),
    ("four sentences on one line under create", "One is here. Two is here. Three is here. Four is here.", "en", "create", None),
    ("turkish poem", "Sabah güneşi doğuyor,\nÇaydanlıkta çay kaynıyor,\nKuşlar dalda şarkı söylüyor,\nGüzel bir gün başlıyor.", "tr", "create", None),
    ("riddle with its answer", "I have hands but cannot clap. What am I?\nAnswer: a clock", "en", "create", None),
    ("poem with seven lines", "\n".join(["Rain falls down"] * 7), "en", "create", "bad_lines"),
    ("poem with a blank line", "Rain falls down,\n\nSoft and slow", "en", "create", "bad_lines"),
    ("poem with a literal backslash n", "Rain falls down,\\nSoft and slow", "en", "create", "bad_lines"),
    ("poem with a backtick", "Rain falls `down`,\nSoft and slow", "en", "create", "markdown_or_emoji"),
    ("poem with a bullet", "- Rain falls down,\nSoft and slow", "en", "create", "markdown_or_emoji"),
    ("poem with an identity leak", "I am a language model,\nand I write rhymes", "en", "create", "format_or_identity"),
    ("poem over sixty words", "\n".join(["rain falls down on the quiet town and the sleepy hill"] * 6), "en", "create", "overlong_reply"),
    ("code with a sentence", CODE, "en", "code", None),
    ("code with an indented block", "x = 12\nif x > 10:\n    print('big')\nelse:\n    print('small')\nThis prints big because 12 is more than 10.", "en", "code", None),
    ("turkish code", "x = 7\nprint(x)\nİlk satır 7'yi x'te saklar, ikincisi onu ekrana yazar.", "tr", "code", None),
    ("one explaining sentence without code", "This line stores the number 5 in the variable x.", "en", "code", None),
    ("code without a sentence", "print('Hello')", "en", "code", "bad_code"),
    ("sentence before the code", "This prints Hello.\nprint('Hello')", "en", "code", "bad_code"),
    ("code that does not parse", "print('Hello'\nThis prints Hello.", "en", "code", "bad_code"),
    ("code with import", "import math\nprint(math.sqrt(9))\nThis prints 3.0.", "en", "code", "bad_code"),
    ("code with open(", "f = open('a.txt')\nprint(f)\nThis opens a file.", "en", "code", "bad_code"),
    ("code with while True", "while True:\n    print('hi')\nThis prints hi forever.", "en", "code", "bad_code"),
    ("code in a fence", "```python\nprint('Hello')\n```\nThis prints Hello.", "en", "code", "markdown_or_emoji"),
    ("code with a comment line", "# say hello\nprint('Hello')\nThis prints Hello.", "en", "code", "markdown_or_emoji"),
    ("code with too many lines", "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\nThis stores six numbers.", "en", "code", "bad_lines"),
]


def newlines_survive():
    """A create row keeps its line breaks all the way into the output file; a line break under another task drops the row."""
    seed = dict(lang="en", topic="poem", turns=1)
    rows = [
        dict(seed=dict(seed, id="a", task="create"), finish="stop", text=json.dumps(dict(turns=[dict(user="write me a poem", assistant=POEM)]))),
        dict(seed=dict(seed, id="b", task="facts"), finish="stop", text=json.dumps(dict(turns=[dict(user="why is the sky blue", assistant=POEM)]))),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        raw, out, report = Path(tmp, "raw.jsonl"), Path(tmp, "out.jsonl"), Path(tmp, "report.json")
        raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
        run([raw], [], out, report, [], "digest")
        kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert [k["id"] for k in kept] == ["a"], kept
    assert kept[0]["turns"][0]["assistant"] == POEM and kept[0]["task"] == "create"


if __name__ == "__main__":
    for name, conv, expected in CASES:
        got = check(conv, "en", TESTS)
        print(f"{name}: {'dropped (' + got + ')' if got else 'kept'}")
        assert got == expected, (name, got, expected)
    for name, reply, lang, task, expected in TASK_CASES:
        got = check(one(reply, lang), lang, TESTS, task=task)
        print(f"{name}: {'dropped (' + got + ')' if got else 'kept'}")
        assert got == expected, (name, got, expected)
    newlines_survive()
    print("line breaks survive into the output rows")
    assert "tests/v5_user_conversations.json" in TEST_FILES
