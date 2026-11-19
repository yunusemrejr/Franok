"""Parse, validate and de-contaminate Franok 0.5 conversations.

Same parsing, grading and held-out rules as the fourth round, with these changes: tests/v5_conversations.json joins the held-out
files when it exists, and so does tests/v5_user_conversations.json; a user turn with fewer than six content words is dropped only when it equals
a held-out turn exactly (short chit-chat cannot avoid sharing words with a test turn); replies with markdown, an emoji or more than 60 words are dropped;
Turkish typed without diacritics passes the language check; short openers such as "hi" may repeat across conversations.
Rows whose seed task is `create` or `code` may span up to six lines (line breaks are kept) and need not be one to three sentences; backticks and markdown
are still dropped, and `code` replies must be Python lines followed by one sentence, without imports, files or endless loops.
"""

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ..fourth.filter import (
    BAD_REPLY, EMOJI, EN_STOP, FOREIGN_SCRIPT, TEST_FILES as FOURTH_TEST_FILES, TR_CHARS, TR_STOP, benchmark_questions, content, fold, grams, jaccard, key, parse,
    sentences, words,
)

TEST_FILES = FOURTH_TEST_FILES + ("tests/v5_conversations.json", "tests/v5_user_conversations.json")
SHORT_TURN = 6
MAX_REPLY_WORDS = 60
MARKDOWN_BODY = r"\*\*|^#|^\s*[-*•]\s|^\s*\d+[.)]\s|`"
MARKDOWN = re.compile(MARKDOWN_BODY + r"|\n", re.M)
MARKDOWN_LINES = re.compile(MARKDOWN_BODY, re.M)  # for tasks whose replies legitimately hold line breaks
LINE_TASKS = ("create", "code")
MAX_LINES = 6
CODE_BANNED = ("import ", "open(", "while True")
ASCII_TR = str.maketrans("çğıöşü", "cgiosu")
TR_STOP_ASCII = {w.translate(ASCII_TR) for w in TR_STOP}


def normal(text):
    return re.sub(r"\W+", " ", fold(text)).strip()


def language_ok(text, lang):
    """False when a text of three or more words clearly belongs to the other language or script; Turkish may lack diacritics."""
    if FOREIGN_SCRIPT.search(text):
        return False
    w = words(text)
    if len(w) < 3:
        return True
    en = sum(x in EN_STOP for x in w)
    tr = sum(x in TR_STOP or x in TR_STOP_ASCII for x in w)
    accents = sum(any(c in TR_CHARS for c in x) for x in w)
    if lang == "tr":
        return tr + accents >= en
    return en >= tr + accents or (accents == 0 and tr == 0)


def prepare(texts):
    return [(normal(t), content(t), grams(t)) for t in texts]


def test_files():
    return [f for f in TEST_FILES if Path(f).exists()]


def load_tests(extra_questions=()):
    """Held-out user turns of the existing test files plus benchmark question text, and a digest of the files."""
    texts, files = [], test_files()
    for file in files:
        data = json.loads(Path(file).read_text())
        texts += [row[1] for row in data] if file.endswith("dialogue.json") else [turn for case in data for turn in case["turns"]]
    texts += list(extra_questions)
    return prepare(texts), hashlib.sha256(b"".join(Path(f).read_bytes() for f in files)).hexdigest()


def contaminated(user, tests):
    """Exact normalised match for short turns; 6-gram or Jaccard overlap for longer ones."""
    c = content(user)
    if len(c) < SHORT_TURN:
        return "test_exact" if normal(user) in {n for n, _, _ in tests} else None
    g = grams(user)
    for _, tc, tg in tests:
        if g & tg:
            return "test_6gram"
        if jaccard(c, tc) >= 0.3:
            return "test_jaccard"
    return None


def is_python(text):
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return True


def line_problem(reply, task):
    """The reason a `create` or `code` reply is dropped for its line layout or code, or None."""
    lines = reply.split("\n")
    if len(lines) > MAX_LINES or not all(line.strip() for line in lines) or "\\" in reply:
        return "bad_lines"
    if task == "code":
        if any(banned in reply for banned in CODE_BANNED):
            return "bad_code"
        # Python lines come first and one sentence comes last; the sentence is not Python and the lines before it are.
        if is_python(lines[-1]) or (len(lines) > 1 and not is_python("\n".join(lines[:-1]))):
            return "bad_code"
    return None


def check(conv, lang, tests, language=True, trusted=False, task=None):
    """The reason a conversation is dropped, or None. `trusted` rows (programmatic identity dialogues) may name Franok's own nature.

    `task` is the seed's family: `create` and `code` replies may hold line breaks and need not be one to three sentences.
    """
    lined = task in LINE_TASKS
    for t in conv:
        u, a = t["user"], t["assistant"]
        if not u or not a:
            return "empty"
        if len(words(a)) > MAX_REPLY_WORDS or (not lined and sentences(a) > 3):
            return "overlong_reply"
        if len(words(u)) > 45:
            return "overlong_user"
        if (MARKDOWN_LINES if lined else MARKDOWN).search(a) or EMOJI.search(a):
            return "markdown_or_emoji"
        reason = line_problem(a, task) if lined else None
        if reason:
            return reason
        if EMOJI.search(u) or (not trusted and BAD_REPLY.search(a.replace("\n", " ") if lined else a)):
            return "format_or_identity"
        if fold(a).strip(" .!?") == fold(u).strip(" .!?"):
            return "echo"
    if language:
        for t in conv:
            # In a `code` reply only the closing sentence is prose; the lines before it are Python.
            prose = t["assistant"].split("\n")[-1] if task == "code" else t["assistant"]
            if not language_ok(prose, lang):
                return "wrong_language_reply"
        for t in conv:
            if not language_ok(t["user"], lang):
                return "wrong_language_user"
    for t in conv:
        reason = contaminated(t["user"], tests)
        if reason:
            return reason
    return None


def opening(conv):
    first = normal(conv[0]["user"])
    return first if len(first.split()) >= 5 else None


def run(raw_files, identity_files, out, report, tests, test_digest, grades=(), existing=()):
    counts, verdict = Counter(), {}
    for file in grades:
        for line in Path(file).read_text().splitlines():
            if line:
                row = json.loads(line)
                verdict[row["id"]] = row["verdict"]
    kept, openings, whole = [], set(), set()
    for file in existing:
        for line in Path(file).read_text().splitlines():
            if line:
                row = json.loads(line)
                counts[f"existing_{row['source']}_{row['lang']}"] += 1
                if opening(row["turns"]):
                    openings.add(opening(row["turns"]))
                whole.add(key(row["turns"]))
                kept.append(row)
    counts["existing_kept"] = len(kept)
    for file in raw_files:
        for line in Path(file).read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            s = r["seed"]
            counts["teacher_in"] += 1
            counts["in_" + s["lang"]] += 1
            conv, reason = parse(r)
            if reason is None:
                reason = check(conv, s["lang"], tests, task=s["task"])
            if reason is None and verdict and not verdict.get(s["id"], "").startswith("GOOD"):
                reason = "grader_flagged" if s["id"] in verdict else "grader_missing"
            if reason is None and (opening(conv) in openings or key(conv) in whole):
                reason = "duplicate"
            if reason:
                counts["drop_" + reason] += 1
                counts[f"drop_{reason}_{s['lang']}"] += 1
                continue
            if opening(conv):
                openings.add(opening(conv))
            whole.add(key(conv))
            counts["kept_" + s["lang"]] += 1
            kept.append(dict(id=s["id"], lang=s["lang"], source="teacher", task=s["task"], topic=s["topic"], turns=conv))
    counts["teacher_kept"] = len(kept) - counts["existing_kept"]
    for file in identity_files:
        for line in Path(file).read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            counts["identity_in"] += 1
            reason = check(r["turns"], r["lang"], tests, language=False, trusted=True)
            if reason is None and key(r["turns"]) in whole:
                reason = "duplicate"
            if reason:
                counts["identity_drop_" + reason] += 1
                continue
            whole.add(key(r["turns"]))
            counts["identity_kept"] += 1
            counts[f"identity_kept_{r['lang']}"] += 1
            kept.append(r)
    for c in kept:
        if "split" not in c:
            c["split"] = "dev" if int.from_bytes(hashlib.sha256(c["id"].encode()).digest()[:4], "little") % 100 < 3 else "train"
    by_lang = Counter((c["source"], c["lang"], c["split"]) for c in kept)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in kept))
    report.write_text(
        json.dumps(
            dict(counts=dict(counts), kept=len(kept), by_source_lang_split={"/".join(k): v for k, v in sorted(by_lang.items())},
                 test_files=test_files(), test_files_sha256=test_digest, test_turns=len(tests)),
            indent=2, ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps(dict(counts), indent=1), len(kept), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw", nargs="*", type=Path, help="teacher output files")
    p.add_argument("--identity", nargs="*", type=Path, default=[], help="JSONL files from training.fifth.identity")
    p.add_argument("--existing", nargs="*", type=Path, default=[], help="filtered conversation files from an earlier round, kept as they are")
    p.add_argument("--grades", nargs="*", type=Path, default=[], help="JSONL verdicts from the teacher grading pass")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--no-benchmarks", action="store_true", help="skip the XQuAD and Belebele question lists (offline smoke tests only)")
    a = p.parse_args()
    tests, digest = load_tests([] if a.no_benchmarks else benchmark_questions())
    run(a.raw, a.identity, a.out, a.report, tests, digest, a.grades, a.existing)


if __name__ == "__main__":
    main()
