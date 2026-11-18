"""Parse, validate and de-contaminate teacher conversations. Logs how many each rule removes."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

TEST_FILES = (
    "tests/transfer_conversations.json",
    "tests/v4_conversations.json",
    "tests/conversations.json",
    "tests/dialogue.json",
)
EN_STOP = set(
    """a about after again all also am an and any are as at be because been before being but by can could did do does
    doing don down for from further had has have having he her here hers him his how i if in into is it its just me more
    most my no nor not of off on once only or other our out over own same she should so some such than that the their
    them then there these they this those through to too under until up us very was we were what when where which while
    who whom why will with would you your yours yourself please thanks thank hi hello ok okay yes also still now""".split()
)
TR_STOP = set(
    """ve ile da de ki bu şu o bir biz ben sen siz onlar mi mı mu mü ne neden nasıl neyi nerede kim kaç için gibi ama fakat
    veya ya yani çok daha en her hiç bazı şey var yok olarak olan olur olsa mı mi mu mü ise ancak sadece lütfen tamam evet
    hayır merhaba teşekkürler teşekkür ederim sağ ol peki şimdi artık zaten bile hem kadar sonra önce göre bana beni sana
    seni onu ona bunu buna bunun şunu benim senin bizim sizin onun nın nin nun nün ın in un ün""".split()
)
TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
BAD_REPLY = re.compile(
    r"\*\*|^#|^\s*[-*•]\s|^\s*\d+[.)]\s|\n|`|\bqwen\b|alibaba|openai|chatgpt|anthropic|language model|dil modeli|yapay zek[aâ] olarak|as an ai\b|<think>|</think>",
    re.I | re.M,
)
FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿ֐-ۿऀ-෿぀-ヿ㐀-鿿가-힯]")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def fold(text):
    text = unicodedata.normalize("NFKC", text).replace("İ", "i").replace("I", "ı")
    return text.lower()


def words(text):
    return re.findall(r"\w+", fold(text))


def content(text):
    return {w for w in words(text) if w not in EN_STOP and w not in TR_STOP and (len(w) > 1 or w.isdigit())}


def grams(text, n=6):
    w = words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def sentences(text):
    return len([s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s])


def language_ok(text, lang):
    """False when a text of three or more words clearly belongs to the other language or script."""
    if FOREIGN_SCRIPT.search(text):
        return False
    w = words(text)
    if len(w) < 3:
        return True
    en = sum(x in EN_STOP for x in w)
    tr = sum(x in TR_STOP for x in w)
    accents = sum(any(c in TR_CHARS for c in x) for x in w)
    if lang == "tr":
        return tr + 1.5 * accents > en or (tr + accents >= 1 and en == 0 and len(w) < 6)
    return en >= tr + accents or (accents == 0 and tr == 0)


def load_tests(extra_questions=()):
    """User turns of the held-out files plus the question text of XQuAD and Belebele items."""
    texts = []
    for file in TEST_FILES:
        data = json.loads(Path(file).read_text())
        if file.endswith("dialogue.json"):
            texts += [row[1] for row in data]
        else:
            texts += [turn for case in data for turn in case["turns"]]
    texts += list(extra_questions)
    return [(t, content(t), grams(t)) for t in texts], hashlib.sha256(
        b"".join(Path(f).read_bytes() for f in TEST_FILES)
    ).hexdigest()


def benchmark_questions():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tools.reading import REV
    from tools.transfer import XQUAD

    out = []
    for lang, dialect in (("en", "eng_Latn"), ("tr", "tur_Latn")):
        file = hf_hub_download("google/xquad", f"xquad.{lang}/validation-00000-of-00001.parquet", revision=XQUAD, repo_type="dataset")
        out += [r["question"] for r in pq.read_table(file).to_pylist()]
        file = hf_hub_download("facebook/belebele", "data/" + dialect + ".jsonl", revision=REV, repo_type="dataset")
        out += [json.loads(line)["question"] for line in Path(file).read_text().splitlines()]
    return out


def parse(record):
    """Return (conversation, reason). A conversation is a list of {user, assistant}."""
    text = record["text"].strip()
    if record["finish"] != "stop":
        return None, "truncated"
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
        turns = data["turns"]
    except (ValueError, KeyError, TypeError):
        return None, "malformed_json"
    if (
        not isinstance(turns, list)
        or len(turns) != record["seed"]["turns"]
        or not all(isinstance(t, dict) and isinstance(t.get("user"), str) and isinstance(t.get("assistant"), str) for t in turns)
    ):
        return None, "malformed_json"
    return [dict(user=t["user"].strip(), assistant=t["assistant"].strip()) for t in turns], None


def check(conv, lang, tests, language=True):
    for t in conv:
        u, a = t["user"], t["assistant"]
        if not u or not a:
            return "empty"
        if len(words(a)) > 55 or sentences(a) > 3:
            return "overlong_reply"
        if len(words(u)) > 45:
            return "overlong_user"
        if BAD_REPLY.search(a) or EMOJI.search(a) or EMOJI.search(u):
            return "format_or_identity"
        if fold(a).strip(" .!?") == fold(u).strip(" .!?"):
            return "echo"
    for t in conv if language else ():
        if not language_ok(t["assistant"], lang):
            return "wrong_language_reply"
    for t in conv if language else ():
        if not language_ok(t["user"], lang):
            return "wrong_language_user"
    for t in conv:
        c, g = content(t["user"]), grams(t["user"])
        for _, tc, tg in tests:
            if g & tg:
                return "test_6gram"
            if jaccard(c, tc) >= 0.3:
                return "test_jaccard"
    return None


def key(conv):
    return hashlib.sha256(
        "\n".join(re.sub(r"\W+", " ", fold(t["user"] + " " + t["assistant"])).strip() for t in conv).encode()
    ).hexdigest()


def run(raw_files, arithmetic, out, report, tests, test_digest, grades=(), existing=(), programmatic=()):
    counts = Counter()
    verdict = {}
    for file in grades:
        for line in Path(file).read_text().splitlines():
            if line:
                row = json.loads(line)
                verdict[row["id"]] = row["verdict"]
    kept, first_turns, whole = [], set(), set()
    for file in existing:
        # Rows of an earlier round were filtered with the same rules; they stay as they are and their openings count as seen.
        for line in Path(file).read_text().splitlines():
            if line:
                row = json.loads(line)
                counts[f"existing_{row['source']}_{row['lang']}"] += 1
                first_turns.add(re.sub(r"\W+", " ", fold(row["turns"][0]["user"])).strip())
                whole.add(key(row["turns"]))
                kept.append(row)
    counts["existing_kept"] = len(kept)
    records = []
    for file in raw_files:
        records += [json.loads(line) for line in Path(file).read_text().splitlines() if line]
    for r in records:
        counts["teacher_in"] += 1
        counts["in_" + r["seed"]["lang"]] += 1
        conv, reason = parse(r)
        if reason is None:
            reason = check(conv, r["seed"]["lang"], tests)
        if reason is None and verdict and not verdict.get(r["seed"]["id"], "").startswith("GOOD"):
            reason = "grader_flagged" if r["seed"]["id"] in verdict else "grader_missing"
        if reason is None:
            first = re.sub(r"\W+", " ", fold(conv[0]["user"])).strip()
            if first in first_turns or key(conv) in whole:
                reason = "duplicate"
            else:
                first_turns.add(first)
                whole.add(key(conv))
        if reason:
            counts["drop_" + reason] += 1
            counts[f"drop_{reason}_{r['seed']['lang']}"] += 1
            continue
        s = r["seed"]
        counts["kept_" + s["lang"]] += 1
        kept.append(dict(id=s["id"], lang=s["lang"], source="teacher", task=s["task"], topic=s["topic"], turns=conv))
    counts["teacher_kept"] = len(kept) - counts["existing_kept"]
    for r in arithmetic:
        counts["arithmetic_in"] += 1
        reason = check(r["turns"], r["lang"], tests, language=False)
        if reason is None:
            first = re.sub(r"\W+", " ", fold(r["turns"][0]["user"])).strip()
            if first in first_turns:
                reason = "duplicate"
            first_turns.add(first)
        if reason:
            counts["arithmetic_drop_" + reason] += 1
            continue
        counts["arithmetic_kept"] += 1
        kept.append(r)
    for r in programmatic:
        counts["programmatic_in"] += 1
        counts[f"programmatic_in_{r['kind']}"] += 1
        reason = check(r["turns"], r["lang"], tests, language=False)
        if reason is None and key(r["turns"]) in whole:
            reason = "duplicate"
        if reason:
            counts["programmatic_drop_" + reason] += 1
            counts[f"programmatic_drop_{r['kind']}_{reason}"] += 1
            continue
        whole.add(key(r["turns"]))
        counts["programmatic_kept"] += 1
        counts[f"programmatic_kept_{r['kind']}"] += 1
        counts[f"programmatic_kept_{r['lang']}"] += 1
        kept.append(r)
    for c in kept:
        if "split" in c:
            continue
        c["split"] = "dev" if int.from_bytes(hashlib.sha256(c["id"].encode()).digest()[:4], "little") % 100 < 3 else "train"
    by_lang = Counter((c["source"], c["lang"], c["split"]) for c in kept)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in kept))
    report.write_text(
        json.dumps(
            dict(
                counts=dict(counts),
                kept=len(kept),
                by_source_lang_split={"/".join(k): v for k, v in sorted(by_lang.items())},
                test_files=list(TEST_FILES),
                test_files_sha256=test_digest,
                test_turns=len(tests),
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps(dict(counts), indent=1), len(kept), flush=True)


def main():
    from .arithmetic import generate

    p = argparse.ArgumentParser()
    p.add_argument("raw", nargs="*", type=Path)
    p.add_argument("--arithmetic", type=int, default=3000)
    p.add_argument("--existing", nargs="*", type=Path, default=[], help="filtered conversation files from an earlier round, kept as they are")
    p.add_argument("--programmatic", type=int, default=0, help="number of exact programmatic dialogues to add")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--grades", nargs="*", type=Path, default=[], help="JSONL verdicts from the teacher grading pass")
    p.add_argument("--no-benchmarks", action="store_true", help="skip the XQuAD and Belebele question lists (offline smoke tests only)")
    a = p.parse_args()
    tests, digest = load_tests([] if a.no_benchmarks else benchmark_questions())
    from .dialogues import generate as dialogues

    run(a.raw, generate(a.arithmetic) if a.arithmetic else [], a.out, a.report, tests, digest, a.grades, a.existing, dialogues(a.programmatic) if a.programmatic else [])


if __name__ == "__main__":
    main()
