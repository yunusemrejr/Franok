"""Prepare human conversations and passage questions with article/tree-level splits."""

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from training.extended.dialogue import DEFAULT
from .data import tokenizer

ROOT = Path("work/third/instructions")
SQUAD = {
    "en": (
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002",
    ),
    "tr": (
        "https://tulap.cmpe.bogazici.edu.tr/server/api/core/bitstreams/b261bb3c-6348-4b15-9ce4-7987419db1c9/content",
        "94670715bb17cc570fea34e8d730d4fdb0569100d7927e3edf78d1beb854f5d2",
    ),
}
OASST = (
    "OpenAssistant/oasst2",
    "179dd21fc55192153d94adb0e0ce8f69e222bf75",
    "2023-11-05_oasst2_all.messages.jsonl.gz",
)


COMMONSENSE = (
    "tau/commonsense_qa",
    "94630fe30dad47192a8546eb75f094926d47e155",
    "data/train-00000-of-00001.parquet",
)
TURKISH_CHAT = (
    "cisimcik/turkish-chat-max-25k",
    "7cf29049fd24970df3a32fb19a49eedd937128b9",
)
TURKISH_TASKS = (
    "casual_conversation",
    "creative_writing",
    "extractive_qa",
    "grammatical_error_correction",
    "information_extraction",
    "instruction_following",
    "math_and_logic",
    "multiple_choice_qa",
    "natural_language_inference",
    "open_qa",
    "planning",
    "summarization",
    "text_classification",
)


def task_records():
    repo, rev, fname = COMMONSENSE
    path = hf_hub_download(repo, fname, revision=rev, repo_type="dataset")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            question = row["question"].strip()
            split = split_for("commonsense:" + row["question_concept"])
            choices = list(zip(row["choices"]["label"], row["choices"]["text"]))
            gold = next(t for l, t in choices if l == row["answerKey"])
            rng = random.Random(hashed(row["id"]))
            rng.shuffle(choices)
            choices = [t for _, t in choices]
            false = rng.choice([t for t in choices if t != gold])
            variant = rng.randrange(4)
            if variant:
                options = [f"{chr(65+i)}) {ans}" for i, ans in enumerate(choices)]
                question += "\n" + "\n".join(options)
                if variant == 1:
                    question += "\nReply with just the letter."
                    gold, false = [chr(65 + choices.index(x)) for x in (gold, false)]
            yield split, "en", "commonsense", [question, gold], false
    repo, rev = TURKISH_CHAT
    for task in TURKISH_TASKS:
        fname = f"data/{task}/train.parquet"
        path = hf_hub_download(repo, fname, revision=rev, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=96):
            for row in batch.to_pylist():
                if (
                    row["has_tools"]
                    or row["language_mode"] != "tr_only"
                    or row["prompt_language"] != "tr"
                ):
                    continue
                messages = [m for m in row["messages"] if m["role"] != "system"]
                if (
                    len(messages) % 2
                    or any(
                        m.get("tool_calls")
                        or m["role"] != ("user", "assistant")[j % 2]
                        or not (m["content"] or "").strip()
                        for j, m in enumerate(messages)
                    )
                    or not messages
                ):
                    continue
                texts = [m["content"].strip() for m in messages]
                split = split_for("chat-max:" + texts[0].casefold())
                for end in range(2, len(texts) + 1, 2):
                    if len(texts[end - 1]) <= 4500:
                        yield split, "tr", "turkish-tasks", texts[:end], None


def hashed(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def split_for(group):
    return "val" if hashed(group) % 40 == 0 else "train"


def passage_records(articles, lang):
    """Foils come from *other* questions about the same paragraph."""
    missing = {"en": "The passage does not say.", "tr": "Metin bunu belirtmiyor."}
    for article in articles:
        split = split_for("squad:" + article["title"].casefold())
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            if not context:
                continue
            qs = paragraph["qas"]
            paragraph_answers = {
                a["text"].strip() for q in qs for a in q["answers"] if a["text"].strip()
            }
            for q in qs:
                rng = random.Random(hashed(q["id"]))
                answers = {
                    a["text"].strip()
                    for a in q["answers"]
                    if a["text"].strip() in context
                }
                if q["is_impossible"]:
                    if rng.random() > 0.10:
                        continue
                    answer = missing[lang]
                elif answers:
                    answer = min(answers, key=lambda s: (len(s), s))
                else:
                    continue
                foil_pool = sorted(
                    a
                    for a in paragraph_answers
                    if a
                    and all(
                        a.casefold() not in g.casefold()
                        and g.casefold() not in a.casefold()
                        for g in answers
                        if g
                    )
                )
                # Prefer nearby lengths, breaking ties reproducibly.
                rng.shuffle(foil_pool)
                foil_pool.sort(key=lambda a: abs(len(a.split()) - len(answer.split())))
                foil = foil_pool[0] if foil_pool else None
                prompt = context + "\n\n" + q["question"].strip()
                if rng.random() < 0.19 and len(foil_pool) >= 2:
                    options = [answer, *foil_pool[:3]]
                    rng.shuffle(options)
                    prompt += "\n" + " / ".join(options)
                yield split, lang, "passages", [prompt, answer], foil


def conversations(path):
    messages = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if (
                row["lang"] not in ("en", "tr")
                or row["deleted"]
                or row["synthetic"]
                or not row["review_result"]
                or not row["text"].strip()
                or row["tree_state"] != "ready_for_export"
            ):
                continue
            labels = row.get("labels") or {}
            if any(
                labels.get(k, {}).get("value", 0) >= 0.5
                for k in ("pii", "lang_mismatch", "spam")
            ):
                continue
            messages[row["message_id"]] = row
    siblings = defaultdict(list)
    for row in messages.values():
        if row["role"] == "assistant":
            siblings[row["parent_id"]].append(row)
    for row in messages.values():
        if row["role"] != "assistant" or row["rank"] not in (0, None):
            continue
        chain = [row]
        while chain[-1]["parent_id"] in messages:
            chain.append(messages[chain[-1]["parent_id"]])
        if chain[-1]["parent_id"] is not None:
            continue
        chain.reverse()
        if (
            len(chain) % 2
            or len(chain) > 12
            or any(
                m["role"] != ("prompter", "assistant")[i % 2]
                or m["lang"] != row["lang"]
                for i, m in enumerate(chain)
            )
        ):
            continue
        turns = [m["text"].strip() for m in chain]
        if any(len(turn) > 7500 for turn in turns):
            continue
        lower = [
            m
            for m in siblings[row["parent_id"]]
            if row["rank"] == 0
            and (m["rank"] or 0) > 0
            and m["lang"] == row["lang"]
            and m["text"].strip() != turns[-1]
        ]
        foil = max(lower, key=lambda m: m["rank"])["text"].strip() if lower else None
        yield split_for("oasst:" + row["message_tree_id"]), row[
            "lang"
        ], "human", turns, foil


def read_squad(lang, local):
    url, checksum = SQUAD[lang]
    if local:
        data = local.read_bytes()
    else:
        with urlopen(url, timeout=120) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError(f"Unexpected upstream {lang} SQuAD checksum")
    if lang == "tr":
        data = gzip.decompress(data)
    return json.loads(data)["data"]


def prepare(args):
    root = ROOT if args.collection == "passages" else ROOT.parent / args.collection
    root.mkdir(exist_ok=True, parents=True)
    tok = tokenizer()
    sysids = {l: tok.encode(s).ids for l, s in DEFAULT.items()}
    files, indices, offsets = {}, defaultdict(list), Counter()
    foil_offsets, foils = Counter(), defaultdict(list)
    seen = set()
    rankable = Counter()
    role_ids = (3, 4)
    rng = random.Random(14307)
    repo, rev, fn = OASST

    def streams():
        if args.collection == "short-replies":
            from .short_dialogue import records

            yield from records()
            return
        if args.collection == "tasks":
            yield from task_records()
            return
        path = hf_hub_download(repo, fn, repo_type="dataset", revision=rev)
        yield from conversations(path)
        for lang in ("en", "tr"):
            yield from passage_records(read_squad(lang, getattr(args, lang)), lang)

    buffer = []
    with ExitStack() as stack:

        def record(batch):
            # One tokenizer call amortizes the parallel dispatcher over many short answers.
            sequences = [
                text
                for _, _, _, turns, foil in batch
                for text in turns + ([foil] if foil else [])
            ]
            encoded = iter(tok.encode_batch(sequences, add_special_tokens=False))
            for split, lang, kind, turns, foil in batch:
                texts = [next(encoded).ids for _ in turns]
                rejected = next(encoded).ids if foil else []
                key = f"{split}-{lang}-{kind}"
                uid = (
                    lang,
                    hashlib.blake2b("\0".join(turns).encode(), digest_size=12).digest(),
                )
                if uid in seen:
                    continue
                seen.add(uid)
                header = [1, 5 if lang == "en" else 6]
                if rng.random() < 0.70:
                    header += sysids[lang] + [2]
                # Remove only complete oldest exchanges if the remaining two turns fit.
                while (
                    len(texts) > 2
                    and sum(map(len, texts)) + len(header) + 2 * len(texts) > 1025
                ):
                    texts = texts[2:]
                pre = header.copy()
                for j, tokens in enumerate(texts[:-1]):
                    pre += [role_ids[j % 2], *tokens]
                    if j % 2:
                        pre.append(2)
                pre.append(4)
                body = pre + texts[-1] + [2]
                if len(body) > 1025 or not texts[-1] or not texts[-2]:
                    continue
                if key not in files:
                    files[key] = [
                        stack.enter_context((root / (key + s + ".partial")).open("wb"))
                        for s in (".bin", ".foil.bin")
                    ]
                indices[key].append(
                    (
                        offsets[key],
                        offsets[key] + len(pre) - 1,
                        offsets[key] + len(body),
                    )
                )
                files[key][0].write(np.asarray(body, dtype="<u2").tobytes())
                offsets[key] += len(body)
                if (
                    rejected
                    and len(pre) + len(rejected) + 1 <= 1025
                    and rejected != texts[-1]
                ):
                    rejected = rejected + [2]
                    rankable[key] += 1
                else:
                    rejected = []
                foils[key].append(
                    (foil_offsets[key], foil_offsets[key] + len(rejected))
                )
                files[key][1].write(np.asarray(rejected, dtype="<u2").tobytes())
                foil_offsets[key] += len(rejected)

        for item in streams():
            buffer.append(item)
            if len(buffer) == 192:
                record(buffer)
                buffer.clear()
        if buffer:
            record(buffer)
    for k, idx in indices.items():
        for suffix in (".bin", ".foil.bin"):
            (root / (k + suffix + ".partial")).replace(root / (k + suffix))
        np.save(root / (k + ".idx.npy"), np.asarray(idx, dtype="<i8"))
        np.save(root / (k + ".foil.idx.npy"), np.asarray(foils[k], dtype="<i8"))
    from .short_dialogue import ORIGINS

    result = dict(
        sources=(
            ORIGINS
            if args.collection == "short-replies"
            else (
                dict(oasst=OASST, squad=SQUAD)
                if args.collection == "passages"
                else dict(
                    commonsense=COMMONSENSE,
                    turkish_chat=TURKISH_CHAT,
                    tasks=TURKISH_TASKS,
                )
            )
        ),
        rows={k: len(v) for k, v in indices.items()},
        tokens=dict(offsets),
        rankable=dict(rankable),
    )
    (root / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--collection",
        choices=("passages", "tasks", "short-replies"),
        default="passages",
    )
    p.add_argument(
        "--en", type=Path, help="Local, checksum-verified SQuAD 2.0 train JSON"
    )
    p.add_argument(
        "--tr", type=Path, help="Local, checksum-verified SQuAD-TR train gzip"
    )
    prepare(p.parse_args())
