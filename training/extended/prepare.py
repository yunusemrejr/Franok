"""Pinned open corpora for the extended bilingual run. No pretrained tokenizer or teacher."""

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

PARENT = "7633d06180ebbc836af02c590f96f8ac28a4829a"
ROOT = Path("work/extended/data")
WIKI = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
COSMO = "0ae6ec63f91742bd2d1eaef4f02232c55d719385"
SOURCES = {
    "wiki-en": (
        "wikimedia/wikipedia",
        WIKI,
        "20231101.en/train-00001-of-00041.parquet",
    ),
    "wiki-tr": (
        "wikimedia/wikipedia",
        WIKI,
        "20231101.tr/train-00001-of-00002.parquet",
    ),
    "school": (
        "HuggingFaceTB/cosmopedia",
        COSMO,
        "data/openstax/train-00000-of-00002.parquet",
    ),
    "stories": (
        "HuggingFaceTB/cosmopedia",
        COSMO,
        "data/stories/train-00000-of-00043.parquet",
    ),
    "natural": (
        "HuggingFaceTB/smol-smoltalk",
        "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        "data/train-00000-of-00004.parquet",
    ),
    "parallel": (
        "malhajar/OpenOrca-tr",
        "e15f1876b52e85b229a3fedcd2091ce1e26ca787",
        "data/train-00000-of-00018-78b414d81b28197c.parquet",
    ),
    "aya": (
        "CohereForAI/aya_dataset",
        "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
        "data/train-00000-of-00001.parquet",
    ),
    "reading-en": (
        "rajpurkar/squad",
        "7b6d24c440a36b6815f21b70d25016731768db1f",
        "plain_text/train-00000-of-00001.parquet",
    ),
    "reading-tr": (
        "GoktugD/turkish-extractive-qa-1.5m",
        "8d1453c6643e6c76d6ecae9a63094f47748b474b",
        "data/train-00001-of-00015.parquet",
    ),
    "logic": (
        "yunusemrejr/Franok-Session-Tasks",
        "db21b480885e9f8194be8615d5d6234372ff672c",
        "train.jsonl",
    ),
    "math": (
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
        "main/train-00000-of-00001.parquet",
    ),
}
SOURCES.update(
    {
        "facts-en": SOURCES["wiki-en"],
        "facts-tr": SOURCES["wiki-tr"],
        "translation": SOURCES["parallel"],
    }
)


# records are (language, group, text segments). Alternating chat segments start with a user.
def records(name, row, i):
    if name.startswith("facts"):
        lang = name[-2:]
        title = row["title"].strip()
        # Lead sentences retain the original wording. Titles come from the same article group.
        lead = re.split(r"(?<=[.!?])\s+|\n", row["text"].strip(), maxsplit=1)[0]
        if (
            55 <= len(lead) <= 420
            and len(title) < 80
            and lead.endswith((".", "?", "!"))
        ):
            question = (
                (
                    f"What can you tell me about {title}?"
                    if i % 2
                    else f"Briefly describe {title}."
                )
                if lang == "en"
                else (
                    f"{title} hakkında kısaca bilgi verir misin?"
                    if i % 2
                    else f"{title} nedir?"
                )
            )
            yield lang, row["id"], [question, lead]
    elif name == "translation":
        en, tr = row["question"].strip(), row["question-turkish"].strip()
        if 12 < len(en) < 260 and 12 < len(tr) < 310 and en != tr:
            yield "en", row["id"], ["Translate into English: " + tr, en]
            yield "tr", row["id"], ["Türkçeye çevir: " + en, tr]
    elif name.startswith("wiki"):
        yield name[-2:], row["id"], [row["text"][:16000]]
    elif name in ("stories", "school"):
        yield "en", str(i), [row["text"][:12000]]
    elif name == "natural":
        turns = [
            m["content"] for m in row["messages"] if m["role"] in ("user", "assistant")
        ]
        if len(turns) >= 2 and len(turns) % 2 == 0:
            yield "en", turns[0], turns[:6]
    elif name == "parallel":
        yield "en", row["id"], [row["question"], row["response"]]
        yield "tr", row["id"], [row["question-turkish"], row["response-turkish"]]
    elif name == "aya":
        code = str(row["language_code"]).lower()
        lang = {"en": "en", "eng": "en", "tr": "tr", "tur": "tr"}.get(code)
        if lang:
            yield lang, row["inputs"], [row["inputs"], row["targets"]]
    elif name.startswith("reading"):
        answer = row["answers"]["text"][0] if name.endswith("en") else row["answer"]
        yield (
            name[-2:],
            row["context"],
            [row["context"] + "\n" + row["question"], answer],
        )
    elif name == "logic":
        yield row["language"], row["id"], [row["prompt"], row["answer"]]
    elif name == "math":
        yield (
            "en",
            row["question"],
            [row["question"], row["answer"].replace("#### ", "Answer: ")],
        )


def chunks(path):
    if str(path).endswith(".jsonl"):
        with open(path) as f:
            batch = []
            for line in f:
                batch.append(json.loads(line))
                if len(batch) == 256:
                    yield batch
                    batch = []
            if batch:
                yield batch
    else:
        for b in pq.ParquetFile(path).iter_batches(batch_size=192):
            yield b.to_pylist()


def tokenizer_path():
    local = os.environ.get("FRANOK_TOKENIZER")
    return local or hf_hub_download(
        "yunusemrejr/Franok", "tokenizer.json", revision=PARENT
    )


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    parent = tokenizer_path()

    def build(entry):
        name, (repo, rev, file) = entry
        complete = ROOT / (name + ".json")
        if complete.exists():
            return
        tok = Tokenizer.from_file(parent)
        path = hf_hub_download(repo, file, revision=rev, repo_type="dataset")
        limit = {
            "wiki-en": 45000000,
            "wiki-tr": 60000000,
            "school": 19000000,
            "stories": 11000000,
            "parallel": 65000000,
            "natural": 20000000,
        }.get(name, 25000000)
        opened = {}
        counts = {}
        indexes = {}
        seen = set()
        total = 0
        idx = 0
        try:
            for batch in chunks(path):
                recs = []
                for row in batch:
                    for lang, group, parts in records(name, row, idx):
                        if not all(isinstance(s, str) and s.strip() for s in parts):
                            continue
                        parts = [p.strip()[:18000] for p in parts]
                        if len(parts) == 1 and len(parts[0]) < 120:
                            continue
                        digest = hashlib.blake2b(
                            "\0".join(parts).encode(), digest_size=12
                        ).digest()
                        if digest in seen:
                            continue
                        seen.add(digest)
                        family = {
                            "facts-en": "wiki-en",
                            "facts-tr": "wiki-tr",
                            "translation": "parallel",
                        }.get(name, name)
                        val = (
                            int.from_bytes(
                                hashlib.sha256((family + str(group)).encode()).digest()[
                                    :4
                                ],
                                "little",
                            )
                            % 200
                            == 0
                        )
                        recs.append((lang, "val" if val else "train", parts))
                    idx += 1
                if not recs:
                    continue
                encoded = iter(tok.encode_batch([p for _, _, ps in recs for p in ps]))
                for lang, split, parts in recs:
                    prefix = [1, 5 if lang == "en" else 6]
                    middle = 0
                    for j in range(len(parts)):
                        ids = next(encoded).ids
                        if len(parts) > 1:
                            if j == len(parts) - 1:
                                middle = len(prefix)
                            prefix.append(3 if j % 2 == 0 else 4)
                        prefix.extend(ids)
                        if len(parts) == 1 or j % 2:
                            prefix.append(2)
                    key = split + "-" + lang + "-" + name
                    if key not in opened:
                        opened[key] = (ROOT / (key + ".bin")).open("wb")
                        indexes[key] = []
                        counts[key] = 0
                    start = counts[key]
                    indexes[key].append((start, start + middle, start + len(prefix)))
                    opened[key].write(np.asarray(prefix, dtype="<u2").tobytes())
                    counts[key] += len(prefix)
                    total += len(prefix)
                if total >= limit:
                    break
        finally:
            for handle in opened.values():
                handle.close()
        for key, idxs in indexes.items():
            np.save(ROOT / (key + ".idx.npy"), np.asarray(idxs, dtype="<i8"))
        complete.write_text(
            json.dumps(
                dict(
                    source=SOURCES[name],
                    tokens=counts,
                    rows={k: len(v) for k, v in indexes.items()},
                ),
                indent=2,
            )
        )
        print(name, counts, flush=True)

    with ThreadPoolExecutor(int(os.getenv("FRANOK_DATA_WORKERS", "3"))) as pool:
        list(pool.map(build, SOURCES.items()))


if __name__ == "__main__":
    prepare()
