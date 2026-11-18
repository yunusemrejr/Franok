"""Multi-turn conversation prefixes and system-conditioned examples, with tree-level splits."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from .prepare import PARENT, ROOT, tokenizer_path

SOURCES = {
    "turkish-dialogue": (
        "MarulAI/Marul-SFT-TR",
        "6751921b6b50fdad26906e2e3c6a671a8f6b5f5e",
        "marul_sft_tr.jsonl",
    ),
    "english-dialogue": (
        "HuggingFaceH4/ultrachat_200k",
        "8049631c405ae6576f93f445c6b8166f76f5505a",
        "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
    ),
    "short-dialogue": (
        "HuggingFaceTB/smol-smoltalk",
        "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        "data/train-00001-of-00004.parquet",
    ),
}
DEFAULT = {
    "en": "You are Franok. Reply in English. Answer the latest message directly and use earlier turns when relevant.",
    "tr": "Sen Franok'sun. Türkçe yanıt ver. Son mesaja doğrudan cevap ver ve gerektiğinde önceki konuşmayı kullan.",
}


def examples(path, lang, source):
    if path.endswith(".jsonl"):
        with open(path) as f:
            yield from (json.loads(s) for s in f)
    else:
        for b in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from b.to_pylist()


def prepare():
    tok = Tokenizer.from_file(tokenizer_path())
    ROOT.mkdir(exist_ok=True, parents=True)
    for source, (repo, rev, file) in SOURCES.items():
        marker = ROOT / (source + ".json")
        if marker.exists():
            continue
        lang = "tr" if source == "turkish-dialogue" else "en"
        paths = {}
        idxs = {}
        counts = {}
        path = hf_hub_download(repo, file, repo_type="dataset", revision=rev)
        try:
            for i, row in enumerate(examples(path, lang, source)):
                ms = row["messages"]
                if any(
                    m.get("tool_calls")
                    or m["role"] not in ("system", "assistant", "user")
                    for m in ms
                ):
                    continue
                system = next((m["content"] for m in ms if m["role"] == "system"), None)
                seq = [m for m in ms if m["role"] != "system"]
                if len(seq) < 2 or any(
                    m["role"] != ("user", "assistant")[j % 2]
                    or not (m.get("content") or "").strip()
                    for j, m in enumerate(seq)
                ):
                    continue
                digest = int.from_bytes(
                    hashlib.sha256(seq[0]["content"].encode()).digest()[:4], "little"
                )
                key = (
                    ("val" if digest % 160 == 0 else "train")
                    + "-"
                    + lang
                    + "-"
                    + source
                )
                if key not in paths:
                    paths[key] = (ROOT / (key + ".bin")).open("wb")
                    idxs[key] = []
                    counts[key] = 0
                prefix = [1, 5 if lang == "en" else 6]
                if system and i % 3 != 0:
                    prefix += tok.encode(system[:1000]).ids[:150] + [2]
                elif i % 4:
                    prefix += tok.encode(DEFAULT[lang]).ids + [2]
                tokens = tok.encode_batch(
                    [m["content"].strip()[:7000] for m in seq[:8]]
                )
                for j in range(0, len(tokens) - 1, 2):
                    # Preserve ordinary short exchanges as well as long follow-ups.
                    question = tokens[j].ids
                    answer = tokens[j + 1].ids
                    if len(question) > 600 or len(answer) > 650:
                        break
                    prompt = prefix + [3, *question, 4]
                    if len(prompt) > 900:
                        break
                    body = prompt + answer + [2]
                    start = counts[key]
                    idxs[key].append(
                        (start, start + len(prompt) - 1, start + len(body))
                    )
                    paths[key].write(np.asarray(body, dtype="<u2").tobytes())
                    counts[key] += len(body)
                    prefix = body
                if sum(counts.values()) > 24000000:
                    break
        finally:
            for f in paths.values():
                f.close()
        for key, view in idxs.items():
            np.save(ROOT / (key + ".idx.npy"), np.asarray(view, dtype="<i8"))
        marker.write_text(
            json.dumps(
                dict(
                    source=SOURCES[source],
                    tokens=counts,
                    rows={k: len(v) for k, v in idxs.items()},
                ),
                indent=2,
            )
        )
        print(source, counts, flush=True)


if __name__ == "__main__":
    prepare()
