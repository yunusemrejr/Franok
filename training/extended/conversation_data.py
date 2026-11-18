"""Concise everyday exchanges, original correction practice and explicit role boundaries."""

import hashlib, json
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from .prepare import ROOT, PARENT, tokenizer_path
from .dialogue import DEFAULT

SOURCES = {
    "everyday-en": (
        "HuggingFaceTB/everyday-conversations-llama3.1-2k",
        "14f543216b9ba42b6b951dc5bd199460d193b162",
        "data/train_sft-00000-of-00001.parquet",
    ),
    "everyday-tr": (
        "SoAp9035/everyday-conversations-tur",
        "f54bf0a5497ac56b4c257d81186e07110bd4dbdf",
        "everyday-conversations.json",
    ),
    "native-chat": (
        "serhatkagans/turkish-chat-sft-v3",
        "e53f32ef4b6fa8be987c2f5dc462a2466586112f",
        "data/train-00000-of-00001.parquet",
    ),
    "practice": (
        "yunusemrejr/Franok-Session-Tasks",
        "2cc669f9af880921f64541efbd89847f9e89078b",
        "practice-train.jsonl",
    ),
}


def rows(file):
    if file.endswith(".jsonl"):
        with open(file) as f:
            yield from map(json.loads, f)
    elif file.endswith(".json"):
        with open(file) as f:
            yield from json.load(f)
    else:
        for b in pq.ParquetFile(file).iter_batches(batch_size=96):
            yield from b.to_pylist()


def main():
    ROOT.mkdir(exist_ok=True, parents=True)
    tok = Tokenizer.from_file(tokenizer_path())
    for name, (repo, rev, file) in SOURCES.items():
        marker = ROOT / (name + ".json")
        if marker.exists():
            continue
        files = {}
        idxs = {}
        sizes = {}
        seen = set()
        with ExitStack() as stack:
            for upstream in (
                [file, "practice-validation.jsonl"] if name == "practice" else [file]
            ):
                path = hf_hub_download(
                    repo, upstream, repo_type="dataset", revision=rev
                )
                for i, row in enumerate(rows(path)):
                    if (
                        row.get("thinking")
                        or name == "native-chat"
                        and row.get("kategori")
                        not in ["gunluk", "genel_kultur", "yaratici_yazma", "turkiye"]
                    ):
                        continue
                    seq = [
                        m
                        for m in (row.get("messages") or row.get("conversations"))
                        if m["role"] in ("user", "assistant")
                    ]
                    if len(seq) < 2 or any(
                        m["role"] != ("user", "assistant")[j % 2]
                        or not m["content"].strip()
                        for j, m in enumerate(seq)
                    ):
                        continue
                    lang = row.get("language") or (
                        "en" if name.endswith("-en") else "tr"
                    )
                    # Group by the first substantive question, since common greetings repeat.
                    group = seq[
                        2 if len(seq) >= 4 and len(seq[0]["content"]) < 30 else 0
                    ]["content"]
                    group_hash = int.from_bytes(
                        hashlib.sha256(group.encode()).digest()[:4], "little"
                    )
                    split = (
                        ("val" if "validation" in upstream else "train")
                        if name == "practice"
                        else ("val" if group_hash % 67 == 0 else "train")
                    )
                    key = split + "-" + lang + "-" + name
                    if key not in files:
                        files[key] = stack.enter_context(
                            (ROOT / (key + ".bin")).open("wb")
                        )
                        idxs[key] = []
                        sizes[key] = 0
                    prefix = [1, 5 if lang == "en" else 6]
                    if i % 8:
                        prefix += tok.encode(DEFAULT[lang]).ids + [2]
                    for j in range(0, len(seq) - 1, 2):
                        q = tok.encode(seq[j]["content"].strip()).ids
                        a = tok.encode(seq[j + 1]["content"].strip()).ids
                        pre = prefix + [3] + q + [4]
                        if len(a) > 380 or len(pre) + len(a) > 960:
                            break
                        body = pre + a + [2]
                        uid = tuple(body)
                        if uid not in seen:
                            seen.add(uid)
                            start = sizes[key]
                            idxs[key].append(
                                (start, start + len(pre) - 1, start + len(body))
                            )
                            files[key].write(np.asarray(body, dtype="<u2").tobytes())
                            sizes[key] += len(body)
                        prefix = body
        for key, idx in idxs.items():
            np.save(ROOT / (key + ".idx.npy"), np.asarray(idx, dtype="<i8"))
        marker.write_text(
            json.dumps(
                dict(
                    source=SOURCES[name],
                    tokens=sizes,
                    rows={k: len(v) for k, v in idxs.items()},
                ),
                indent=2,
            )
        )
        print(name, sizes, flush=True)


if __name__ == "__main__":
    main()
