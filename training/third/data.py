"""Prepare disjoint, progressively available shards of open English and Turkish web text."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from tokenizers import Tokenizer

PARENT = "db3e8a10bd3fcefde40acdc25ba7c6dc59cde32b"
ROOT = Path("work/third/data")
SOURCES = {
    "en": (
        "HuggingFaceFW/fineweb-edu",
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "sample/10BT/000_00000.parquet",
    ),
    "tr": (
        "HuggingFaceFW/fineweb-2",
        "af9c13333eb981300149d5ca60a8e9d659b276b9",
        "data/tur_Latn/train/000_00000.parquet",
    ),
}


STORIES = {
    "en": (
        "roneneldan/TinyStories",
        "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    ),
    "tr": (
        "esat-krky/TinyStories_Turkish",
        "f039de0f75a000ccc62a6eff43e30eb5d8576c34",
        "train/data-00000-of-00001.arrow",
    ),
}


def tokenizer():
    path = os.environ.get("FRANOK_TOKENIZER") or hf_hub_download(
        "yunusemrejr/Franok", "tokenizer.json", revision=PARENT
    )
    return Tokenizer.from_file(path)


def prepare(lang, limit, shard_tokens, stream, collection="web"):
    root = ROOT / lang
    root.mkdir(exist_ok=True, parents=True)
    base_receipt = root / "receipt.json"
    suffix = "" if collection == "web" else collection + "-"
    receipt = base_receipt if not suffix else root / (collection + ".json")
    skip = 0
    if collection == "web-more":
        skip = sum(json.loads(base_receipt.read_text())["docs"].values())
    if receipt.exists() and json.loads(receipt.read_text())["target"] == limit:
        return
    repo, rev, fname = (STORIES if collection == "stories" else SOURCES)[lang]
    tok = tokenizer()
    counts = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    buffers = {"train": [], "val": []}
    shards = {"train": 0, "val": 0}
    seen = set()

    def flush(split):
        if not buffers[split]:
            return
        file = root / f"{split}-{suffix}{shards[split]:05d}.bin"
        tmp = file.with_suffix(".part")
        array = np.asarray(buffers[split], dtype="<u2")
        array.tofile(tmp)
        tmp.replace(file)
        counts[split] += len(array)
        buffers[split].clear()
        shards[split] += 1

    if stream:
        path = f"datasets/{repo}@{rev}/{fname}"
        source = HfFileSystem().open(path, "rb", block_size=16 << 20)
    else:
        source = open(
            hf_hub_download(repo, fname, repo_type="dataset", revision=rev), "rb"
        )
    with source:
        columns = (
            ["text"] if collection == "stories" else ["text", "url", "language_score"]
        )
        batches = (
            pa.ipc.open_stream(source)
            if fname.endswith(".arrow")
            else pq.ParquetFile(source).iter_batches(batch_size=128, columns=columns)
        )
        for batch in batches:
            texts, splits = [], []
            for row in batch.to_pylist():
                text = row["text"].strip()
                if len(text) < 180 or row.get("language_score", 1.0) < 0.75:
                    continue
                digest = hashlib.blake2b(text.encode(), digest_size=12).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                if len(seen) <= skip:
                    continue
                group = hashlib.blake2b(
                    (row.get("url") or digest.hex()).encode(), digest_size=8
                ).digest()
                splits.append(
                    "val" if int.from_bytes(group, "little") % 200 == 0 else "train"
                )
                texts.append(text[:24000])
            for split, encoded in zip(splits, tok.encode_batch(texts)):
                buffers[split].extend([1, 5 if lang == "en" else 6, *encoded.ids, 2])
                docs[split] += 1
                if len(buffers[split]) >= (shard_tokens if split == "train" else 8192):
                    flush(split)
            if counts["train"] + len(buffers["train"]) >= limit:
                break
    for split in buffers:
        flush(split)
    if not counts["val"] or counts["train"] < min(limit, 20000):
        raise RuntimeError(f"Insufficient {lang} corpus: {counts}")
    receipt.write_text(
        json.dumps(
            dict(
                source=(repo, rev, fname),
                skipped_unique_documents=skip,
                docs=docs,
                tokens=counts,
                target=limit,
                shards=shards,
            ),
            indent=2,
        )
        + "\n"
    )
    print(lang, docs, counts, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("language", choices=SOURCES)
    p.add_argument("--tokens", type=int, default=180000000)
    p.add_argument("--shard-tokens", type=int, default=1000000)
    p.add_argument("--stream", action="store_true")
    p.add_argument(
        "--collection", choices=("web", "web-more", "stories"), default="web"
    )
    a = p.parse_args()
    prepare(a.language, a.tokens, a.shard_tokens, a.stream, a.collection)
