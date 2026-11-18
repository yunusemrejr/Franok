"""Select short, complete replies and follow-ups from additional open dialogue records."""

import hashlib
import json

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from training.extended.dialogue import SOURCES

SECOND_SHARD = "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet"
ORIGINS = {
    "en": (*SOURCES["english-dialogue"][:2], SECOND_SHARD),
    "tr": SOURCES["turkish-dialogue"],
}


def records():
    for lang, (repo, rev, file) in ORIGINS.items():
        path = hf_hub_download(repo, file, revision=rev, repo_type="dataset")
        if lang == "en":
            source = (
                row
                for batch in pq.ParquetFile(path).iter_batches(batch_size=128)
                for row in batch.to_pylist()
            )
        else:
            stream = open(path, encoding="utf8")
            source = (json.loads(line) for i, line in enumerate(stream) if i >= 60000)
        counts = 0
        try:
            for row in source:
                if row.get("tools"):
                    continue
                ms = [m for m in row["messages"] if m["role"] != "system"]
                if len(ms) < 2 or any(
                    m["role"] != ("user", "assistant")[i % 2]
                    or m.get("tool_calls")
                    or not (m.get("content") or "").strip()
                    for i, m in enumerate(ms)
                ):
                    continue
                # Match the source grouping used in the original multi-turn importer.
                digest = int.from_bytes(
                    hashlib.sha256(ms[0]["content"].encode()).digest()[:4], "little"
                )
                split = "val" if digest % 160 == 0 else "train"
                texts = [m["content"].strip() for m in ms]
                for j in range(1, len(texts), 2):
                    q, answer = texts[j - 1 : j + 1]
                    if not (
                        6 <= len(q) <= 850
                        and 12 <= len(answer) <= 850
                        and len(answer.split()) <= 100
                    ):
                        continue
                    if any(
                        t in answer.casefold()
                        for t in (
                            "as an ai language model",
                            "<think>",
                            "</think>",
                            "yedikule",
                            "marul ai",
                            chr(96) * 3,
                        )
                    ):
                        continue
                    # If the antecedent does not fit, omit the example instead of training on an orphan question.
                    pre = texts[max(0, j - 5) : j + 1]
                    if sum(map(len, pre)) > 4200:
                        continue
                    yield split, lang, "short-replies", pre, None
                    counts += 1
                if counts >= 40000:
                    break
        finally:
            if lang == "tr":
                stream.close()
