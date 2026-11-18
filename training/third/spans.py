"""Prepare word-aligned gold answer spans and impossible questions from SQuAD train splits."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from franok.span import boundaries, pieces

from .data import tokenizer
from .instructions import read_squad, split_for, hashed

ROOT = Path("work/third/spans")
MAX_SPAN = 48


def examples(articles, lang, tok, table):
    for article in articles:
        split = split_for("squad:" + article["title"].casefold())
        for passage in article["paragraphs"]:
            text = passage["context"].strip()
            if not text:
                continue
            ids = tok.encode(text).ids
            raw, ends, legal = boundaries(text, ids, table)
            spans = []
            for question in passage["qas"]:
                if question["is_impossible"]:
                    if hashed("span:" + question["id"]) % 100 >= 15:
                        continue
                    targets = (-1, -1)
                else:
                    targets = None
                    for a in question["answers"]:
                        answer = a["text"].strip()
                        position = a["answer_start"]
                        if not answer:
                            continue
                        if text[position : position + len(answer)] != answer:
                            position = text.find(answer)
                        if position < 0:
                            continue
                        beg = len(text[:position].encode())
                        end = beg + len(answer.encode())
                        i = int(np.searchsorted(ends, beg, side="right"))
                        j = int(np.searchsorted(ends, end, side="left"))
                        if j - i >= MAX_SPAN or not legal[i] or not legal[j + 1]:
                            continue
                        quote = (
                            raw[0 if i == 0 else ends[i - 1] : ends[j]].decode().strip()
                        )
                        if quote == answer:
                            targets = (i, j)
                            break
                    if targets is None:
                        continue
                spans.append((question["question"], targets))
            for (q, target), encoded in zip(
                spans, tok.encode_batch([q for q, _ in spans])
            ):
                pre = [1, 5 if lang == "en" else 6, 3, *encoded.ids, 2, 3]
                if len(q) > 1500 or len(pre) + len(ids) + 1 > 1024:
                    continue
                yield split, pre, ids, legal, target


def prepare(a):
    ROOT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    table = pieces(tok)
    result = {}
    for lang in ("en", "tr"):
        outputs = {s: [] for s in ("train", "val")}
        counts = Counter()
        with (ROOT / f"{lang}.bin").open("wb") as f, (ROOT / f"{lang}.legal").open(
            "wb"
        ) as mask:
            pos = 0
            for split, pre, ids, legal, target in examples(
                read_squad(lang, getattr(a, lang)), lang, tok, table
            ):
                body = [*pre, *ids, 4]
                f.write(np.array(body, dtype="<u2").tobytes())
                mask.write(bytes(len(pre)) + bytes(legal))
                i, j = target
                outputs[split].append((pos, len(pre), len(body), i, j))
                counts[split + ("-null" if i < 0 else "-answer")] += 1
                pos += len(body)
        for split, rows in outputs.items():
            np.save(ROOT / f"{lang}-{split}.npy", np.array(rows, dtype="<i8"))
        result[lang] = counts
    (ROOT / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


class SpanCorpus:
    def __init__(self, split):
        self.files = {}
        self.groups = {}
        for l in ("en", "tr"):
            ids = np.memmap(ROOT / f"{l}.bin", mode="r", dtype="<u2")
            legal = np.memmap(ROOT / f"{l}.legal", mode="r", dtype=np.uint8)
            index = np.load(ROOT / f"{l}-{split}.npy", mmap_mode="r")
            self.files[l] = ids, legal, index
            for lo, hi in ((0, 128), (128, 256), (256, 512), (512, 1024)):
                self.groups[l, hi] = np.flatnonzero(
                    (index[:, 2] > lo) & (index[:, 2] <= hi)
                )

    def at(self, lang, j):
        ids, legal, index = self.files[lang]
        pos, n, length, i, j = index[j]
        return (
            ids[pos : pos + length],
            int(n),
            legal[pos + n : pos + length],
            int(i),
            int(j),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--en", type=Path)
    p.add_argument("--tr", type=Path)
    prepare(p.parse_args())
