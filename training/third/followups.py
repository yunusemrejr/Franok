"""Pair a reply with an earlier, different reply from the same dialogue.

Each stored pair consists of two offsets into an existing open corpus. A high
overlap between the questions or replies makes the negative ambiguous, so skip it.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("work/third/followups")
SOURCES = {
    "en": (
        "english-dialogue",
        "short-dialogue",
        "everyday-en",
        "short-replies",
        "human",
    ),
    "tr": (
        "turkish-dialogue",
        "native-chat",
        "everyday-tr",
        "short-replies",
        "turkish-tasks",
    ),
}


def overlap(a, b):
    left, right = set(a), set(b)
    return len(left & right) / max(1, len(left | right))


def build():
    ROOT.mkdir(exist_ok=True, parents=True)
    receipt = {}
    for split in ("train", "val"):
        for lang, names in SOURCES.items():
            for name in names:
                key = f"{split}-{lang}-{name}"
                if name == "human":
                    # This source already contains human-ranked responses.
                    continue
                roots = [
                    Path("work/extended/data"),
                    Path("work/third/tasks"),
                    Path("work/third/short-replies"),
                ]
                root = next(r for r in roots if (r / (key + ".idx.npy")).exists())
                index = np.load(root / (key + ".idx.npy"), mmap_mode="r")
                data = np.memmap(root / (key + ".bin"), mode="r", dtype="<u2")
                alt = np.zeros((len(index), 2), dtype="<i8")
                used = 0
                for j, (start, mark, end) in enumerate(index):
                    pre = data[start:mark]
                    positive = data[mark + 1 : end - 1]
                    if not 6 <= len(positive) <= 200 or end - start > 1025:
                        continue
                    users = np.flatnonzero(pre == 3)
                    if len(users) < 2:
                        continue
                    earlier = pre[users[-2] + 1 : users[-1]]
                    bounds = np.flatnonzero(earlier == 4)
                    if len(bounds) != 1 or earlier[-1] != 2:
                        continue
                    marker = bounds[0]
                    old = earlier[marker + 1 : -1]
                    if (
                        not 6 <= len(old) <= 200
                        or not 0.4 <= len(old) / len(positive) <= 2.5
                    ):
                        continue
                    if len(pre) + 1 + len(old) > 1024 or overlap(old, positive) > 0.60:
                        continue
                    if overlap(earlier[:marker], pre[users[-1] + 1 :]) > 0.65:
                        continue
                    alt[j] = (start + users[-2] + 1 + marker + 1, start + users[-1])
                    used += 1
                np.save(ROOT / (key + ".idx.npy"), alt)
                receipt[key] = dict(source_rows=len(index), paired_rows=used)
    (ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
