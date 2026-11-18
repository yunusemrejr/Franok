"""Pack open text and instruction examples and sample rankable correct/foil pairs."""

import numpy as np
import torch
from .prepare import ROOT

TEXT = {"en": {"wiki-en": 0.5, "school": 0.32, "stories": 0.18}, "tr": {"wiki-tr": 1}}
CHAT = {
    "en": {
        "natural": 0.34,
        "parallel": 0.17,
        "facts-en": 0.12,
        "reading-en": 0.12,
        "translation": 0.05,
        "aya": 0.03,
        "logic": 0.10,
        "math": 0.07,
    },
    "tr": {
        "parallel": 0.31,
        "aya": 0.16,
        "facts-tr": 0.18,
        "reading-tr": 0.11,
        "translation": 0.10,
        "logic": 0.14,
    },
}
DIALOGUE = {
    "en": {
        "english-dialogue": 0.17,
        "short-dialogue": 0.17,
        "everyday-en": 0.32,
        "practice": 0.08,
        "natural": 0.06,
        "facts-en": 0.06,
        "reading-en": 0.05,
        "translation": 0.03,
        "aya": 0.02,
        "logic": 0.04,
    },
    "tr": {
        "turkish-dialogue": 0.28,
        "everyday-tr": 0.29,
        "native-chat": 0.10,
        "practice": 0.08,
        "aya": 0.06,
        "parallel": 0.05,
        "facts-tr": 0.04,
        "reading-tr": 0.04,
        "translation": 0.03,
        "logic": 0.03,
    },
}
GROUND = {
    "en": {"logic": 0.45, "reading-en": 0.3, "facts-en": 0.15, "translation": 0.1},
    "tr": {"logic": 0.45, "reading-tr": 0.3, "facts-tr": 0.15, "translation": 0.1},
}


class Corpus:
    def __init__(self, split):
        self.data = {}
        self.views = {}
        for f in ROOT.glob(split + "-*.bin"):
            key = f.stem[len(split) + 1 :]
            self.data[key] = np.memmap(f, mode="r", dtype="<u2")
            self.views[key] = np.load(f.with_suffix(".idx.npy"), mmap_mode="r")
        self.pools = {}
        for kind, table in [
            ("text", TEXT),
            ("chat", CHAT),
            ("ground", GROUND),
            ("dialogue", DIALOGUE),
        ]:
            for lang, weights in table.items():
                keys = []
                mass = []
                for source, w in weights.items():
                    k = lang + "-" + source
                    if k in self.views and len(self.views[k]) > 1:
                        keys.append(k)
                        mass.append(w)
                if not keys:
                    raise ValueError(f"Empty {split} {kind} {lang}")
                self.pools[kind, lang] = keys, np.cumsum(mass) / sum(mass)

    def choose(self, kind, lang, rng):
        keys, cdf = self.pools[kind, lang]
        return keys[np.searchsorted(cdf, rng.random())]

    def example(self, key, rng, room):
        data, view = self.data[key], self.views[key]
        # Prefer records that fit; long records keep the question and answer boundary.
        for _ in range(6):
            start, mid, end = view[rng.integers(len(view))]
            if end - start <= room:
                break
        a = data[mid + 1 : end]
        prefix = data[start : mid + 1]
        if len(prefix) + len(a) > room:
            pmax = max(4, room - min(len(a), max(12, room // 3)))
            if len(prefix) > pmax:
                prefix = np.r_[prefix[:3], prefix[-(pmax - 3) :]]
            a = a[: room - len(prefix)]
        return prefix, a

    def batch(self, rng, b, l, stage):
        x = np.zeros((b, l + 1), dtype=np.int64)
        y = np.full((b, l), -100, dtype=np.int64)
        for i in range(b):
            lang = "en" if i % 2 == 0 else "tr"
            if stage == "lm":
                kind = "text" if rng.random() < 0.54 else "chat"
                key = self.choose(kind, lang, rng)
                data = self.data[key]
                offset = rng.integers(len(data) - l)
                x[i] = data[offset : offset + l + 1]
                y[i] = x[i, 1:]
                continue
            # Preserve natural text generation throughout instruction tuning.
            if stage == "sft" and rng.random() < 0.12:
                data = self.data[self.choose("text", lang, rng)]
                offset = rng.integers(len(data) - l)
                x[i] = data[offset : offset + l + 1]
                y[i] = x[i, 1:]
                continue
            at = 0
            while l - at > 24:
                kind = (
                    "dialogue"
                    if stage in ("dialogue", "evidence")
                    or (stage == "sft" and rng.random() < 0.65)
                    else "chat"
                )
                key = self.choose(kind, lang, rng)
                prefix, answer = self.example(key, rng, l + 1 - at)
                x[i, at : at + len(prefix)] = prefix
                at += len(prefix)
                x[i, at : at + len(answer)] = answer
                y[i, at - 1 : at + len(answer) - 1] = answer
                at += len(answer)
        return torch.from_numpy(x[:, :-1].copy()).cuda(), torch.from_numpy(y).cuda()

    def pairs(self, rng, b, l):
        # Construct a distractor from another record in the same language/source.
        x = np.zeros((2 * b, l + 1), dtype=np.int64)
        y = np.full((2 * b, l), -100, dtype=np.int64)
        for i in range(b):
            lang = "en" if i % 2 == 0 else "tr"
            key = self.choose("ground", lang, rng)
            for _ in range(40):
                prefix, good = self.example(key, rng, l + 1)
                if 1 < len(good) <= min(100, l // 3) and good[-1] == 2:
                    break
            for _ in range(40):
                _, bad = self.example(key, rng, l + 1)
                if not np.array_equal(bad, good) and abs(len(bad) - len(good)) <= max(
                    4, len(good) // 2
                ):
                    break
            room = l + 1 - len(prefix)
            for j, tail in [(i, good), (i + b, bad[:room])]:
                x[j, : len(prefix)] = prefix
                x[j, len(prefix) : len(prefix) + len(tail)] = tail
                y[j, len(prefix) - 1 : len(prefix) - 1 + len(tail)] = tail
        return torch.from_numpy(x[:, :-1].copy()).cuda(), torch.from_numpy(y).cuda()
