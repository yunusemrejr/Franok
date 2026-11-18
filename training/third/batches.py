"""Bucket complete supervision records by length, keeping unrelated dialogues isolated."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import tokenizer
from training.extended.dialogue import DEFAULT

BOUNDARIES = (64, 128, 256, 512, 1024)
# Within each length band weights apply to records, separately for both languages.
WEIGHTS = {
    "en": {
        "human": 0.17,
        "short-replies": 0.12,
        "natural": 0.12,
        "english-dialogue": 0.10,
        "short-dialogue": 0.11,
        "everyday-en": 0.09,
        "passages": 0.14,
        "commonsense": 0.08,
        "math": 0.04,
        "logic": 0.045,
        "aya": 0.035,
        "practice": 0.025,
        "facts-en": 0.035,
    },
    "tr": {
        "turkish-dialogue": 0.17,
        "short-replies": 0.14,
        "turkish-tasks": 0.18,
        "native-chat": 0.07,
        "everyday-tr": 0.12,
        "passages": 0.16,
        "parallel": 0.07,
        "logic": 0.055,
        "aya": 0.055,
        "practice": 0.035,
        "facts-tr": 0.045,
    },
}
PREFERENCES = {
    "en": {
        "human": 0.23,
        "short-replies": 0.12,
        "passages": 0.25,
        "commonsense": 0.18,
        "english-dialogue": 0.14,
        "short-dialogue": 0.10,
        "everyday-en": 0.09,
    },
    "tr": {
        "passages": 0.35,
        "short-replies": 0.14,
        "turkish-dialogue": 0.23,
        "turkish-tasks": 0.16,
        "native-chat": 0.08,
        "everyday-tr": 0.14,
    },
}
SEARCH_PATHS = (
    Path("work/extended/data"),
    Path("work/third/instructions"),
    Path("work/third/tasks"),
    Path("work/third/short-replies"),
)


@dataclass
class Records:
    data: np.ndarray
    index: np.ndarray
    foils: np.ndarray | None
    foil_index: np.ndarray | None


class Conversations:
    def __init__(self, split):
        self.records = {}
        self.pools = {}
        self.system = {l: tokenizer().encode(s).ids + [2] for l, s in DEFAULT.items()}
        for lang, sources in WEIGHTS.items():
            for name in sources:
                stem = f"{split}-{lang}-{name}"
                paths = [p for p in SEARCH_PATHS if (p / (stem + ".idx.npy")).is_file()]
                if len(paths) != 1:
                    raise FileNotFoundError(
                        f"Expected one prepared {stem} corpus; found {paths}"
                    )
                root = paths[0]
                data = np.memmap(root / (stem + ".bin"), mode="r", dtype="<u2")
                index = np.load(root / (stem + ".idx.npy"), mmap_mode="r")
                f = root / (stem + ".foil.bin")
                foil = (
                    np.memmap(f, mode="r", dtype="<u2")
                    if f.exists() and f.stat().st_size
                    else None
                )
                fi = (
                    np.load(root / (stem + ".foil.idx.npy"), mmap_mode="r")
                    if foil is not None
                    else None
                )
                followup = Path("work/third/followups") / (stem + ".idx.npy")
                if followup.exists() and name in PREFERENCES[lang]:
                    previous = np.load(followup, mmap_mode="r")
                    if previous.shape != (len(index), 2):
                        raise ValueError("Follow-up offsets do not match " + stem)
                    foil, fi = data, previous
                self.records[lang, name] = Records(data, index, foil, fi)
                prefix = index[:, 1] - index[:, 0] + 1
                ordinary = index[:, 2] - index[:, 0] - 1
                has_foil = (
                    (fi[:, 1] > fi[:, 0])
                    if fi is not None
                    else np.zeros(len(index), dtype=bool)
                )
                for paired in (False, True):
                    sizes = (
                        np.maximum(ordinary, prefix + (fi[:, 1] - fi[:, 0]) - 1)
                        if paired and fi is not None
                        else ordinary
                    )
                    lower = 0
                    for upper in BOUNDARIES:
                        selected = (sizes > lower) & (sizes <= upper)
                        if paired:
                            selected &= has_foil
                        pool = np.flatnonzero(selected)
                        if len(pool):
                            self.pools[paired, lang, name, upper] = pool
                        lower = upper
        self.draws = {}
        for paired, weights in ((True, PREFERENCES), (False, WEIGHTS)):
            for lang, sources in weights.items():
                for upper in BOUNDARIES:
                    names, mass = [], []
                    for name, w in sources.items():
                        pool = self.pools.get((paired, lang, name, upper))
                        if pool is not None:
                            names.append(name)
                            # Tiny categories do not repeat a handful of rows all day.
                            mass.append(
                                w * min(1, len(pool) / (120 if split == "train" else 8))
                            )
                    if names:
                        self.draws[paired, lang, upper] = (
                            names,
                            np.cumsum(mass) / sum(mass),
                        )

    def sample(self, rng, lang, upper, paired):
        names, cdf = self.draws[paired, lang, upper]
        name = names[np.searchsorted(cdf, rng.random())]
        pool = self.pools[paired, lang, name, upper]
        j = pool[rng.integers(len(pool))]
        records = self.records[lang, name]
        start, marker, end = records.index[j]
        pre = records.data[start : marker + 1]
        good = records.data[marker + 1 : end]
        bad = None
        if paired:
            a, b = records.foil_index[j]
            bad = records.foils[a:b]
        if pre[-1] != 4 or good[-1] != 2:
            raise ValueError(f"Incorrect assistant boundary in {lang}-{name} row {j}")
        if pre[2] == 3 and rng.random() < 0.72:
            default = self.system[lang]
            longest = max(len(good), len(bad) if bad is not None else 0)
            if len(pre) + len(default) + longest - 1 <= upper:
                pre = np.concatenate([pre[:2], default, pre[2:]])
        return pre, good, bad, name

    def batch(self, rng, upper, capacity, paired=False, lang=None):
        per = max(2, capacity // upper // (2 if paired else 1) // 2 * 2)
        records = [
            self.sample(rng, lang or ("en", "tr")[i % 2], upper, paired)
            for i in range(per)
        ]
        groups = [(pre, good) for pre, good, _, _ in records]
        if paired:
            groups += [(pre, bad) for pre, _, bad, _ in records]
        width = min(upper, 8 * ((max(len(p) + len(a) - 1 for p, a in groups) + 7) // 8))
        x = np.zeros((len(groups), width), dtype=np.int64)
        y = np.full_like(x, -100)
        for i, (pre, ans) in enumerate(groups):
            n = len(pre)
            x[i, :n] = pre
            x[i, n : n + len(ans) - 1] = ans[:-1]
            y[i, n - 1 : n + len(ans) - 1] = ans
        return x, y, [r[-1] for r in records]
