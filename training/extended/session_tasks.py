"""MIT-licensed, reproducible English/Turkish records with computed answers."""

import argparse
import gzip
import json
import random
from pathlib import Path

NAMES = [
    "Mira",
    "Arda",
    "Lina",
    "Cem",
    "Ada",
    "Ozan",
    "Duru",
    "Ekin",
    "Ayla",
    "Onur",
    "Sena",
    "Can",
    "Dilan",
    "Eren",
    "Bora",
    "Lale",
]
COLORS = [
    ("ochre", "okra"),
    ("mint", "nane"),
    ("coral", "mercan"),
    ("jade", "yeşim"),
    ("indigo", "çivit"),
    ("copper", "bakır"),
    ("ivory", "fildişi"),
    ("amber", "kehribar"),
]
PLACES = [
    ("hall", "salon"),
    ("studio", "atölye"),
    ("library", "kütüphane"),
    ("station", "istasyon"),
    ("office", "ofis"),
    ("garden", "bahçe"),
    ("harbor", "liman"),
    ("museum", "müze"),
]
LABELS = [
    "Atlas",
    "Pera",
    "Merlin",
    "Vega",
    "Orion",
    "Lodos",
    "Sedef",
    "Delta",
    "Nova",
    "Luna",
    "Topaz",
    "Vela",
]


def record(rng, i, lang, held=False):
    tr = lang == "tr"
    a, b, c = rng.sample(NAMES, 3)
    n, m, k = rng.sample(range(2, 80), 3)
    task = i % 8
    pick = lambda x: x[1 if tr else 0]
    if task in (0, 1):
        names = rng.sample(LABELS, rng.randint(5, 11))
        target = rng.choice(names)
        entries = [(t, str(rng.randrange(1000, 9999))) for t in names]
        if task == 1:
            entries.insert(
                rng.randint(0, len(entries)), (target, str(rng.randrange(1000, 9999)))
            )
        # Last assignment wins, even when the insertion precedes the first value.
        answer = dict(entries)[target]
        lines = [f"{t} = {v}" for t, v in entries]
        if i % 3 == 0:
            lines[1:1] = [
                f"{rng.choice(NAMES)} / {rng.randrange(100, 990)}"
                for _ in range(rng.randint(12, 65))
            ]
        if held:
            question = pick(
                (
                    f"Read the log backwards. Which number belongs to {target}?",
                    f"Kaydı sondan başa oku. {target} için hangi sayı geçerli?",
                )
            )
        else:
            question = rng.choice(
                [
                    pick(
                        (
                            f"What is the latest code for {target}? Return only the digits.",
                            f"{target} için son kod nedir? Yalnızca rakamları yaz.",
                        )
                    ),
                    pick(
                        (
                            f"Find the current value of {target}.",
                            f"{target} kaydının güncel değerini bul.",
                        )
                    ),
                ]
            )
        prompt = "\n".join(lines + [question])
    elif task == 2:
        name1, name2 = rng.sample(LABELS, 2)
        prompt = pick(
            (
                f"The manifest records {name1} under {a}. Who signed for {name2}? If the manifest does not say, say so.",
                f"Listede {name1} için {a} yazıyor. {name2} için kim imzaladı? Bilgi yoksa bunu belirt.",
            )
        )
        answer = pick(("The manifest does not say.", "Listede bu bilgi yok."))
    elif task == 3:
        start = rng.randrange(20, 90)
        drop = rng.randrange(1, start)
        prompt = pick(
            (
                f"{a} has {start} tickets, gives away {drop}, then receives {k}. How many tickets remain?",
                f"{a} elindeki {start} biletin {drop} tanesini veriyor, sonra {k} bilet alıyor. Kaç bileti kaldı?",
            )
        )
        answer = pick(
            (
                f"{start} − {drop} + {k} = {start - drop + k} tickets.",
                f"{start} − {drop} + {k} = {start - drop + k} bilet.",
            )
        )
    elif task == 4:
        color = pick(rng.choice(COLORS))
        place = pick(rng.choice(PLACES))
        prompt = pick(
            (
                f"{a} owns the {color} folder. Each folder belonging to {a} goes to the {place}. Where does the {color} folder go?",
                f"{color.capitalize()} dosyanın sahibi {a}. {a} adına gelen dosyaların yeri: {place}. {color.capitalize()} dosyanın yeri neresi?",
            )
        )
        answer = place.capitalize() + "."
    elif task == 5:
        label = rng.choice(LABELS)
        val = rng.randrange(100, 999)
        prompt = pick(
            (
                f'Use JSON with keys "label" and "value". The label is {label} and the value is {val}.',
                f'JSON kullan. Anahtarlar "label" ve "value" olsun. Etiket {label}, değer {val}.',
            )
        )
        answer = json.dumps(dict(label=label, value=val), ensure_ascii=False)
    elif task == 6:
        nums = rng.sample(range(10, 100), rng.randint(3, 6))
        prompt = pick(
            (
                "Sort these integers from smallest to largest: ",
                "Bu tam sayıları küçükten büyüğe sırala: ",
            )
        ) + ", ".join(map(str, nums))
        answer = ", ".join(map(str, sorted(nums)))
    else:
        vals = dict(zip([a, b, c], [n, m, k]))
        winner = max(vals, key=vals.get)
        prompt = pick(
            (
                f"Scores: {a} {n}, {b} {m}, {c} {k}. Who has the highest score?",
                f"Puanlar: {a} {n}, {b} {m}, {c} {k}. En yüksek puan kime ait?",
            )
        )
        answer = winner + "."
    return dict(
        id=f"{lang}-{i}", language=lang, task=task, prompt=prompt, answer=answer
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("work/tasks"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split, size, seed in [("train", 32000, 92711), ("validation", 1200, 45819)]:
        rng = random.Random(seed)
        with (args.out / (split + ".jsonl")).open("w") as f:
            for i in range(size):
                for lang in ["en", "tr"]:
                    f.write(
                        json.dumps(
                            record(rng, i, lang, split == "validation"),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    main()
