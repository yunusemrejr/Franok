"""Sample dev conversations from a training file into judge cases (tests/*.json format)."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

RUBRIC = "Reply sensibly and helpfully to this message, in the same spirit as this reference answer: {}"


def sample(rows, count, seed):
    rng = random.Random(seed)
    dev = [r for r in rows if r.get("split") == "dev"]
    langs = sorted({r["lang"] for r in dev})
    if not langs:
        raise SystemExit("no rows with split 'dev'")
    chosen = []
    for lang in langs:
        groups = defaultdict(list)
        for r in dev:
            if r["lang"] == lang:
                groups[r["source"], r.get("family")].append(r)
        for g in groups.values():
            rng.shuffle(g)
        quota = count // len(langs) + (langs.index(lang) < count % len(langs))
        keys = sorted(groups, key=str)
        picked = []
        while len(picked) < quota and any(groups.values()):
            rng.shuffle(keys)
            for k in keys:
                if groups[k] and len(picked) < quota:
                    picked.append(groups[k].pop())
        if len(picked) < quota:
            raise SystemExit(f"only {len(picked)} dev rows for {lang}, wanted {quota}")
        chosen += picked
    rng.shuffle(chosen)
    return chosen


def to_case(row, n):
    turns = row["turns"][: 2 if n % 2 == 0 else 1]
    return dict(
        case=row["id"],
        lang=row["lang"],
        turns=[t["user"] for t in turns],
        expect=[None] * len(turns),
        rubric=[RUBRIC.format(t["assistant"]) for t in turns],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conversations", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.conversations.read_text().splitlines() if line]
    cases = [to_case(r, i) for i, r in enumerate(sample(rows, args.count, args.seed))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(cases)} cases -> {args.out}")


if __name__ == "__main__":
    main()
