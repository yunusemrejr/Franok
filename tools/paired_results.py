"""Compare release outcomes on aligned samples, keeping languages of one passage together."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def records(data):
    if "rows" in data:
        return {
            (row["case"].split("-", 1)[1], row["language"], row["turn"]): float(
                row["passed"]
            )
            for row in data["rows"]
        }
    out = {}
    for r in data["items"]:
        key = json.dumps(r["key"], ensure_ascii=False)
        outcome = r["correct"] if "correct" in r else r["exact"]
        out[key, r["lang"], 0] = float(outcome)
    return out


def compare(candidate, reference, seed=8527):
    if not candidate or candidate.keys() != reference.keys():
        raise ValueError("Both evaluations must contain the same complete paired items")
    groups = defaultdict(list)
    better = worse = ties = 0
    for key in candidate:
        diff = candidate[key] - reference[key]
        groups[key[0]].append(diff)
        better += int(diff > 0)
        worse += int(diff < 0)
        ties += int(diff == 0)
    if len({len(group) for group in groups.values()}) != 1:
        raise ValueError(
            "Each passage or scenario must have the same number of outcomes"
        )
    means = np.array([np.mean(group) for group in groups.values()])
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, size=(20000, len(means)), replace=True).mean(1)
    return dict(
        items=len(candidate),
        clusters=len(means),
        candidate_correct=sum(candidate.values()),
        reference_correct=sum(reference.values()),
        gains=better,
        losses=worse,
        ties=ties,
        difference=float(means.mean()),
        percentile_95=np.quantile(draws, [0.025, 0.975]).tolist(),
        seed=seed,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = compare(
        *(records(json.loads(f.read_text())) for f in (args.candidate, args.reference))
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
