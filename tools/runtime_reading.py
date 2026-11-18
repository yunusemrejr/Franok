"""Prepare and score the same bilingual reading items for the browser and native models."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from tools.transfer import cases, grade, SYSTEM, XQUAD
from tools.reading import REV


def prepare(output):
    ids = json.loads(Path("tests/transfer_ids.json").read_text())
    suites = {suite: list(cases(suite, keys)) for suite, keys in ids.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(system=SYSTEM, suites=suites), ensure_ascii=False)
    )


def summarize(inputs, result, output):
    suites = json.loads(inputs.read_text())["suites"]
    lookup = {
        (suite, row["lang"], json.dumps(row["key"])): row
        for suite, items in suites.items()
        for row in items
    }
    summaries = defaultdict(list)
    rows = []
    for line in result.read_text().splitlines():
        row = json.loads(line)
        suite = row["suite"]
        ref = lookup[
            "belebele" if suite == "belebele" else "xquad",
            row["lang"],
            json.dumps(row["key"]),
        ]
        metrics = (
            {"correct": int(ref["gold"] == row["prediction"])}
            if suite == "belebele"
            else grade(row["answer"], ref["answers"], row["lang"])
        )
        row.update(metrics)
        summaries[suite, row["lang"]].append(metrics)
        rows.append(row)
    seen = {(r["suite"], r["lang"], json.dumps(r["key"])) for r in rows}
    if len(seen) != len(rows) or not rows:
        raise ValueError("Empty result or duplicate reading item")
    summary = defaultdict(dict)
    for (suite, lang), values in summaries.items():
        expected = sum(
            1
            for item in suites["belebele" if suite == "belebele" else "xquad"]
            if item["lang"] == lang
        )
        if len(values) != expected:
            raise ValueError("Incomplete " + suite + "/" + lang)
        summary[suite][lang] = {
            "count": len(values),
            **{k: sum(r[k] for r in values) / len(values) for k in values[0]},
        }
    data = dict(
        dataset_revisions={"belebele": REV, "xquad": XQUAD}, summary=summary, items=rows
    )
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("--results", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if bool(args.results) != bool(args.out):
        p.error("Supply --results and --out together.")
    if args.results:
        summarize(args.inputs, args.results, args.out)
    else:
        prepare(args.inputs)
