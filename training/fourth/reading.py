"""Short-span extractive reading examples from SQuAD v1.1 train (EN) and SQuAD-TR train, checked against XQuAD."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from tools.transfer import XQUAD
from training.third.instructions import SQUAD, read_squad
from .filter import fold, words

MAX_WORDS = 8


def normal(text):
    return re.sub(r"\W+", " ", fold(text)).strip()


def shingles(text, n=8):
    w = words(text)
    return {hash(tuple(w[i : i + n])) for i in range(len(w) - n + 1)}


def xquad_rows():
    rows = []
    for lang in ("en", "tr"):
        file = hf_hub_download("google/xquad", f"xquad.{lang}/validation-00000-of-00001.parquet", revision=XQUAD, repo_type="dataset")
        rows += pq.read_table(file).to_pylist()
    return rows


def squad(lang, local):
    """Answerable train questions; the checksum-verified v2.0 file holds exactly the v1.1 train questions plus impossible ones."""
    for article in read_squad(lang, local):
        for passage in article["paragraphs"]:
            for q in passage["qas"]:
                if q.get("is_impossible") or not q["answers"]:
                    continue
                yield dict(lang=lang, id=q["id"], title=article["title"], context=passage["context"], question=q["question"], answers=[a["text"] for a in q["answers"]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--en", type=Path, help="local SQuAD 2.0 train json; downloaded and checksum-verified when omitted")
    p.add_argument("--tr", type=Path, help="local SQuAD-TR train json.gz; downloaded and checksum-verified when omitted")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    benchmark = xquad_rows()
    ids = {r["id"] for r in benchmark}
    contexts = {normal(r["context"]) for r in benchmark}
    held = set().union(*(shingles(r["context"]) for r in benchmark))
    counts, rows, seen = Counter(), [], set()
    for ex in [*squad("en", a.en), *squad("tr", a.tr)]:
        counts["in_" + ex["lang"]] += 1
        if ex["id"] in ids:
            raise SystemExit("XQuAD id found in training reading data: " + ex["id"])
        if normal(ex["context"]) in contexts:
            raise SystemExit("XQuAD context found in training reading data: " + ex["id"])
        if shingles(ex["context"]) & held:
            counts["drop_xquad_8gram"] += 1
            continue
        answer = ex["answers"][0].strip()
        if not answer or len(words(answer)) > MAX_WORDS or len(answer) > 64 or answer not in ex["context"]:
            counts["drop_answer"] += 1
            continue
        if (ex["context"], ex["question"]) in seen:
            counts["drop_duplicate"] += 1
            continue
        seen.add((ex["context"], ex["question"]))
        split = "dev" if int.from_bytes(hashlib.sha256(("squad:" + ex["title"]).encode()).digest()[:4], "little") % 50 == 0 else "train"
        counts[f"{split}_{ex['lang']}"] += 1
        rows.append(dict(lang=ex["lang"], id=ex["id"], split=split, context=ex["context"].strip(), question=ex["question"].strip(), answer=answer))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    a.out.with_suffix(".json").write_text(
        json.dumps(dict(counts=dict(counts), xquad_ids=len(ids), xquad_contexts=len(contexts), id_overlap=0, context_overlap=0, sources=dict(en=SQUAD["en"], tr=SQUAD["tr"]), xquad_revision=XQUAD), indent=2) + "\n"
    )
    print(dict(counts), flush=True)


if __name__ == "__main__":
    main()
