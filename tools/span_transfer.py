"""Score the extractive reading mode separately from free next-token generation."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from franok.model import Franok, Config
from franok.span import SpanReadout
from franok.reading import PassageReader
from .transfer import cases, grade, XQUAD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("web/tokenizer.json"))
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config(**json.loads((args.run / "config.json").read_text()))
    config = json.loads((args.head / "config.json").read_text())
    model = Franok(cfg).eval()
    model.load_state_dict(load_file(str(args.run / "model.safetensors")))
    head = SpanReadout(cfg.width, config["hidden"], config["max_span"]).eval()
    head.load_state_dict(load_file(str(args.head / "head.safetensors")))
    reader = PassageReader(
        model, head, Tokenizer.from_file(str(args.tokenizer)), config["threshold"]
    )
    ids = json.loads(Path("tests/transfer_ids.json").read_text())["xquad"]
    results = []
    for item in cases("xquad", ids):
        lang = item["lang"]
        prediction = reader.read(item["question"], item["passage"], lang)
        results.append(
            dict(
                lang=lang,
                key=item["key"],
                **prediction,
                **grade(prediction["answer"], item["answers"], lang)
            )
        )
    summary = {}
    for lang in ("en", "tr"):
        rows = [r for r in results if r["lang"] == lang]
        summary[lang] = dict(
            n=len(rows),
            exact=sum(r["exact"] for r in rows),
            f1=float(np.mean([r["f1"] for r in rows])),
            answered=sum(r["accepted"] for r in rows),
        )

    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    info = dict(
        suite="xquad-extractive-head",
        dataset_revision=XQUAD,
        threshold=reader.threshold,
        base=digest(args.run / "model.safetensors"),
        head=digest(args.head / "head.safetensors"),
        summary=summary,
        items=results,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
