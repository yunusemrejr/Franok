"""Measure next-text bits per UTF-8 byte on exactly the same strings with either vocabulary."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer, decoders
from safetensors.torch import load_file
from franok.model import Config, Franok
from tools.reading import PEERS


def sample(data, tokenizer):
    rng = np.random.default_rng(4419073)
    for lang in ("en", "tr"):
        for name, pattern in (
            ("web", r"val-\d+\.bin"),
            ("web-more", r"val-web-more-\d+\.bin"),
            ("stories", r"val-stories-\d+\.bin"),
        ):
            files = sorted(
                f
                for f in (data / lang).glob("*.bin")
                if re.fullmatch(pattern, f.name) and f.stat().st_size > 1100
            )
            if len(files) < 2:
                raise ValueError("Need several held-out shards for " + name)
            # The first shard is used by the training-time checkpoint selector.
            files = files[1:]
            seen = set()
            attempts = 0
            while len(seen) < 32:
                attempts += 1
                if attempts > 10000:
                    raise ValueError("Insufficient intact source passages")
                file = files[rng.integers(len(files))]
                ids = np.memmap(file, dtype="<u2", mode="r")
                at = int(rng.integers(len(ids) - 512))
                values = ids[at : at + 512].astype(int)
                chunks = np.split(values, np.flatnonzero(values < 7))
                segment = max(chunks, key=len)
                segment = segment[segment >= 7]
                text = tokenizer.decode(segment.tolist())
                words = list(re.finditer(r"\S+", text))
                if len(words) < 60:
                    continue
                lo, edge, end = words[1].start(), words[18].end(), words[58].end()
                prefix, suffix = text[lo:edge], text[edge:end]
                digest = hashlib.sha256((prefix + suffix).encode()).hexdigest()
                if (
                    digest in seen
                    or "�" in prefix + suffix
                    or len(suffix.encode()) > 900
                ):
                    continue
                seen.add(digest)
                yield (
                    lang,
                    name,
                    prefix,
                    suffix,
                    dict(file=file.name, offset=at, sha256=digest),
                )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path)
    g.add_argument("--peer", nargs="?", const="smollm2", choices=PEERS)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    franok_tokenizer = Tokenizer.from_file("web/tokenizer.json")
    franok_tokenizer.decoder = decoders.ByteLevel()
    if a.peer:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        repo, revision = PEERS[a.peer]
        tk = AutoTokenizer.from_pretrained(
            repo, revision=revision, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, local_files_only=True, dtype=torch.float32
        ).eval()
        ident = revision

        def encode(s):
            return tk.encode(s, add_special_tokens=False)

        def decode(v):
            return tk.decode(v, clean_up_tokenization_spaces=False)

        def predict(x):
            return model(x, use_cache=False).logits
    else:
        model = Franok(Config(**json.loads((a.run / "config.json").read_text()))).eval()
        file = a.run / "model.safetensors"
        with file.open("rb") as f:
            ident = hashlib.file_digest(f, "sha256").hexdigest()
        model.load_state_dict(load_file(str(file)))

        def encode(s):
            return franok_tokenizer.encode(s, add_special_tokens=False).ids

        def decode(v):
            return franok_tokenizer.decode(v)

        predict = model
    result = {}
    rows = []
    for lang, name, pre, suffix, source in sample(a.data, franok_tokenizer):
        head = encode(pre)
        allids = encode(pre + suffix)
        if not head or allids[: len(head)] != head or decode(allids) != pre + suffix:
            raise RuntimeError(
                "The two text spans must tokenize and decode exactly: "
                + source["sha256"]
            )
        scores = predict(torch.tensor([allids[:-1]]))[0, len(head) - 1 :].float()
        total = F.cross_entropy(
            scores, torch.tensor(allids[len(head) :]), reduction="sum"
        ).item() / math.log(2)
        size = len(suffix.encode())
        rows.append(
            dict(
                lang=lang,
                collection=name,
                **source,
                bytes=size,
                bits=total,
                tokens=len(allids) - len(head),
            )
        )
        for key in (lang, lang + "-" + name):
            sums = result.setdefault(key, dict(windows=0, bytes=0, bits=0))
            sums["windows"] += 1
            sums["bytes"] += size
            sums["bits"] += total
    for sums in result.values():
        sums["bits_per_byte"] = sums["bits"] / sums["bytes"]
    data = dict(weights=ident, seed=4419073, summary=result, items=rows)
    a.out.parent.mkdir(exist_ok=True, parents=True)
    a.out.write_text(json.dumps(data, indent=2) + "\n")
    print({k: round(v["bits_per_byte"], 4) for k, v in result.items()}, flush=True)


if __name__ == "__main__":
    main()
