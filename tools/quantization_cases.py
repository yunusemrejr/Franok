"""Build bilingual validation prefixes and full-precision distributions for Q4/Q8 checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from franok.model import Config, Franok
from training.third.batches import Conversations


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-language", type=int, default=48)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    config = Config(**json.loads((a.run / "config.json").read_text()))
    model = Franok(config).eval()
    model.load_state_dict(load_file(str(a.run / "model.safetensors")))
    data = Conversations("val")
    g = np.random.default_rng(97521)
    cases = []
    seen = set()
    for lang in ("en", "tr"):
        for i in range(a.per_language):
            band = (64, 128, 256, 512)[i % 4]
            for _ in range(2000):
                pre, ans, _, name = data.sample(g, lang, band, False)
                if len(pre) > 210 or len(ans) < 3:
                    continue
                digest = hashlib.sha256(
                    np.asarray(pre, dtype="<u2").tobytes()
                ).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    break
            else:
                raise RuntimeError("Could not draw an independent prefix")
            ans = ans[:16]
            ids = pre.astype(int).tolist() + ans.astype(int).tolist()
            out = model(torch.tensor([ids[:-1]]))[0, len(pre) - 1 :].log_softmax(-1)
            v, k = out.topk(64)
            gold = out[torch.arange(len(ans)), torch.tensor(ans.astype(int))]
            cases.append(
                dict(
                    lang=lang,
                    source=name,
                    group=("dev" if (i // 4) % 2 == 0 else "confirm"),
                    ids=ids,
                    at=len(pre),
                    key=digest,
                    gold=gold.tolist(),
                    top=k.tolist(),
                    probs=v.exp().tolist(),
                )
            )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(cases, separators=(",", ":")))
    print(
        len(cases),
        "prefixes",
        sum(len(r["gold"]) for r in cases),
        "reference positions",
    )


if __name__ == "__main__":
    main()
