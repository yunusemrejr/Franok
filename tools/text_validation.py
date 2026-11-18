"""Score fixed English and Turkish windows beyond the early checkpoint-selection shard."""

import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from franok.model import Config, Franok


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("work/third/data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    model = Franok(Config(**json.loads((a.run / "config.json").read_text()))).eval()
    weights = a.run / "model.safetensors"
    model.load_state_dict(load_file(str(weights)))
    results = {}
    for lang in ("en", "tr"):
        for collection, pat in [
            ("web", r"val-\d+\.bin"),
            ("web-more", r"val-web-more-\d+\.bin"),
            ("stories", r"val-stories-\d+\.bin"),
        ]:
            root = a.data / lang
            files = [
                f
                for f in sorted(root.glob("val-*.bin"))
                if re.fullmatch(pat, f.name) and f.name != "val-00000.bin"
            ]
            arrays = [
                np.memmap(f, dtype="<u2", mode="r")
                for f in files
                if f.stat().st_size >= 257 * 2
            ]
            if not arrays:
                raise ValueError("Missing validation shards for " + collection)
            sizes = np.cumsum([len(arr) - 256 for arr in arrays])
            g = np.random.default_rng(19551)
            losses = []
            for _ in range(8):
                samples = []
                for offset in g.integers(sizes[-1], size=8):
                    idx = np.searchsorted(sizes, offset, side="right")
                    start = offset - (sizes[idx - 1] if idx else 0)
                    samples.append(arrays[idx][start : start + 257])
                ids = torch.tensor(np.stack(samples).astype(np.int64))
                loss = F.cross_entropy(
                    model(ids[:, :-1]).flatten(0, 1),
                    ids[:, 1:].reshape(-1),
                    reduction="none",
                )
                losses += loss.reshape(len(ids), -1).mean(1).tolist()
            results[lang + "-" + collection] = dict(
                nll=float(np.mean(losses)),
                windows=len(losses),
                tokens=len(losses) * 256,
            )
    with weights.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    a.out.parent.mkdir(exist_ok=True, parents=True)
    a.out.write_text(
        json.dumps(dict(sha256=sha, seed=19551, groups=results), indent=2) + "\n"
    )
    print(results, flush=True)


if __name__ == "__main__":
    main()
