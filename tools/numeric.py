"""Write native logits on 2,048 deterministic bilingual validation targets."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from franok.model import Config, Franok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    m = Franok(Config(**json.loads((args.run / "config.json").read_text()))).eval()
    m.load_state_dict(load_file(str(args.run / "model.safetensors")))
    rows = []
    rng = np.random.default_rng(918)
    with (args.out / "reference.f32").open("wb") as f, torch.inference_mode():
        for lang in ["en", "tr"]:
            for group in ["text", "chat"]:
                a = np.memmap(
                    "work/data/val-" + lang + "-" + group + ".bin",
                    mode="r",
                    dtype="<u2",
                )
                for i in range(4):
                    off = rng.integers(len(a) - 129)
                    ids = a[off : off + 129].astype("int64").tolist()
                    logits = m(torch.tensor([ids[:-1]]))[0].numpy().astype("<f4")
                    f.write(logits.tobytes())
                    rows.append(dict(lang=lang, source=group, ids=ids))
    (args.out / "numeric.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
