"""Rebuild the final checkpoint from the released pre-adapter base and adapters."""

import argparse, json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .model import Franok, Config
from .lora import load_adapter, merge


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "directory",
        type=Path,
        help="HF checkout containing adapters/base and the three named adapter directories",
    )
    p.add_argument("output", type=Path)
    args = p.parse_args()
    folder = args.directory / "adapters"
    base = folder / "base"
    cfg = Config(**json.loads((base / "config.json").read_text()))
    net = Franok(cfg).eval()
    net.load_state_dict(load_file(str(base / "model.safetensors")))
    for name in ("dialogue", "evidence", "rl"):
        load_adapter(net, folder / name)
        merge(net)
        with torch.no_grad():
            for param in net.parameters():
                param.copy_(param.half())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.half().contiguous() for k, v in net.state_dict().items()},
        str(args.output),
    )


if __name__ == "__main__":
    main()
