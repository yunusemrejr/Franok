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
    p.add_argument(
        "--compare",
        type=Path,
        help="Compare the rebuilt parameter values to a released FP16 checkpoint",
    )
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
    if args.compare:
        actual = load_file(str(args.compare))
        differences, largest, total = 0, 0.0, 0
        for name, tensor in net.state_dict().items():
            difference = (tensor.half().float() - actual[name].float()).abs()
            differences += torch.count_nonzero(difference).item()
            largest = max(largest, difference.max().item())
            total += difference.numel()
        print(
            json.dumps(
                {
                    "parameters": total,
                    "different_after_fp16_rounding": differences,
                    "max_absolute_difference": largest,
                }
            )
        )


if __name__ == "__main__":
    main()
