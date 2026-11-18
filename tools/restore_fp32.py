"""Recover the unrounded final LoRA merge and verify its FP16 projection against the release."""

import argparse, json, shutil, hashlib
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from franok.model import Config, Franok
from franok.lora import load_adapter, merge


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = Config(**json.loads((args.run / "config.json").read_text()))
    net = Franok(cfg).eval()
    net.load_state_dict(load_file(str(args.run / "pre-rl.safetensors")))
    load_adapter(net, args.run / "adapters/rl")
    merge(net)
    actual = load_file(str(args.run / "model.safetensors"))
    mismatches = 0
    worst = 0.0
    for k, v in net.state_dict().items():
        delta = (v.half().float() - actual[k].float()).abs()
        mismatches += int(torch.count_nonzero(delta))
        worst = max(worst, delta.max().item())
    args.out.mkdir(exist_ok=True, parents=True)
    shutil.copy2(args.run / "config.json", args.out / "config.json")
    dest = args.out / "model.safetensors"
    save_file(net.state_dict(), str(dest))
    with dest.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    info = dict(
        method="FP16 pre-reinforcement parent plus FP32 reinforcement adapter update before final rounding",
        sha256=sha,
        fp16_different_elements=mismatches,
        fp16_max_absolute_difference=worst,
    )
    (args.out / "reconstruction.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info)


if __name__ == "__main__":
    main()
