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
    products = args.run / "adapters/rl/updates.safetensors"
    if products.is_file():
        weights = dict(net.named_parameters())
        updates = load_file(str(products))
        expected = {
            f"blocks.{i}.{key}"
            for i in range(cfg.layers)
            for key in ("qkv.weight", "out.weight", "up", "down")
        }
        if set(updates) != expected:
            raise ValueError("Incomplete reinforcement merge products")
        for name, value in updates.items():
            if value.shape != weights[name].shape[-2:]:
                raise ValueError("Wrong update dimensions for " + name)
            weights[name].add_(value)
    else:
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
    receipt_path = args.run / "rl-finished.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    if mismatches or (receipt.get("fp32_sha256") and receipt["fp32_sha256"] != sha):
        raise ValueError(
            f"Reconstructed weights disagree with training host: {mismatches} FP16 differences, FP32 SHA-256 {sha}"
        )
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
