"""Measure activation second moments on complete, balanced training dialogues for Q4."""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from franok.model import Config, Franok, routing
from training.third.batches import Conversations
import json


@torch.inference_mode()
def collect(model, batches):
    totals = {}
    contexts = {}

    def add(name, x, mask):
        # First dimension is the bank (one for a shared projection).
        weight = mask[..., None]
        acc = (x.float().square() * weight).sum(1)
        counts = weight.sum(1)
        if name not in totals:
            totals[name] = (acc.double(), counts.double())
        else:
            totals[name][0].add_(acc)
            totals[name][1].add_(counts)

    def projection(name):
        def observer(module, args):
            x = args[0]
            add(name, x.reshape(1, -1, x.shape[-1]), contexts["valid"].reshape(1, -1))

        return observer

    def banks(layer):
        def observer(module, args, h):
            c = model.c
            ids = contexts["ids"]
            ordered, _ = routing(ids, layer, c.banks)
            b, t, d = h.shape
            slots = h.reshape(b, -1, c.banks, d)
            inputs = (
                slots.gather(2, ordered[..., None].expand_as(slots))
                .reshape(-1, c.banks, d)
                .transpose(0, 1)
            )
            masks = (
                contexts["valid"]
                .reshape(b, -1, c.banks)
                .gather(2, ordered)
                .reshape(-1, c.banks)
                .T
            )
            add(f"blocks.{layer}.up", inputs, masks)
            # A strided subsample limits calibration work on the wide gate.
            block = model.blocks[layer]
            inputs = inputs[:, ::3, :]
            halves = torch.bmm(inputs, block.up.transpose(1, 2)).chunk(2, -1)
            gated = F.silu(halves[0]) * halves[1]
            add(f"blocks.{layer}.down", gated, masks[:, ::3])

        return observer

    hooks = []
    for n, block in enumerate(model.blocks):
        for name in ("qkv", "out"):
            hooks.append(
                getattr(block, name).register_forward_pre_hook(
                    projection(f"blocks.{n}.{name}.weight")
                )
            )
        hooks.append(block.post.register_forward_hook(banks(n)))

    def final(module, args, h):
        _, _, d = h.shape
        add("embed.weight", h.reshape(1, -1, d), contexts["valid"].reshape(1, -1))

    hooks.append(model.norm.register_forward_hook(final))
    try:
        for x, _y, _names in batches:
            ids = torch.from_numpy(x)
            contexts.update(ids=ids, valid=(ids != 0))
            model(ids)
    finally:
        for hook in hooks:
            hook.remove()
    result = {}
    for name, (sums, count) in totals.items():
        if (count < 1).any():
            raise ValueError("Empty calibration bank for " + name)
        var = (sums / count).clamp_min(1e-10)
        # Shrink toward a common importance. The diagonal estimate ignores
        # inter-channel covariance, and the embedding also serves token lookup.
        var = var / var.mean(-1, keepdim=True)
        importance = 0.25 + 0.75 * var
        result[name] = importance.cpu().numpy().astype("float32")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batches", type=int, default=72)
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    cfg = Config(**json.loads((a.run / "config.json").read_text()))
    model = Franok(cfg).eval()
    model.load_state_dict(load_file(str(a.run / "model.safetensors")))
    data = Conversations("train")
    rng = np.random.default_rng(4381)

    def batches():
        for j in range(a.batches):
            yield data.batch(rng, (64, 128, 256, 512)[j % 4], 1024)

    values = collect(model, batches())
    a.out.parent.mkdir(exist_ok=True, parents=True)
    np.savez(a.out, **values)
    print(len(values), "measured matrices", flush=True)


if __name__ == "__main__":
    main()
