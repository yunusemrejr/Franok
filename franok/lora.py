"""Low-rank updates to attention and the common subspace of the eight banks."""

import json
import torch
from torch import nn
from safetensors.torch import save_file, load_file


class LowRank(nn.Module):
    def __init__(self, indim, outdim, rank, alpha, device):
        super().__init__()
        self.a = nn.Linear(indim, rank, bias=False, device=device)
        self.b = nn.Linear(rank, outdim, bias=False, device=device)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.b(self.a(x)) * self.scale


def attach(model, rank, alpha):
    if any(hasattr(b, "lora_up") for b in model.blocks):
        raise ValueError("Merge the existing adapter before attaching another.")
    model.requires_grad_(False)
    c = model.c
    shapes = {
        "qkv": (c.width, c.width + 2 * c.kv_heads * (c.width // c.heads)),
        "out": (c.width, c.width),
        "up": (c.width, 2 * c.hidden),
        "down": (c.hidden, c.width),
    }
    for block in model.blocks:
        for name, (indim, outdim) in shapes.items():
            setattr(
                block,
                "lora_" + name,
                LowRank(indim, outdim, rank, alpha, block.up.device),
            )


def save_adapter(model, dest, rank, alpha, parent):
    dest.mkdir(exist_ok=True, parents=True)
    weights = {
        k: p.detach().float().cpu().contiguous()
        for k, p in model.named_parameters()
        if ".lora_" in k
    }
    save_file(weights, str(dest / "adapter.safetensors"))
    (dest / "adapter.json").write_text(
        json.dumps(
            dict(
                format="franok-lora-v1",
                rank=rank,
                alpha=alpha,
                parent=parent,
                targets=["qkv", "out", "up", "down"],
                feedforward_update="broadcast across all eight banks",
                parameters=sum(v.numel() for v in weights.values()),
            ),
            indent=2,
        )
    )


def load_adapter(model, dest):
    cfg = json.loads((dest / "adapter.json").read_text())
    attach(model, cfg["rank"], cfg["alpha"])
    state = load_file(str(dest / "adapter.safetensors"))
    params = {k: p for k, p in model.named_parameters() if ".lora_" in k}
    if params.keys() != state.keys():
        raise ValueError("Adapter keys do not match")
    with torch.no_grad():
        for k, p in params.items():
            p.copy_(state[k])


@torch.no_grad()
def merge(model):
    for block in model.blocks:
        for name in ("qkv", "out", "up", "down"):
            lora = getattr(block, "lora_" + name)
            base = getattr(block, name)
            weight = base.weight if isinstance(base, nn.Linear) else base
            # CPU float32 fixes merge order and rounding on every deployment target.
            delta = (
                lora.b.weight.detach().float().cpu()
                @ lora.a.weight.detach().float().cpu()
            )
            weight.add_((delta * lora.scale).to(weight.device))
            delattr(block, "lora_" + name)
    return model
