"""Franok's causal, balanced anchor-rotation decoder."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab: int = 8192
    width: int = 512
    layers: int = 12
    heads: int = 8
    kv_heads: int = 2
    hidden: int = 2048
    banks: int = 8
    window: int = 256


def routing(tokens, layer, banks):
    """Indices to gather positions in bank order and to undo that gather."""
    groups = tokens.reshape(tokens.shape[0], -1, banks)
    anchors = F.pad(groups[:, :-1, -1], (1, 0))
    rotation = ((anchors >> (layer % 4)) + layer) % banks
    slots = torch.arange(banks, device=tokens.device)
    return (slots - rotation[..., None]) % banks, (slots + rotation[..., None]) % banks


def rope(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        dim = c.width // c.heads
        self.pre = nn.RMSNorm(c.width, eps=1e-5)
        self.qkv = nn.Linear(c.width, c.width + 2 * c.kv_heads * dim, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        self.post = nn.RMSNorm(c.width, eps=1e-5)
        self.up = nn.Parameter(torch.empty(c.banks, 2 * c.hidden, c.width))
        self.down = nn.Parameter(torch.empty(c.banks, c.width, c.hidden))

    def project_qkv(self, x, cos, sin):
        c = self.c
        b, t, d = x.shape
        hd = d // c.heads
        u = self.pre(x)
        projected = self.qkv(u)
        if hasattr(self, "lora_qkv"):
            projected = projected + self.lora_qkv(u)
        q, k, v = projected.split((d, c.kv_heads * hd, c.kv_heads * hd), dim=-1)

        def heads(a):
            return a.reshape(b, t, -1, hd).transpose(1, 2)

        q, k, v = rope(heads(q), cos, sin), rope(heads(k), cos, sin), heads(v)
        return q, k, v

    def attend(self, x, q, k, v, causal):
        repeat = self.c.heads // self.c.kv_heads
        attended = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(repeat, 1),
            v.repeat_interleave(repeat, 1),
            is_causal=causal,
        )
        attention = attended.transpose(1, 2).reshape_as(x)
        projected = self.out(attention)
        if hasattr(self, "lora_out"):
            projected = projected + self.lora_out(attention)
        return x + projected

    def forward(self, x, routes, cos, sin, cache=None):
        c = self.c
        b, t, d = x.shape
        q, k, v = self.project_qkv(x, cos, sin)
        if cache is not None:
            cache.append((k, v))
        x = self.attend(x, q, k, v, causal=True)
        ordered, restore = routes
        # Each bank owns exactly b*t/banks rows, including in the backward pass.
        h = self.post(x).reshape(b, -1, c.banks, d)
        h = (
            h.gather(2, ordered[..., None].expand_as(h))
            .reshape(-1, c.banks, d)
            .transpose(0, 1)
        )
        expanded = torch.bmm(h, self.up.transpose(1, 2).to(h.dtype))
        if hasattr(self, "lora_up"):
            expanded = expanded + self.lora_up(h)
        a, g = expanded.chunk(2, -1)
        activated = F.silu(a) * g
        h = torch.bmm(activated, self.down.transpose(1, 2).to(h.dtype))
        if hasattr(self, "lora_down"):
            h = h + self.lora_down(activated)
        h = h.transpose(0, 1).reshape(b, -1, c.banks, d)
        return x + h.gather(2, restore[..., None].expand_as(h)).reshape(b, t, d)


class Franok(nn.Module):
    def __init__(self, c=Config()):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.width)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.RMSNorm(c.width, eps=1e-5)
        for name, p in self.named_parameters():
            if p.ndim > 1:
                nn.init.normal_(
                    p,
                    std=(
                        0.02 / math.sqrt(2 * c.layers)
                        if name.endswith(("out.weight", "down"))
                        else 0.02
                    ),
                )
        hd = c.width // c.heads
        theta = torch.outer(
            torch.arange(c.window, dtype=torch.float32),
            10000 ** (-torch.arange(0, hd, 2, dtype=torch.float32) / hd),
        )
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def hidden(self, ids, cache=None):
        """Return one normalized causal decoder state per unpadded input token."""
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.c.window:
            raise ValueError("expected a nonempty batch of in-window token sequences")
        length = ids.shape[1]
        ids = F.pad(ids, (0, -length % self.c.banks))
        x = self.embed(ids)
        cos, sin = (
            self.cos[: ids.shape[1]].to(x.dtype),
            self.sin[: ids.shape[1]].to(x.dtype),
        )
        for layer, block in enumerate(self.blocks):
            x = block(x, routing(ids, layer, self.c.banks), cos, sin, cache)
        return self.norm(x[:, :length])

    def forward(self, ids, cache=None):
        return F.linear(self.hidden(ids, cache), self.embed.weight)
