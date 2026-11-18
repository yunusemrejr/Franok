"""Single-sequence PyTorch inference using the same anchor routing as the WASM decoder."""

import torch
from torch.nn import functional as F


class Decoder:
    def __init__(self, model):
        self.model = model
        self.layers = []
        self.position = 0
        self.last = self.anchor = 0

    @torch.inference_mode()
    def prefill(self, ids):
        """Begin a new sequence. A partial final routing group is valid."""
        self.layers.clear()
        self.position = len(ids)
        if not 0 < self.position <= self.model.c.window:
            raise ValueError("Prompt length outside the model window")
        device = self.model.embed.weight.device
        inputs = torch.as_tensor(ids, device=device, dtype=torch.long).reshape(1, -1)
        logits = self.model(inputs, cache=self.layers)
        self.last = int(ids[-1])
        group = (self.position - 1) // self.model.c.banks
        self.anchor = int(ids[group * self.model.c.banks - 1]) if group else 0
        # The training code pads groups for balanced matrix multiplication. A new
        # token must never attend to the synthetic padding after the actual prompt.
        self.layers = [
            (k[:, :, : self.position], v[:, :, : self.position]) for k, v in self.layers
        ]
        return logits[0, -1]

    @torch.inference_mode()
    def step(self, token):
        c = self.model.c
        if not self.layers or self.position >= c.window or not 0 <= token < c.vocab:
            raise ValueError("Token outside the active sequence")
        if self.position % c.banks == 0:
            self.anchor = self.last
        self.last = int(token)
        x = self.model.embed.weight[token].reshape(1, 1, c.width)
        cos = self.model.cos[self.position : self.position + 1].to(x.dtype)
        sin = self.model.sin[self.position : self.position + 1].to(x.dtype)
        for layer, block in enumerate(self.model.blocks):
            q, k, v = block.project_qkv(x, cos, sin)
            pk, pv = self.layers[layer]
            k, v = torch.cat((pk, k), dim=2), torch.cat((pv, v), dim=2)
            self.layers[layer] = k, v
            x = block.attend(x, q, k, v, causal=False)
            bank = (self.position + (self.anchor >> (layer % 4)) + layer) % c.banks
            h = block.post(x)
            up = F.linear(h, block.up[bank])
            if hasattr(block, "lora_up"):
                up = up + block.lora_up(h)
            a, g = up.chunk(2, dim=-1)
            h = F.silu(a) * g
            down = F.linear(h, block.down[bank])
            if hasattr(block, "lora_down"):
                down = down + block.lora_down(h)
            x = x + down
        self.position += 1
        return F.linear(self.model.norm(x), self.model.embed.weight)[0, 0]
