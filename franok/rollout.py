"""Share a prompt and cache complete routing groups during on-policy sampling."""

import torch
from torch.nn import functional as F
from .model import routing


class GroupDecoder:
    """Several independent sampled continuations of one prompt at one adapter scale."""

    def __init__(self, model, prompt, count):
        self.model = model
        banks = model.c.banks
        if prompt.ndim != 1 or not 0 < len(prompt) <= model.c.window or count < 1:
            raise ValueError(
                "Expected one valid shared prompt and at least one continuation"
            )
        self.offset = (len(prompt) - 1) // banks * banks
        self.history = []
        if self.offset:
            model.hidden(prompt[: self.offset][None], self.history)
        self.anchor = (
            prompt[self.offset - 1] if self.offset else prompt.new_zeros(())
        ).expand(count)
        self.tail = prompt[self.offset :][None].expand(count, -1)
        self.committed = None

    def logits(self):
        model, count = self.model, self.tail.shape[1]
        banks = model.c.banks
        if self.offset + count > model.c.window:
            raise ValueError("Rollout exceeds the neural window")
        ids = F.pad(self.tail, (0, banks - count))
        x = model.embed(ids)
        upto = self.offset + banks
        cos = model.cos[self.offset : upto].to(x.dtype)
        sin = model.sin[self.offset : upto].to(x.dtype)
        mask = None
        if self.offset:
            keys = torch.arange(upto, device=ids.device)
            mask = (
                keys[None]
                <= self.offset + torch.arange(banks, device=ids.device)[:, None]
            )
        self.committed = [] if count == banks else None
        for i, block in enumerate(model.blocks):
            previous = self.history[i] if self.history else None
            routes = routing(ids, i, banks, self.anchor)
            x = block(x, routes, cos, sin, self.committed, previous, mask)
        return F.linear(model.norm(x[:, count - 1]), model.embed.weight)

    def advance(self, token):
        if self.tail.shape[1] == self.model.c.banks:
            if self.committed is None:
                raise RuntimeError("Process the completed group before advancing")
            self.history, self.committed = self.committed, None
            self.anchor = self.tail[:, -1]
            self.offset += self.model.c.banks
            self.tail = token[:, None]
        else:
            self.tail = torch.cat((self.tail, token[:, None]), 1)
