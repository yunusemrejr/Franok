"""Read an exact answer span from Franok's hidden states, with an explicit no-answer score."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SpanReadout(nn.Module):
    def __init__(self, width=512, hidden=96, max_span=48):
        super().__init__()
        self.max_span = max_span
        self.text = nn.Linear(width, hidden)
        self.question = nn.Linear(width, hidden, bias=False)
        self.product = nn.Linear(width, hidden, bias=False)
        self.start = nn.Linear(hidden, 1)
        self.end = nn.Linear(hidden, 1)
        self.length = nn.Linear(hidden, max_span)
        self.null = nn.Linear(hidden, 1)
        nn.init.zeros_(self.length.weight)
        with torch.no_grad():
            self.length.bias.copy_(-0.025 * torch.arange(max_span))

        self.query_token = nn.Linear(width, hidden, bias=False)
        self.passage_token = nn.Linear(width, hidden, bias=False)
        self.fuse = nn.Linear(2 * hidden, hidden, bias=False)
        self.near = nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
        self.wide = nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4, groups=hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.absence = nn.Linear(hidden, 1, bias=False)

    def forward(self, states, offsets, lengths, legal):
        """Scores are indexed by end token and zero-based span length."""
        b, t, _ = states.shape
        row = torch.arange(b, device=states.device)
        q = states[row, offsets - 1]
        pos = torch.arange(t, device=states.device)
        # Compare each passage token to all actual question tokens (excluding the role delimiters).
        q_end = int(offsets.max()) - 2
        q_states = torch.tanh(self.query_token(states[:, 3:q_end]))
        p_states = torch.tanh(self.passage_token(states))
        attention = (p_states @ q_states.transpose(1, 2)) / (p_states.shape[-1] ** 0.5)
        q_mask = pos[None, 3:q_end] < offsets[:, None] - 2
        alignment = attention.masked_fill(~q_mask[:, None], -1e4).softmax(-1) @ q_states
        features = torch.tanh(
            self.text(states)
            + self.question(q)[:, None]
            + self.product(states * q[:, None])
            + self.fuse(torch.cat([alignment, alignment * p_states], -1))
        )
        features = features.masked_fill(
            pos[None, :, None] >= lengths[:, None, None], 0.0
        )
        local = features.transpose(1, 2)
        features = features + torch.tanh(
            self.mix((self.near(local) + self.wide(local)).transpose(1, 2))
        )
        starts = self.start(features).squeeze(-1)
        ends = self.end(features).squeeze(-1)
        index = torch.arange(t, device=states.device)[:, None] - torch.arange(
            self.max_span, device=states.device
        )
        start_ix = index.clamp_min(0)
        logits = ends[:, :, None] + starts[:, start_ix] + self.length(features)
        positions = torch.arange(t, device=states.device)
        passage = (index[None] >= offsets[:, None, None]) & (
            positions[None, :, None] < lengths[:, None, None] - 1
        )
        mask = passage & legal[:, start_ix] & F.pad(legal[:, 1:], (0, 1))[:, :, None]
        masked = logits.masked_fill(~mask, -1e9)
        passage_tokens = (pos[None] >= offsets[:, None]) & (
            pos[None] < lengths[:, None] - 1
        )
        summary = (features * passage_tokens[:, :, None]).sum(1) / passage_tokens.sum(
            1, keepdim=True
        ).clamp_min(1)
        none = self.null(features[row, lengths - 1]) + self.absence(summary)
        return torch.cat([masked.flatten(1), none], 1)


def pack(rows):
    width = 8 * ((max(len(r[0]) for r in rows) + 7) // 8)
    ids = np.zeros((len(rows), width), dtype=np.int64)
    legal = np.zeros_like(ids, dtype=bool)
    metadata = []
    for j, (tokens, pre, flags, first, last) in enumerate(rows):
        n = len(tokens)
        ids[j, :n] = tokens
        legal[j, pre:n] = flags
        metadata.append((pre, n, first, last))
    return torch.from_numpy(ids), torch.from_numpy(legal), torch.tensor(metadata)


def labels(meta, total, width):
    pre, n, first, last = meta.unbind(-1)
    return torch.where(first < 0, total - 1, (pre + last) * width + (last - first))


def pieces(tok):
    alphabet = [i for i in range(256) if 33 <= i <= 126 or 161 <= i <= 172 or i >= 174]
    more = [i for i in range(256) if i not in alphabet]
    revers = dict(
        zip(map(chr, alphabet + list(range(256, 256 + len(more)))), alphabet + more)
    )
    return [
        bytes(revers[c] for c in tok.id_to_token(i))
        for i in range(tok.get_vocab_size())
    ]


def boundaries(text, ids, table):
    raw = text.encode("utf-8")
    ends = np.cumsum([len(table[i]) for i in ids])
    if not len(ends) or ends[-1] != len(raw):
        raise ValueError("Tokenizer did not preserve bytes")
    # A legal extract may start or stop only on a UTF-8 character boundary.
    offsets = np.r_[0, ends]
    legal = np.array(
        [i == len(raw) or raw[i] & 192 != 128 for i in offsets], dtype=np.uint8
    )
    return raw, ends, legal
