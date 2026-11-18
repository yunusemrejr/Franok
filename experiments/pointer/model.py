"""A causal output adapter that mixes Franok's vocabulary and context-copy distributions."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class PointerHead(nn.Module):
    def __init__(self, width=512, rank=32):
        super().__init__()
        self.q = nn.Linear(width, rank, bias=False)
        self.k = nn.Linear(width, rank, bias=False)
        self.prev = nn.Linear(width, rank, bias=False)
        self.gate = nn.Linear(width, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.8)
        self.suffix = nn.Parameter(torch.zeros(3))
        self.user = nn.Parameter(torch.zeros(()))

    def forward(self, hidden, tokens, query_positions):
        """Return source-position probabilities and the logit of the copy gate.

        hidden[b, t] is the frozen decoder's final normalized state after reading
        tokens[b, t]. A prediction at t may copy a token from positions 0 through t.
        """
        batch, length, _ = hidden.shape
        batch_indices = torch.arange(batch, device=tokens.device)[:, None]
        positions = torch.arange(length, device=tokens.device)[None, :]
        keys = self.k(hidden) + F.pad(self.prev(hidden[:, :-1]), (0, 0, 1, 0))
        selected = hidden[batch_indices, query_positions]
        queries = self.q(selected)
        scores = queries @ keys.transpose(1, 2) / math.sqrt(keys.shape[-1])

        boundaries = torch.where((tokens == 3) | (tokens == 4), positions, 0)
        last_role = tokens.gather(1, boundaries.cummax(-1).values)
        scores = scores + self.user * (last_role == 3)[:, None, :]

        # A matching suffix makes its continuation a candidate at the next position.
        target = query_positions[:, :, None]
        source = positions[:, None, :]
        matched = torch.ones_like(scores, dtype=torch.bool)
        for n, coefficient in enumerate(self.suffix, 1):
            last = tokens[batch_indices, (query_positions - n + 1).clamp_min(0)]
            prior = F.pad(tokens, (n, 0))[:, :length]
            matched = (
                matched
                & (last[:, :, None] == prior[:, None, :])
                & (source >= n)
                & (target >= n - 1)
            )
            scores = scores + coefficient * matched

        valid = (source <= target) & (tokens[:, None, :] >= 7)
        weights = scores.masked_fill(~valid, -1e9).softmax(-1) * valid
        gate = self.gate(selected).squeeze(-1).masked_fill(~valid.any(-1), -torch.inf)
        return weights, gate


def target_log_probs(base_log_probs, weights, gate, tokens, targets):
    """Log probability of each known next token under the normalized mixture."""
    copy_mass = (weights * (tokens[:, None, :] == targets[:, :, None])).sum(-1)
    return torch.logaddexp(
        F.logsigmoid(-gate) + base_log_probs,
        F.logsigmoid(gate) + copy_mass.clamp_min(1e-20).log(),
    )
