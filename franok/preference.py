"""A preference head for comparing possible replies to the same dialogue prefix."""

import torch
from torch import nn


class ReplyPreference(nn.Module):
    def __init__(self, width=512, hidden=96):
        super().__init__()
        self.norm = nn.LayerNorm(width * 4)
        self.hidden = nn.Linear(width * 4, hidden)
        self.score = nn.Linear(hidden, 1)

    @staticmethod
    def features(states, mark, counts):
        """Pool actual answer tokens, using the assistant delimiter as the query state."""
        rows = torch.arange(states.shape[0], device=states.device)
        q = states[rows, mark]
        pos = torch.arange(states.shape[1], device=states.device)[None, :]
        mask = (pos > mark[:, None]) & (pos <= mark[:, None] + counts[:, None])
        pooled = (states * mask[:, :, None]).sum(1) / counts[:, None].clamp_min(1)
        last = states[rows, (mark + counts).clamp_max(states.shape[1] - 1)]
        return torch.cat((pooled, last, q * pooled, q * last), 1).float()

    def forward(self, features):
        return self.score(torch.tanh(self.hidden(self.norm(features)))).squeeze(-1)
