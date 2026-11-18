"""Verify completed and partial reply features and separation from backbone gradients."""

import tempfile
from pathlib import Path
import torch
from franok.model import Config, Franok
from franok.preference import ReplyPreference
from training.third.preference import DialogueReward, labelled_states
from training.third.posttrain import losses
from tests.isolated_batches import ExampleBatch
import numpy as np


def main():
    torch.set_num_threads(2)
    torch.manual_seed(126)
    cfg = Config(
        vocab=256,
        width=32,
        heads=4,
        kv_heads=2,
        layers=2,
        banks=4,
        hidden=64,
        window=32,
    )
    model = Franok(cfg)
    x, y, _ = ExampleBatch().batch(np.random.default_rng(9), 32, 128, paired=True)
    score, per, hidden = losses(model, x, y, True)
    assert not hidden.requires_grad and score.requires_grad
    alias = torch.as_tensor(y)
    with torch.no_grad():
        states = model.hidden(torch.as_tensor(x))
    marks = (alias != -100).int().argmax(1)
    n = (alias != -100).sum(1) - 1
    torch.testing.assert_close(hidden, ReplyPreference.features(states, marks, n))
    torch.testing.assert_close(
        labelled_states(states, alias, 1),
        ReplyPreference.features(states, marks, n.clamp_max(1)),
    )
    with tempfile.TemporaryDirectory() as d:
        reward = DialogueReward(32, torch.device("cpu"), Path(d))
        snapshot = reward.head.score.weight.clone()
        reward.update(hidden, ["human", "short-replies"])
        assert not torch.equal(snapshot, reward.head.score.weight)
        assert all(p.grad is None for p in model.parameters())
        reward.save()
    score.backward()
    assert model.embed.weight.grad is not None
    print(
        "Only the reward head learned from detached reply comparisons. The language-model gradient stayed intact."
    )


if __name__ == "__main__":
    main()
