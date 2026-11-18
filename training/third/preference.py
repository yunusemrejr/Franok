"""Learn a dialogue reward from ranked responses and same-conversation alternatives."""

from collections import Counter
import json
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from franok.preference import ReplyPreference


def dialogue_source(name):
    return name in {
        "human",
        "english-dialogue",
        "short-dialogue",
        "everyday-en",
        "turkish-dialogue",
        "turkish-tasks",
        "native-chat",
        "everyday-tr",
        "short-replies",
    }


def labelled_states(states, labels, early=None):
    # labels begin at the assistant marker and end with the EOS target.
    mask = labels != -100
    marker = mask.int().argmax(1)
    count = mask.sum(1) - 1
    if early is not None:
        count = count.clamp_max(early)
    return ReplyPreference.features(states.detach(), marker, count)


class DialogueReward:
    def __init__(self, width, device, root):
        self.head = ReplyPreference(width).to(device)
        self.opt = torch.optim.AdamW(
            self.head.parameters(), lr=0.00065, weight_decay=0.035
        )
        self.steps = 0
        self.total = Counter()
        self.root = root
        self.root.mkdir(exist_ok=True, parents=True)
        if (root / "head.safetensors").exists():
            self.head.load_state_dict(load_file(str(root / "head.safetensors")))

    def update(self, features, names):
        eligible = [i for i, name in enumerate(names) if dialogue_source(name)]
        if not eligible:
            return
        good, bad = features.chunk(2)
        self.head.train()
        x = self.head(torch.cat((good[eligible], bad[eligible])))
        preferred, rejected = x.chunk(2)
        loss = F.softplus(rejected - preferred).mean() + 0.02 * x.square().mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.head.parameters(), 1.0, error_if_nonfinite=True
        )
        self.opt.step()
        self.steps += 1
        self.total.update(names[i] for i in eligible)

    def save(self):
        file = self.root / "head.next"
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in self.head.state_dict().items()
            },
            str(file),
        )
        file.replace(self.root / "head.safetensors")
        (self.root / "training.json").write_text(
            json.dumps(dict(updates=self.steps, pairs=dict(self.total)), indent=2)
            + "\n"
        )

    @torch.no_grad()
    def evaluate(self, model, conversations):
        self.head.eval()
        device = model.embed.weight.device
        output = {}
        for lang in ("en", "tr"):
            rng = np.random.default_rng(99157)
            results = Counter()
            intervals = Counter()
            for upper in (128, 256, 512):
                for _ in range(10):
                    x, y, names = conversations.batch(
                        rng, upper, 4096, paired=True, lang=lang
                    )
                    idx = [i for i, m in enumerate(names) if dialogue_source(m)]
                    if not idx:
                        continue
                    order = idx + [i + len(names) for i in idx]
                    x = torch.as_tensor(x[order], device=device)
                    y = torch.as_tensor(y[order], device=device)
                    with torch.autocast(
                        device.type, enabled=device.type == "cuda", dtype=torch.bfloat16
                    ):
                        states = model.hidden(x)
                    for name, cut in [("whole", None), ("opening", 20)]:
                        feat = labelled_states(states, y, cut)
                        good, bad = self.head(feat).chunk(2)
                        results[name] += int((good > bad).sum())
                        intervals[name] += len(good)
            output[lang] = {
                k: dict(wins=results[k], pairs=n, accuracy=results[k] / n)
                for k, n in intervals.items()
            }
        (self.root / "validation.json").write_text(json.dumps(output, indent=2) + "\n")
        return output
