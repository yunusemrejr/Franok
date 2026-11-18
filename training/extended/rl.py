"""Group-relative, on-policy reinforcement learning with a deterministic text verifier.

The model samples unrestricted ordinary vocabulary tokens. Rewards can only see
the sampled text and the symbolic answer, never the model's hidden state.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer, decoders
from franok.lora import attach, save_adapter, merge
from franok.model import Config, Franok
from .prepare import PARENT, SOURCES, tokenizer_path
from .dialogue import DEFAULT
from .train import OUT, save_net
from .batches import Corpus


def clean(s):
    return unicodedata.normalize("NFC", s).casefold().strip().strip(" .,;:!?\n")


def reward(text, answer):
    p, g = clean(text), clean(answer)
    if p == g:
        return 1.0
    if not p:
        return 0.0
    a, b = Counter(re.findall(r"\w+", p)), Counter(re.findall(r"\w+", g))
    f1 = 2 * sum((a & b).values()) / max(1, sum(a.values()) + sum(b.values()))
    partial = 0
    if g.isdigit() and p.isdigit() and len(p) == len(g):
        partial = 0.20 * sum(x == y for x, y in zip(p, g)) / len(g)
    return max(0.35 * f1, partial)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deadline", type=float)
    p.add_argument("--minutes", type=float, default=11)
    args = p.parse_args()
    args.deadline = args.deadline or time.time() + 60 * args.minutes
    torch.manual_seed(1719)
    rng = np.random.default_rng(1719)
    torch.set_num_threads(4)
    cfg = Config(**json.loads((OUT / "config.json").read_text()))
    model = Franok(cfg)
    model.load_state_dict(load_file(str(OUT / "model.safetensors")))
    model.cuda().eval()
    # adapters/dialogue and adapters/evidence are merged into this parent.
    attach(model, 16, 32)
    adapters = [
        v
        for _, v in model.named_modules()
        if hasattr(v, "scale") and hasattr(v, "a") and hasattr(v, "b")
    ]
    params = [v for v in model.parameters() if v.requires_grad]
    optim = torch.optim.AdamW(
        params, lr=0.00018, betas=(0.9, 0.97), weight_decay=0.005, fused=True
    )
    repo, rev, _ = SOURCES["logic"]

    def rows(split):
        path = hf_hub_download(
            repo, split + ".jsonl", repo_type="dataset", revision=rev
        )
        return [json.loads(line) for line in Path(path).read_text().splitlines()]

    tr, va = rows("train"), rows("validation")
    tok = Tokenizer.from_file(tokenizer_path())
    tok.decoder = decoders.ByteLevel()
    val = Corpus("val")
    logpath = OUT / "rl.jsonl"

    def log(**v):
        with logpath.open("a") as f:
            f.write(json.dumps(v) + "\n")
        print(v, flush=True)

    def prefix(row):
        lng = row["language"]
        return [
            1,
            5 if lng == "en" else 6,
            *tok.encode(DEFAULT[lng]).ids,
            2,
            3,
            *tok.encode(row["prompt"]).ids,
            4,
        ]

    def policy_logits(ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(ids).float()
        v[..., 0:2] = -1e9
        v[..., 3:7] = -1e9
        return v

    @torch.no_grad()
    def sample(row, size, greedy=False):
        pre = prefix(row)
        inp = torch.tensor([pre], device="cuda").expand(size, -1).contiguous()
        generated = []
        oldlogs = []
        live = torch.ones(size, device="cuda", dtype=torch.bool)
        count = min(28, max(8, len(tok.encode(row["answer"]).ids) + 7))
        for _ in range(count):
            logits = policy_logits(inp)[:, -1]
            lp = logits.log_softmax(-1)
            token = (
                lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1).squeeze(1)
            )
            token = torch.where(live, token, 2)
            generated.append(token)
            oldlogs.append(lp.gather(1, token[:, None])[:, 0])
            live &= token != 2
            inp = torch.cat([inp, token[:, None]], 1)
            if not live.any():
                break
        seq = torch.stack(generated, 1)
        old = torch.stack(oldlogs, 1)
        before = F.pad((seq == 2).cumsum(1)[:, :-1], (1, 0)) == 0
        answers = [
            tok.decode([t for t in r[: r.index(2) if 2 in r else None] if t >= 7])
            for r in seq.tolist()
        ]
        scores = torch.tensor(
            [reward(s, row["answer"]) for s in answers], device="cuda"
        )
        return inp, seq, old, before, scores, answers

    def scale(mult):
        for mod in adapters:
            mod.scale = 2 * mult

    @torch.no_grad()
    def evaluate():
        gen = np.random.default_rng(771)
        items = [va[i] for i in gen.choice(len(va), 24, replace=False)]
        vals = []
        for row in items:
            scores = sample(row, 1, True)[4]
            vals.append(float(scores[0]))
        batchgen = np.random.default_rng(913)
        ce = 0
        for _ in range(4):
            x, y = val.batch(batchgen, 8, 512, "dialogue")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ce += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() / 4
        return dict(
            reward=float(np.mean(vals)),
            exact=sum(v == 1 for v in vals),
            total=len(vals),
            dialogue_nll=ce,
        )

    start = time.time()
    initial = evaluate()
    log(initial=initial, method="group-relative policy gradient", group=6)
    step = 0
    draws = 0
    updates = 0
    rewards = 0.0
    example_tokens = 0
    stop = args.deadline - 100
    while time.time() < stop:
        row = tr[rng.integers(len(tr))]
        batch, seq, old, mask, values, answers = sample(row, 6)
        draws += len(answers)
        rewards += values.sum().item()
        step += 1
        example_tokens += int(mask.sum())
        if values.std(unbiased=False) < 1e-5:
            continue
        advantage = (values - values.mean()) / (values.std(unbiased=False) + 1e-4)

        def logp():
            logits = policy_logits(batch[:, :-1])[:, -seq.shape[1] :]
            return logits.log_softmax(-1).gather(-1, seq[:, :, None]).squeeze(-1)

        with torch.no_grad():
            scale(0)
            reference = logp()
            scale(1)
        current = logp()
        ratio = (current - old).clamp(-12, 12).exp()
        a = advantage[:, None]
        gain = torch.minimum(ratio * a, ratio.clamp(0.8, 1.2) * a)
        diff = (reference - current).clamp(-12, 12)
        kl = diff.exp() - diff - 1
        loss = (((-gain + 0.06 * kl) * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        optim.step()
        updates += 1
        if updates % 25 == 1:
            log(
                updates=updates,
                draws=draws,
                mean_reward=rewards / draws,
                rollout_tokens=example_tokens,
                elapsed=time.time() - start,
            )
    # Select a nonzero scaling on internal, held-out tasks with a conversational-loss penalty.
    candidates = {}
    for factor in [0.35, 1.0]:
        scale(factor)
        candidates[str(factor)] = evaluate()
    factor = max(
        (0.35, 1.0),
        key=lambda f: (
            candidates[str(f)]["reward"]
            - max(0, candidates[str(f)]["dialogue_nll"] - initial["dialogue_nll"])
        ),
    )
    scale(factor)
    save_adapter(
        model, OUT / "adapters" / "rl", 16, 32 * factor, "base+dialogue+evidence"
    )
    merge(model)
    save_net(model, OUT / "rl-model.safetensors")
    with (OUT / "rl-model.safetensors").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    receipt = dict(
        rollouts=draws,
        optimizer_updates=updates,
        tokens=example_tokens,
        reward_sum=rewards,
        initial=initial,
        candidates=candidates,
        scale=factor,
        sha256=digest,
        elapsed=time.time() - start,
    )
    (OUT / "rl-finished.json").write_text(json.dumps(receipt, indent=2))
    log(finished=receipt)


if __name__ == "__main__":
    main()
