"""Group-relative RL on bilingual extractive questions, short reasoning and answer choices."""

import argparse
import hashlib
import json
import time
from collections import Counter

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from tokenizers import decoders

from franok.lora import attach, merge, save_adapter
from franok.model import Config, Franok
from franok.preference import ReplyPreference
from franok.rollout import GroupDecoder
from .preference import dialogue_source
from training.extended.rl import reward
from training.extended.train import save_net
from .batches import Conversations
from .data import tokenizer
from .posttrain import losses, quality, set_scale, validate
from .pretrain import LANGUAGES, OUT


class Tasks:
    def __init__(self, conversations):
        self.records = conversations.records
        self.pool = {}
        self.dialogue = {l: [] for l in LANGUAGES}
        for (lang, name), source in self.records.items():
            if dialogue_source(name):
                a, m, e = source.index.T
                selected = np.flatnonzero((m - a < 190) & (e - m >= 6) & (e - m <= 64))
                if len(selected):
                    self.dialogue[lang].append((source, selected))
        for (lang, name), source in self.records.items():
            if name not in ("passages", "logic", "commonsense"):
                continue
            idx = source.index
            good = (
                (idx[:, 2] - idx[:, 1] <= 19)
                & (idx[:, 1] - idx[:, 0] < 244)
                & (idx[:, 2] - idx[:, 1] > 2)
            )
            self.pool[lang, name] = np.flatnonzero(good)
            if source.foil_index is not None:
                foil = source.foil_index
                self.pool[lang, name + "-choice"] = np.flatnonzero(
                    good
                    & (foil[:, 1] - foil[:, 0] > 1)
                    & (foil[:, 1] - foil[:, 0] <= 20)
                )

    def draw(self, rng, lang, tok, kind):
        if kind == "dialogue":
            source, available = self.dialogue[lang][
                rng.integers(len(self.dialogue[lang]))
            ]
            a, m, e = source.index[available[rng.integers(len(available))]]
            pre = source.data[a : m + 1].astype(int).tolist()
            return (
                pre,
                tok.decode(source.data[m + 1 : e - 1].astype(int).tolist()).strip(),
                kind,
            )
        name, _, choice = kind.partition("-")
        choices = self.pool[lang, kind]
        j = choices[rng.integers(len(choices))]
        corpus = self.records[lang, name]
        start, mark, end = corpus.index[j]
        pre = corpus.data[start : mark + 1].astype(int).tolist()
        answer = tok.decode(
            corpus.data[mark + 1 : end - 1].astype(int).tolist()
        ).strip()
        if choice:
            a, b = corpus.foil_index[j]
            other = tok.decode(corpus.foils[a : b - 1].astype(int).tolist()).strip()
            flip = bool(rng.integers(2))
            options = [other, answer] if flip else [answer, other]
            request = (
                "Reply with only A or B."
                if lang == "en"
                else "Yanıt olarak yalnızca A veya B yaz."
            )
            extra = f"\nA) {options[0]}\nB) {options[1]}\n{request}"
            pre = pre[:-1] + tok.encode(extra).ids + [4]
            answer = "B" if flip else "A"
        return pre, answer, kind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(43317)
    rng = np.random.default_rng(43317)
    device = "cuda"
    model = Franok(Config(window=1024))
    model.load_state_dict(load_file(str(OUT / "evidence-merged.safetensors")))
    model.cuda().eval()
    attach(model, 16, 32)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=1.0e-4, betas=(0.9, 0.97), weight_decay=0.005, fused=True
    )
    tok = tokenizer()
    tok.decoder = decoders.ByteLevel()
    train = Conversations("train")
    val = Conversations("val")
    tasks, held = Tasks(train), Tasks(val)
    kinds = {
        "en": ("logic", "passages", "passages-choice", "commonsense"),
        "tr": ("logic", "passages", "passages-choice"),
    }
    scorer = None
    folder = OUT / "dialogue-reward"
    if (folder / "validation.json").exists():
        proof = json.loads((folder / "validation.json").read_text())

        def qualified(values):
            whole = values["whole"]
            directional = [values["by_length"][k] for k in ("shorter", "longer")]
            return (
                whole["pairs"] >= 32
                and whole["accuracy"] >= 0.58
                and all(v["pairs"] >= 8 for v in directional)
                and np.mean([v["accuracy"] for v in directional]) >= 0.55
            )

        if all(qualified(proof[l]) for l in LANGUAGES):
            scorer = (
                ReplyPreference(model.c.width).to(device).eval().requires_grad_(False)
            )
            scorer.load_state_dict(load_file(str(folder / "head.safetensors")))
            kinds = {l: (*k, "dialogue") for l, k in kinds.items()}
    fixed = np.random.default_rng(202617)
    dev = [
        (lang, held.draw(fixed, lang, tok, k))
        for lang in LANGUAGES
        for k in kinds[lang]
        for _ in range(4)
    ]

    def log(**fields):
        fields = {"utc": time.time(), **fields}
        print(fields, flush=True)
        with (OUT / "rl.jsonl").open("a") as f:
            f.write(json.dumps(fields) + "\n")

    def mask_roles(values):
        values = values.float()
        values[..., 0:2] = -1e9
        values[..., 3:7] = -1e9
        return values

    @torch.no_grad()
    def sample(task, count, greedy=False):
        pre, answer, kind = task
        ids = torch.tensor([pre], device=device).expand(count, -1).contiguous()
        tokens = []
        oldlogs = []
        live = torch.ones(count, dtype=torch.bool, device=device)
        maximum = (
            36
            if kind == "dialogue"
            else min(26, max(6, len(tok.encode(answer).ids) + 6))
        )
        with torch.autocast(device, dtype=torch.bfloat16):
            decoder = GroupDecoder(model, ids[0], count)
            for _ in range(maximum):
                lp = mask_roles(decoder.logits()).log_softmax(-1)
                ids1 = lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1)[:, 0]
                ids1 = torch.where(live, ids1, 2)
                oldlogs.append(lp.gather(1, ids1[:, None])[:, 0])
                tokens.append(ids1)
                live &= ids1 != 2
                ids = torch.cat((ids, ids1[:, None]), dim=1)
                if not live.any():
                    break
                decoder.advance(ids1)
        generated = torch.stack(tokens, 1)
        mask = F.pad((generated == 2).cumsum(1)[:, :-1], (1, 0)) == 0
        decoded = [
            tok.decode(r[: r.index(2)] if 2 in r else r).strip()
            for r in generated.tolist()
        ]
        values = torch.tensor([reward(s, answer) for s in decoded], device=device)
        if kind == "dialogue":
            # Evaluate every candidate with the same frozen decoder and preference head.
            # Preserve the caller's scale during selection of the final RL adapter.
            previous = model.blocks[0].lora_qkv.scale / 2.0
            try:
                set_scale(model, 0)
                with torch.autocast(device, dtype=torch.bfloat16):
                    states = model.hidden(ids)
                content = (mask & (generated != 2)).sum(1)
                mark = torch.full_like(content, len(pre) - 1)
                pref = scorer(ReplyPreference.features(states, mark, content)).sigmoid()
                values = torch.where(content > 0, 0.8 * pref + 0.2 * values, 0.0)
            finally:
                set_scale(model, previous)
        return ids, generated, torch.stack(oldlogs, 1), mask, values

    @torch.no_grad()
    def evaluate():
        scores = {l: [] for l in LANGUAGES}
        for lang, task in dev:
            valreward = sample(task, 1, True)[-1][0]
            scores[lang].append(float(valreward))
        instruction = validate(model, val)
        return dict(
            mean_reward=float(np.mean([np.mean(v) for v in scores.values()])),
            by_language={
                l: dict(
                    mean=float(np.mean(s)), exact=sum(x == 1 for x in s), count=len(s)
                )
                for l, s in scores.items()
            },
            supervision=instruction,
        )

    def measure(value):
        return value["mean_reward"] - 0.48 * quality(value["supervision"])

    begin = time.time()
    initial = evaluate()
    log(initial=initial, seed=43317, group=6, dialogue_reward_active=scorer is not None)
    totals = Counter()
    source_counts = Counter()
    sums = Counter()
    while time.time() < args.deadline - 70:
        lang = LANGUAGES[totals["groups"] % 2]
        kind = kinds[lang][rng.integers(len(kinds[lang]))]
        task = tasks.draw(rng, lang, tok, kind)
        ids, seq, old, mask, values = sample(task, 6)
        totals["groups"] += 1
        totals["rollouts"] += len(values)
        totals["tokens"] += int(mask.sum())
        source_counts[lang + "-" + kind] += len(values)
        sums["reward"] += float(values.sum())
        std = values.std(unbiased=False)
        if std < 1e-5:
            continue
        advantage = (values - values.mean()) / (std + 1e-4)

        def logp():
            with torch.autocast(device, dtype=torch.bfloat16):
                states = model.hidden(ids[:, :-1])[:, -seq.shape[1] :]
                values = F.linear(states, model.embed.weight)
            lp = mask_roles(values).log_softmax(-1)
            return lp.gather(-1, seq[:, :, None]).squeeze(-1)

        with torch.no_grad():
            set_scale(model, 0)
            reference = logp()
            set_scale(model, 1)
        current = logp()
        ratio = (current - old).clamp(-12, 12).exp()
        a = advantage[:, None]
        gain = torch.minimum(ratio * a, ratio.clamp(0.8, 1.2) * a)
        delta = (reference - current).clamp(-12, 12)
        kl = delta.exp() - delta - 1
        objective = (
            ((-gain + 0.075 * kl) * mask).sum(1) / mask.sum(1).clamp_min(1)
        ).mean()
        if totals["updates"] % 4 == 0:
            x, y, _ = train.batch(rng, 128, 1024)
            objective = objective + 0.06 * losses(model, x, y)[0]
        opt.zero_grad(set_to_none=True)
        objective.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 0.5)):
            raise RuntimeError("Non-finite reinforcement gradient")
        opt.step()
        totals["updates"] += 1
        if totals["updates"] % 40 == 1:
            log(**dict(totals), mean_reward=sums["reward"] / totals["rollouts"])
    if totals["updates"] == 0:
        raise RuntimeError("No on-policy update completed")
    candidates = {"0": initial}
    for factor in (0.35, 1.0):
        set_scale(model, factor)
        candidates[str(factor)] = evaluate()
    chosen = max((0, 0.35, 1.0), key=lambda x: measure(candidates[str(x)]))
    set_scale(model, chosen)
    save_adapter(
        model, OUT / "adapters" / "rl", 16, 32 * chosen, "base+dialogue+evidence"
    )
    updates = {}
    merge(model, updates)
    # Exact optimizer-host products allow a byte-identical FP32 merge on a different CPU.
    save_file(updates, str(OUT / "adapters/rl/updates.safetensors"))
    del updates
    # This is the actual final FP32 merge on the training VM, before rounding.
    full = OUT / "fp32"
    full.mkdir(exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(full / "model.safetensors"),
    )
    save_net(model, OUT / "model.safetensors")

    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    result = dict(
        seed=43317,
        dialogue_reward_active=scorer is not None,
        rollouts=totals["rollouts"],
        optimizer_updates=totals["updates"],
        tokens=totals["tokens"],
        source_counts=source_counts,
        initial=initial,
        candidates=candidates,
        scale=chosen,
        elapsed=time.time() - begin,
        sha256=digest(OUT / "model.safetensors"),
        fp32_sha256=digest(full / "model.safetensors"),
    )
    (OUT / "rl-finished.json").write_text(json.dumps(result, indent=2) + "\n")
    log(finished=result)


if __name__ == "__main__":
    main()
