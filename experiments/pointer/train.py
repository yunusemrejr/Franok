"""Fit the experimental pointer on assistant targets from Franok's open corpus."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from franok.model import Config, Franok
from .model import PointerHead, target_log_probs


def select_records(data_dir):
    records = []
    for split, limit in [("train", 185), ("val", 15)]:
        for lang in ("en", "tr"):
            sources = [
                f"everyday-{lang}",
                f"reading-{lang}",
                f"facts-{lang}",
                "logic",
                "practice",
                "english-dialogue" if lang == "en" else "turkish-dialogue",
            ]
            for source in sources:
                stem = data_dir / f"{split}-{lang}-{source}"
                index = np.load(stem.with_suffix(".idx.npy"))
                data = np.memmap(stem.with_suffix(".bin"), mode="r", dtype="<u2")
                seed = int(hashlib.sha256(stem.name.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                seen = set()
                for idx in rng.permutation(len(index)):
                    start, assistant, end = map(int, index[idx])
                    end = min(end, assistant + 49)
                    boundary = assistant - start
                    if boundary < 3 or end - start > 320 or end - assistant < 3:
                        continue
                    ids = data[start:end].astype("int64")
                    if ids[boundary] != 4 or 3 not in ids[:boundary]:
                        continue
                    prefix = tuple(ids[: boundary + 1])
                    if prefix in seen:
                        continue
                    seen.add(prefix)
                    records.append(
                        dict(
                            ids=torch.from_numpy(ids),
                            start=boundary,
                            split=split,
                            lang=lang,
                            source=source,
                        )
                    )
                    if len(seen) >= limit:
                        break
    return records


@torch.inference_mode()
def encode(records, checkpoint):
    net = Franok(Config(**json.loads((checkpoint / "config.json").read_text()))).eval()
    net.load_state_dict(load_file(str(checkpoint / "model.safetensors")))
    hidden = {}
    hook = net.norm.register_forward_hook(
        lambda module, args, out: hidden.update(states=out)
    )
    try:
        ordered = sorted(records, key=lambda row: len(row["ids"]))
        for start in range(0, len(ordered), 12):
            batch = ordered[start : start + 12]
            tokens = nn.utils.rnn.pad_sequence(
                [row["ids"][:-1] for row in batch], batch_first=True
            )
            logits = net(tokens)
            for index, row in enumerate(batch):
                first, end = row["start"], len(row["ids"]) - 1
                row["h"] = hidden["states"][index, :end].half().clone()
                targets = row["ids"][first + 1 :, None]
                dist = logits[index, first:end].log_softmax(-1)
                row["base"] = dist.gather(-1, targets).squeeze(-1).clone()
    finally:
        hook.remove()


def batch_scores(model, rows):
    def pad(values):
        return nn.utils.rnn.pad_sequence(values, batch_first=True)

    tokens = pad([r["ids"][:-1] for r in rows])
    hidden = pad([r["h"] for r in rows]).float()
    lengths = torch.tensor([len(r["base"]) for r in rows])
    offsets = torch.arange(int(lengths.max()))[None, :]
    positions = offsets + torch.tensor([row["start"] for row in rows])[:, None]
    weights, gate = model(hidden, tokens, positions.clamp_max(tokens.shape[1] - 1))
    targets = pad([r["ids"][r["start"] + 1 :] for r in rows])
    base = pad([r["base"] for r in rows])
    predictions = target_log_probs(base, weights, gate, tokens, targets)
    return predictions, base, gate.sigmoid(), offsets < lengths[:, None]


@torch.no_grad()
def validate(model, rows):
    groups = defaultdict(lambda: np.zeros(4))
    for i in range(0, len(rows), 16):
        batch = rows[i : i + 16]
        pred, base, gate, mask = batch_scores(model, batch)
        for j, row in enumerate(batch):
            valid = mask[j]
            totals = np.array(
                [
                    -pred[j, valid].sum().item(),
                    -base[j, valid].sum().item(),
                    gate[j, valid].sum().item(),
                    valid.sum().item(),
                ]
            )
            for key in ("all", row["lang"], row["source"]):
                groups[key] += totals
    return {
        key: dict(
            pointer=round(p / n, 5),
            base=round(b / n, 5),
            gate=round(g / n, 5),
            n=int(n),
        )
        for key, (p, b, g, n) in groups.items()
    }


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint", type=Path, default=Path("work/v2/run"))
    args.add_argument("--data", type=Path, default=Path("work/extended/data"))
    args.add_argument("--out", type=Path, default=Path("work/pointer-reproduction"))
    cfg = args.parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(11941)
    rows = select_records(cfg.data)
    encode(rows, cfg.checkpoint)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    rng = random.Random(3509)
    head = PointerHead()
    opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.03)
    best = float("inf")
    history = []
    for epoch in range(1, 9):
        rng.shuffle(train)
        for i in range(0, len(train), 16):
            pred, _, _, mask = batch_scores(head, train[i : i + 16])
            opt.zero_grad()
            (-pred[mask].mean()).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        stats = validate(head, val)
        history.append(dict(epoch=epoch, groups=stats))
        print(epoch, stats["all"], flush=True)
        if stats["all"]["pointer"] < best:
            best = stats["all"]["pointer"]
            save_file(head.state_dict(), str(cfg.out / "head.safetensors"))
        (cfg.out / "training.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
