"""Continue Franok on equal numbers of English and Turkish natural-text targets."""

import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.nn import functional as F

from franok.model import Config, Franok
from training.extended.train import save_net
from .data import ROOT, PARENT

OUT = Path("work/third/run")
LANGUAGES = ("en", "tr")


class Shards:
    def __init__(self, split):
        self.split = split
        self.maps = {lang: {} for lang in LANGUAGES}
        self.refresh()

    def refresh(self):
        for lang, maps in self.maps.items():
            paths = sorted((ROOT / lang).glob(f"{self.split}-*.bin"))
            # Keep the validation distribution identical for every checkpoint.
            if self.split == "val":
                paths = paths[:1]
            for path in paths:
                if path not in maps and path.stat().st_size >= 2050:
                    maps[path] = np.memmap(path, dtype="<u2", mode="r")

    def batch(self, rng, lang, rows, length):
        arrays = list(self.maps[lang].values())
        ends = np.cumsum([len(arr) - length for arr in arrays])
        offsets = rng.integers(0, ends[-1], size=rows)
        groups = np.searchsorted(ends, offsets, side="right")
        starts = offsets - np.r_[0, ends][groups]
        return np.stack([arrays[g][i : i + length + 1] for g, i in zip(groups, starts)])


def log(**fields):
    record = {"utc": time.time(), **fields}
    print(record, flush=True)
    with (OUT / "pretrain.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--batch-tokens", type=int, default=12288)
    p.add_argument("--peak-lr", type=float, default=2.3e-4)
    args = p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Select an L4 or another BF16 CUDA GPU.")
    OUT.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(3)
    seed = 93183
    torch.manual_seed(seed)
    cfg = Config(window=1024)
    (OUT / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    history = OUT / "recovery.json"
    state = json.loads(history.read_text()) if history.exists() else {}
    parent = PARENT
    if os.getenv("FRANOK_PARENT"):
        with open(os.environ["FRANOK_PARENT"], "rb") as stream:
            parent = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    if state and state.get("parent") != parent:
        raise ValueError("The recovery checkpoint belongs to a different parent")
    checkpoint = (
        OUT / "recovery.safetensors"
        if state
        else (
            os.getenv("FRANOK_PARENT")
            or hf_hub_download(
                "yunusemrejr/Franok", "model.safetensors", revision=PARENT
            )
        )
    )
    model = Franok(cfg)
    model.load_state_dict(load_file(str(checkpoint)))
    model.cuda()
    train, val = Shards("train"), Shards("val")
    while not all(train.maps[l] and val.maps[l] for l in LANGUAGES):
        if time.time() > args.deadline - 60:
            raise RuntimeError("No time left for training after data preparation")
        time.sleep(4)
        train.refresh()
        val.refresh()

    def nll(array):
        tokens = torch.from_numpy(array.astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens[:, :-1])
            return F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].reshape(-1))

    @torch.no_grad()
    def validate():
        model.eval()
        result = {}
        for lang in LANGUAGES:
            fixed = np.random.default_rng(791)
            result[lang] = float(
                np.mean([nll(val.batch(fixed, lang, 8, 256)).item() for _ in range(4)])
            )
        model.train()
        return result

    steps = state.get("steps", 0)
    targets = state.get("targets", 0)
    rng = np.random.default_rng(seed + steps)
    capacity = args.batch_tokens
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.04,
        fused=True,
    )
    scores = validate()
    best = state.get("best", float(np.mean(list(scores.values()))))
    log(
        parent=parent,
        resumed=bool(state),
        steps=steps,
        nll=scores,
        gpu=torch.cuda.get_device_name(),
        seed=seed,
    )
    begin = time.time()
    last_save = begin
    first = steps
    start_targets = targets

    def save():
        nonlocal best
        scores = validate()
        score = float(np.mean(list(scores.values())))
        save_net(model, OUT / "recovery.safetensors")
        if score < best:
            best = score
            # A recovery replacement on the next save preserves this inode.
            bestfile = OUT / "continued.safetensors"
            tmp = OUT / "continued.next"
            tmp.unlink(missing_ok=True)
            os.link(OUT / "recovery.safetensors", tmp)
            tmp.replace(bestfile)
        result = dict(
            steps=steps,
            targets=targets,
            nll=scores,
            best=best,
            parent=parent,
            corpus_token_positions={
                l: sum(len(a) for a in train.maps[l].values()) for l in LANGUAGES
            },
        )
        temp = history.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(history)
        log(**result)

    while time.time() < args.deadline - 15:
        frac = min(1, (time.time() - begin) / max(1, args.deadline - begin))
        length = 256 if frac < 0.18 else (1024 if steps % 6 == 0 else 512)
        rows = max(1, capacity // (2 * length))
        data = np.concatenate([train.batch(rng, l, rows, length) for l in LANGUAGES])
        lr = (
            args.peak_lr
            * min(1, (steps - first + 1) / 120)
            * (0.18 + 0.82 * (1 + math.cos(math.pi * frac)) / 2)
        )
        opt.param_groups[0]["lr"] = lr
        opt.zero_grad(set_to_none=True)
        try:
            loss = nll(data)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Non-finite gradient")
            opt.step()
        except torch.OutOfMemoryError:
            if capacity <= 4096:
                raise
            capacity //= 2
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            log(reduced_batch_tokens=capacity)
            continue
        steps += 1
        targets += rows * 2 * length
        if steps == first + 1 or steps % 100 == 0:
            train.refresh()
            log(
                steps=steps,
                targets=targets,
                loss=loss.item(),
                lr=lr,
                tokens_per_second=round(
                    (targets - start_targets) / max(1, time.time() - begin)
                ),
            )
        if time.time() - last_save > 480:
            save()
            last_save = time.time()
    save()
    if not (OUT / "continued.safetensors").exists():
        raise RuntimeError("Held-out loss did not improve on the parent")
    (OUT / "pretrain.done").touch()


if __name__ == "__main__":
    main()
