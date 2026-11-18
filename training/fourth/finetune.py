"""Full-parameter fine-tune of Franok 0.3 on teacher conversations, short-span reading, replay and raw text.

The mixture is controlled by target tokens: each micro-batch comes from the component that is furthest behind its share.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from torch.nn import functional as F

from franok.model import Config, Franok
from tools.transfer import EXTRACT, SYSTEM
from training.extended.dialogue import DEFAULT
from training.third.batches import BOUNDARIES, Conversations
from training.third.data import ROOT
from training.third.pretrain import LANGUAGES, Shards

SHARES = dict(new=0.55, reading=0.20, replay=0.10, raw=0.15)
REPLAY_BANDS = (0.055, 0.19, 0.30, 0.30, 0.155)
LANG_ID = dict(en=5, tr=6)


def encode_conversation(tok, conv, system):
    ids = [1, LANG_ID[conv["lang"]], *tok.encode(system[conv["lang"]]).ids, 2]
    mask = [False] * len(ids)
    for turn in conv["turns"]:
        user = [3, *tok.encode(turn["user"]).ids, 4]
        answer = [*tok.encode(turn["assistant"]).ids, 2]
        ids += user + answer
        mask += [False] * len(user) + [True] * len(answer)
    return ids, mask


def encode_reading(tok, row):
    lang = row["lang"]
    prompt = row["context"] + "\n\n" + row["question"] + "\n" + EXTRACT[lang]
    head = [1, LANG_ID[lang], *tok.encode(SYSTEM[lang]).ids, 2, 3, *tok.encode(prompt).ids, 4]
    answer = [*tok.encode(row["answer"]).ids, 2]
    return head + answer, [False] * len(head) + [True] * len(answer)


class Examples:
    """Variable-length sequences served as length-bucketed batches, one shuffled pass per epoch."""

    def __init__(self, encoded, seed):
        encoded = [(i, m) for i, m in encoded if len(i) <= 1000]
        self.ids = [np.asarray(i, dtype=np.int64) for i, _ in encoded]
        self.mask = [np.asarray(m, dtype=bool) for _, m in encoded]
        self.rng = np.random.default_rng(seed)
        self.targets = int(sum(m[1:].sum() for m in self.mask))
        self.queue, self.order, self.epoch = [], [], 0

    def __len__(self):
        return len(self.ids)

    def pack(self, rows):
        width = 8 * ((max(len(self.ids[r]) for r in rows) - 1 + 7) // 8)
        x = np.zeros((len(rows), width), dtype=np.int64)
        y = np.full_like(x, -100)
        for k, r in enumerate(rows):
            n = len(self.ids[r]) - 1
            x[k, :n] = self.ids[r][:-1]
            y[k, :n] = np.where(self.mask[r][1:], self.ids[r][1:], -100)
        return x, y

    def chunks(self, rows, capacity):
        rows = sorted(rows, key=lambda r: len(self.ids[r]))
        batch = []
        for r in rows:
            if batch and (len(batch) + 1) * (len(self.ids[r]) + 8) > capacity:
                yield batch
                batch = []
            batch.append(r)
        if batch:
            yield batch

    def next(self, capacity):
        if not self.queue:
            if len(self.order) < 256:
                self.order = list(self.rng.permutation(len(self))) + self.order
                self.epoch += 1
            pool, self.order = self.order[-256:], self.order[:-256]
            self.queue = list(self.chunks(pool, capacity))
            self.rng.shuffle(self.queue)
        return self.pack(self.queue.pop())


def sequence_loss(model, x, y, device, amp):
    x = torch.as_tensor(x, device=device)
    labels = torch.as_tensor(y, device=device)
    rows, positions = torch.where(labels != -100)
    with torch.autocast(device.type, enabled=amp, dtype=torch.bfloat16):
        states = model.hidden(x)
        logits = F.linear(states[rows, positions], model.embed.weight)
    return F.cross_entropy(logits.float(), labels[rows, positions], reduction="sum"), len(rows)


@torch.no_grad()
def widen(shards):
    """Validate on every held-out shard (web, web-more, stories), not only the first small one."""
    for lang, maps in shards.maps.items():
        for path in sorted((ROOT / lang).glob("val-*.bin")):
            if path not in maps and path.stat().st_size >= 2050:
                maps[path] = np.memmap(path, dtype="<u2", mode="r")
    return shards


def evaluate(model, dev, raw_val, device, amp, capacity):
    model.eval()
    result = {}
    for name, examples in dev.items():
        total, count = 0.0, 0
        for rows in examples.chunks(range(len(examples)), capacity):
            loss, n = sequence_loss(model, *examples.pack(rows), device, amp)
            total, count = total + loss.item(), count + n
        result[name] = total / max(1, count)
    for lang in LANGUAGES:
        rng = np.random.default_rng(791)
        total = 0.0
        for _ in range(24):
            data = torch.as_tensor(raw_val.batch(rng, lang, 8, 256).astype(np.int64))
            x, y = data[:, :-1], data[:, 1:]
            with torch.autocast(device.type, enabled=amp, dtype=torch.bfloat16):
                logits = model(x.to(device))
            total += F.cross_entropy(logits.float().flatten(0, 1), y.to(device).reshape(-1)).item()
        result["raw_" + lang] = total / 24
    model.train()
    return result


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(model, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}, str(tmp))
    tmp.replace(path)
    (path.parent / "config.json").write_text(config)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--expect-sha", required=True)
    p.add_argument("--conversations", type=Path, required=True)
    p.add_argument("--reading", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--peak-lr", type=float, default=4e-5)
    p.add_argument("--epochs", type=float, default=3.0, help="passes over the new conversations")
    p.add_argument("--minutes", type=float, default=50.0)
    p.add_argument("--capacity", type=int, default=16384, help="padded positions per micro-batch")
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--eval-minutes", type=float, default=5.0)
    p.add_argument("--max-raw-rise", type=float, default=0.03)
    p.add_argument("--smoke", action="store_true", help="three CPU steps on a small slice (pass --accumulate 1 for the fastest)")
    a = p.parse_args()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not a.smoke else "cpu")
    amp = device.type == "cuda"
    if not a.smoke and device.type != "cuda":
        raise SystemExit("A CUDA GPU is required outside --smoke")
    digest = sha256(a.weights)
    if digest != a.expect_sha:
        raise SystemExit(f"Weights digest {digest} does not match the expected {a.expect_sha}")
    torch.manual_seed(40410)
    tok = Tokenizer.from_file(os.environ.get("FRANOK_TOKENIZER", "web/tokenizer.json"))
    config_text = (a.weights.parent / "config.json").read_text()
    cfg = Config(**json.loads(config_text))
    if cfg.window != 1024:
        raise SystemExit("Expected the 1,024-token release configuration")
    model = Franok(cfg)
    model.load_state_dict(load_file(str(a.weights)))
    model.to(device).train()
    conv = [json.loads(l) for l in a.conversations.read_text().splitlines() if l]
    read = [json.loads(l) for l in a.reading.read_text().splitlines() if l]
    if a.smoke:
        conv, read = conv[:200], read[:200]
    new_train = Examples([encode_conversation(tok, c, DEFAULT) for c in conv if c["split"] == "train"], 1)
    # Selection uses the teacher and arithmetic dev rows alone. Exact programmatic rows are low-entropy and get their own diagnostic set.
    new_dev = Examples([encode_conversation(tok, c, DEFAULT) for c in conv if c["split"] == "dev" and c["source"] != "programmatic"], 2)
    exact_dev = [encode_conversation(tok, c, DEFAULT) for c in conv if c["split"] == "dev" and c["source"] == "programmatic"]
    read_train = Examples([encode_reading(tok, r) for r in read if r["split"] == "train"], 3)
    read_dev = Examples([encode_reading(tok, r) for r in read if r["split"] == "dev"][:600], 4)
    replay = Conversations("train")
    raw, raw_val = Shards("train"), widen(Shards("val"))
    dev = dict(conversations=new_dev, reading=read_dev)
    if exact_dev:
        dev["programmatic"] = Examples(exact_dev, 5)
    print(
        f"new {len(new_train)} train/{len(new_dev)} dev ({new_train.targets} targets), "
        f"reading {len(read_train)} train/{len(read_dev)} dev, prepared in {time.time() - started:.0f}s",
        flush=True,
    )
    rng = np.random.default_rng(40411)
    capacity = 2048 if a.smoke else a.capacity
    accumulate = a.accumulate
    budget = a.epochs * new_train.targets
    out = a.out
    log_file = out / "train.jsonl"
    out.mkdir(parents=True, exist_ok=True)

    def log(**fields):
        record = dict(utc=time.time(), **fields)
        print(json.dumps(record), flush=True)
        with log_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def draw(component):
        if component == "new":
            x, y = new_train.next(capacity)
        elif component == "reading":
            x, y = read_train.next(capacity)
        elif component == "replay":
            upper = int(rng.choice(BOUNDARIES, p=REPLAY_BANDS))
            x, y, _ = replay.batch(rng, upper, capacity)
        else:
            rows = max(1, capacity // 512 // 2)
            data = np.concatenate([raw.batch(rng, l, rows, 512) for l in LANGUAGES]).astype(np.int64)
            x, y = data[:, :-1], data[:, 1:]
        return x, y

    start = evaluate(model, dev, raw_val, device, amp, capacity)
    log(stage="start", **start)
    deadline = time.time() + 60 * a.minutes
    limit = {l: start["raw_" + l] + a.max_raw_rise for l in LANGUAGES}
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.peak_lr, betas=(0.9, 0.95), weight_decay=0.01, fused=device.type == "cuda")
    got = {k: 0 for k in SHARES}
    positions = {k: 0 for k in SHARES}
    candidates = []
    step, last_eval, window_start = 0, time.time(), time.time()
    warmup = 3 if a.smoke else 100
    while time.time() < deadline and got["new"] < budget and not (a.smoke and step >= 3):
        fraction = max((time.time() - window_start) / max(1.0, deadline - window_start), got["new"] / budget)
        rate = a.peak_lr * min(1.0, (step + 1) / warmup) * (0.05 + 0.95 * (1 + math.cos(math.pi * min(1.0, fraction))) / 2)
        for group in opt.param_groups:
            group["lr"] = rate
        micro = []
        for _ in range(accumulate):
            total = max(1, sum(got.values()))
            component = max(SHARES, key=lambda k: SHARES[k] * total - got[k] + (1e-9 if k == "new" else 0))
            x, y = draw(component)
            micro.append((component, x, y, int((y != -100).sum())))
            got[component] += micro[-1][3]
            positions[component] += x.size
        targets = sum(m[3] for m in micro)
        opt.zero_grad(set_to_none=True)
        try:
            summed = 0.0
            for component, x, y, _ in micro:
                loss, _ = sequence_loss(model, x, y, device, amp)
                (loss / targets).backward()
                summed += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Non-finite gradient")
            opt.step()
        except torch.OutOfMemoryError:
            if capacity <= 4096:
                raise
            capacity //= 2
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            for component, x, y, n in micro:
                got[component] -= n
                positions[component] -= x.size
            log(stage="oom", capacity=capacity)
            continue
        step += 1
        if step == 20 and not a.smoke and not all(got.values()):
            raise SystemExit(f"Mixture is missing a component after 20 steps: {got}")
        if step % 50 == 0 or step == 1:
            log(stage="train", step=step, loss=summed / targets, lr=rate, norm=float(norm), targets=dict(got), positions=dict(positions), capacity=capacity, epoch_new=new_train.epoch, epoch_reading=read_train.epoch)
        if time.time() - last_eval >= 60 * a.eval_minutes:
            scores = evaluate(model, dev, raw_val, device, amp, capacity)
            path = out / "candidates" / f"step-{step:05d}" / "model.safetensors"
            save(model, path, config_text)
            candidates.append(dict(step=step, path=str(path), **scores))
            log(stage="candidate", step=step, path=str(path), targets=dict(got), **scores)
            last_eval = time.time()
    scores = evaluate(model, dev, raw_val, device, amp, capacity)
    path = out / "candidates" / f"step-{step:05d}" / "model.safetensors"
    if not any(c["step"] == step for c in candidates):
        save(model, path, config_text)
        candidates.append(dict(step=step, path=str(path), **scores))
    log(stage="final", step=step, targets=dict(got), positions=dict(positions), **scores)
    allowed = [c for c in candidates if all(c["raw_" + l] <= limit[l] for l in LANGUAGES)]
    pool = allowed or candidates
    best = min(pool, key=lambda c: c["conversations"]) if allowed else min(pool, key=lambda c: sum(c["raw_" + l] - start["raw_" + l] for l in LANGUAGES))
    chosen = out / "best"
    chosen.mkdir(parents=True, exist_ok=True)
    (chosen / "config.json").write_text(config_text)
    src = Path(best["path"])
    if (chosen / "model.safetensors").exists():
        (chosen / "model.safetensors").unlink()
    os.link(src, chosen / "model.safetensors")
    summary = dict(
        parent_sha256=digest,
        start=start,
        raw_limits=limit,
        satisfied_raw_limit=bool(allowed),
        selected=best,
        selected_sha256=sha256(chosen / "model.safetensors"),
        candidates=candidates,
        steps=step,
        target_tokens=got,
        positions=positions,
        shares={k: got[k] / max(1, sum(got.values())) for k in got},
        epochs_new=got["new"] / new_train.targets,
        examples=dict(new_train=len(new_train), new_dev=len(new_dev), reading_train=len(read_train), reading_dev=len(read_dev)),
        elapsed_seconds=time.time() - started,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("selected", "selected_sha256", "target_tokens", "shares", "steps")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
