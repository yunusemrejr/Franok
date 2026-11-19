"""Conversation training for Franok 0.5: a long all-token stage on a wide dialogue corpus, then an assistant-only polish.

One script runs both stages. Each micro-batch comes from the mixture component furthest behind its share of
target tokens. Candidates are scored on held-out conversations with assistant-only loss, whatever the training mask,
so stages and learning rates can be compared on one scale.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from franok.model import Config, Franok
from training.extended.dialogue import DEFAULT
from training.fourth.finetune import LANG_ID, REPLAY_BANDS, Examples, encode_reading, evaluate, save, sequence_loss, sha256, widen
from training.third.batches import BOUNDARIES, Conversations
from training.third.pretrain import LANGUAGES, Shards

SHARES = dict(clean=0.45, bulk=0.25, reading=0.08, replay=0.0, raw=0.22)


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_conversations(tok, convs, every_token):
    """Token ids and loss mask per conversation. The mask covers replies, or every token after the system block."""
    systems = {lang: tok.encode(text).ids for lang, text in DEFAULT.items()}
    texts = [t[role] for c in convs for t in c["turns"] for role in ("user", "assistant")]
    pieces = iter(e.ids for e in tok.encode_batch(texts))
    out = []
    for c in convs:
        ids = [1, LANG_ID[c["lang"]], *systems[c["lang"]], 2]
        mask = [False] * len(ids)
        for _ in c["turns"]:
            user = [3, *next(pieces), 4]
            answer = [*next(pieces), 2]
            ids += user + answer
            mask += [every_token] * len(user) + [True] * len(answer)
        out.append((ids, mask))
    return out


def parse_shares(text, present):
    shares = dict(SHARES)
    for part in filter(None, (text or "").split(",")):
        key, value = part.split("=")
        if key not in SHARES:
            raise SystemExit(f"Unknown mixture component {key!r}; choose from {sorted(SHARES)}")
        shares[key] = float(value)
    shares = {k: v for k, v in shares.items() if v > 0 and k in present}
    total = sum(shares.values())
    return {k: v / total for k, v in shares.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--expect-sha")
    p.add_argument("--clean", type=Path, required=True, help="graded conversations with train and dev splits")
    p.add_argument("--bulk", type=Path, nargs="*", default=[], help="public dialogue files in the same row format")
    p.add_argument("--reading", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every-token", action="store_true", help="train on user turns as well as replies")
    p.add_argument("--shares", help="for example clean=0.6,raw=0.1; unnamed components keep their defaults")
    p.add_argument("--peak-lr", type=float, required=True)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--floor", type=float, default=0.05, help="final learning rate as a fraction of the peak")
    p.add_argument("--minutes", type=float, required=True)
    p.add_argument("--capacity", type=int, default=32768, help="padded positions per micro-batch")
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--eval-minutes", type=float, default=10.0)
    p.add_argument("--keep", type=int, default=4, help="candidate checkpoints kept on disk besides the best")
    p.add_argument("--smoke", action="store_true", help="three CPU steps on a small slice")
    a = p.parse_args()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not a.smoke else "cpu")
    amp = device.type == "cuda"
    if not a.smoke and device.type != "cuda":
        raise SystemExit("A CUDA GPU is required outside --smoke")
    digest = sha256(a.weights)
    if a.expect_sha and digest != a.expect_sha:
        raise SystemExit(f"Weights digest {digest} does not match the expected {a.expect_sha}")
    torch.manual_seed(50510)
    tok = Tokenizer.from_file(os.environ.get("FRANOK_TOKENIZER", "web/tokenizer.json"))
    config_text = (a.weights.parent / "config.json").read_text()
    cfg = Config(**json.loads(config_text))
    model = Franok(cfg)
    model.load_state_dict(load_file(str(a.weights)))
    model.to(device).train()

    clean = rows(a.clean)
    bulk = [r for path in a.bulk for r in rows(path)]
    reading = rows(a.reading) if a.reading else []
    if a.smoke:
        clean, bulk, reading = clean[:300], bulk[:200], reading[:200]
    held = [c for c in clean if c.get("split") == "dev"]
    if not held:
        raise SystemExit(f"{a.clean} has no dev rows to select a checkpoint with")
    train = dict(clean=Examples(encode_conversations(tok, [c for c in clean if c.get("split") != "dev"], a.every_token), 1))
    if bulk:
        train["bulk"] = Examples(encode_conversations(tok, bulk, a.every_token), 2)
    dev = dict(conversations=Examples(encode_conversations(tok, held, False), 3))
    for lang in LANGUAGES:
        part = [c for c in held if c["lang"] == lang]
        if part:
            dev["conversations_" + lang] = Examples(encode_conversations(tok, part, False), 4)
    if reading:
        train["reading"] = Examples([encode_reading(tok, r) for r in reading if r["split"] == "train"], 5)
        dev["reading"] = Examples([encode_reading(tok, r) for r in reading if r["split"] == "dev"][:600], 6)
    raw, raw_val = Shards("train"), widen(Shards("val"))
    shares = parse_shares(a.shares, {*train, "replay", "raw"})
    replay = Conversations("train") if "replay" in shares else None
    print(
        "prepared in %.0fs: " % (time.time() - started)
        + ", ".join(f"{k} {len(v)} rows/{v.targets} targets" for k, v in train.items())
        + f", dev {len(held)}, shares {shares}",
        flush=True,
    )

    rng = np.random.default_rng(50511)
    capacity = 2048 if a.smoke else a.capacity
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    log_file = out / "train.jsonl"

    def log(**fields):
        record = dict(utc=time.time(), **fields)
        print(json.dumps(record), flush=True)
        with log_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def draw(component):
        if component in train:
            return train[component].next(capacity)
        if component == "replay":
            x, y, _ = replay.batch(rng, int(rng.choice(BOUNDARIES, p=REPLAY_BANDS)), capacity)
            return x, y
        count = max(1, capacity // 512 // 2)
        data = np.concatenate([raw.batch(rng, lang, count, 512) for lang in LANGUAGES]).astype(np.int64)
        return data[:, :-1], data[:, 1:]

    def candidate(step):
        scores = evaluate(model, dev, raw_val, device, amp, capacity)
        path = out / "candidates" / f"step-{step:06d}" / "model.safetensors"
        save(model, path, config_text)
        candidates.append(dict(step=step, path=str(path), minutes=(time.time() - started) / 60, **scores))
        log(stage="candidate", step=step, path=str(path), targets=dict(got), **scores)
        best = min(candidates, key=lambda c: c["conversations"])
        for old in candidates[: -a.keep]:
            if old is not best and Path(old["path"]).exists():
                Path(old["path"]).unlink()

    start = evaluate(model, dev, raw_val, device, amp, capacity)
    log(stage="start", **start)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.peak_lr, betas=(0.9, 0.95), weight_decay=0.01, fused=device.type == "cuda")
    got = {k: 0 for k in shares}
    positions = {k: 0 for k in shares}
    candidates = []
    step, begin = 0, time.time()
    deadline, last_eval = begin + 60 * a.minutes, begin
    warmup = 3 if a.smoke else a.warmup
    while time.time() < deadline and not (a.smoke and step >= 3):
        fraction = (time.time() - begin) / (deadline - begin)
        rate = a.peak_lr * min(1.0, (step + 1) / warmup) * (a.floor + (1 - a.floor) * (1 + math.cos(math.pi * min(1.0, fraction))) / 2)
        for group in opt.param_groups:
            group["lr"] = rate
        micro = []
        for _ in range(a.accumulate):
            total = max(1, sum(got.values()))
            component = max(shares, key=lambda k: shares[k] * total - got[k])
            x, y = draw(component)
            micro.append((component, x, y, int((y != -100).sum())))
            got[component] += micro[-1][3]
            positions[component] += x.size
        targets = sum(m[3] for m in micro)
        opt.zero_grad(set_to_none=True)
        try:
            summed = 0.0
            for _, x, y, _ in micro:
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
        if step == 40 and not a.smoke and not all(got.values()):
            raise SystemExit(f"Mixture is missing a component after 40 steps: {got}")
        if step % 50 == 0 or step == 1:
            log(stage="train", step=step, loss=summed / targets, lr=rate, norm=float(norm), targets=dict(got), positions=dict(positions), capacity=capacity, epochs={k: v.epoch for k, v in train.items()})
        if time.time() - last_eval >= 60 * a.eval_minutes:
            candidate(step)
            last_eval = time.time()
    if not candidates or candidates[-1]["step"] != step:
        candidate(step)
    best = min(candidates, key=lambda c: c["conversations"])
    chosen = out / "best"
    chosen.mkdir(parents=True, exist_ok=True)
    (chosen / "config.json").write_text(config_text)
    target = chosen / "model.safetensors"
    target.unlink(missing_ok=True)
    os.link(best["path"], target)
    summary = dict(
        parent_sha256=digest,
        every_token=a.every_token,
        peak_lr=a.peak_lr,
        shares=shares,
        start=start,
        selected=best,
        selected_sha256=sha256(target),
        candidates=candidates,
        steps=step,
        target_tokens=got,
        positions=positions,
        examples={k: len(v) for k, v in train.items()},
        elapsed_seconds=time.time() - started,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("selected", "selected_sha256", "target_tokens", "steps")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
