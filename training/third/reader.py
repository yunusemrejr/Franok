"""Fit a separately switchable extractive reader to frozen Franok decoder states."""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from franok.model import Config, Franok
from franok.span import SpanReadout, labels, pack
from .spans import SpanCorpus


def select(rows, threshold):
    """Measure answer-boundary EM and correct abstention separately in both languages."""
    metrics = {}
    for lang in ("en", "tr"):
        group = [r for r in rows if r[0] == lang]
        answered = [r for r in group if not r[2]]
        impossible = [r for r in group if r[2]]
        correct = sum(margin >= threshold and exact for _, margin, _, exact in answered)
        null = sum(margin < threshold for _, margin, _, _ in impossible)
        metrics[lang] = dict(
            answerable=len(answered),
            exact=correct,
            impossible=len(impossible),
            abstained_correctly=null,
            answered=sum(margin >= threshold for _, margin, _, _ in answered),
        )
    return metrics


def utility(metrics):
    return float(
        np.mean(
            [
                0.7 * v["exact"] / max(1, v["answerable"])
                + 0.3 * v["abstained_correctly"] / max(1, v["impossible"])
                for v in metrics.values()
            ]
        )
    )


def encode(model, batch):
    device = model.embed.weight.device
    ids, legal, meta = [x.to(device) for x in pack(batch)]
    with torch.no_grad(), (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    ):
        states = model.hidden(ids)
    return states.float(), legal, meta


class ReaderTraining:
    def __init__(self, model, seed):
        self.model = model
        self.train = SpanCorpus("train")
        self.rng = np.random.default_rng(seed)
        self.head = SpanReadout(model.c.width).to(model.embed.weight.device)
        self.opt = torch.optim.AdamW(
            self.head.parameters(), lr=0.0011, weight_decay=0.015
        )
        self.steps = self.seen = self.tokens = 0

    def update(self, capacity, rate):
        upper = int(self.rng.choice((128, 256, 512, 1024), p=(0.17, 0.45, 0.32, 0.06)))
        count = max(1, capacity // (2 * upper))
        batch = []
        for lang in ("en", "tr"):
            ids = self.rng.choice(self.train.groups[lang, upper], count, replace=True)
            batch.extend(self.train.at(lang, k) for k in ids)
        h, legal, meta = encode(self.model, batch)
        logits = self.head(h, meta[:, 0], meta[:, 1], legal)
        truth = labels(meta, logits.shape[1], self.head.max_span)
        loss = F.cross_entropy(logits, truth)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.head.parameters(), 1.0, error_if_nonfinite=True
        )
        self.opt.param_groups[0]["lr"] = rate
        self.opt.step()
        self.steps += 1
        self.seen += len(batch)
        self.tokens += int(meta[:, 1].sum())
        return float(loss.detach())

    def snapshot(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "head.next"
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in self.head.state_dict().items()
            },
            str(tmp),
        )
        tmp.replace(directory / "head.safetensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("work/third/run"))
    parser.add_argument("--out", type=Path, default=Path("work/third/run/reader"))
    parser.add_argument("--init", type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.deadline is None and args.steps is None:
        parser.error("Supply a step budget or Unix-time deadline.")
    start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.manual_seed(18435)
    device = torch.device(args.device)
    cfg = Config(**json.loads((args.run / "config.json").read_text()))
    model = Franok(cfg).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(args.run / "model.safetensors")))
    model.to(device)
    state = ReaderTraining(model, 18435)
    head = state.head
    if args.init:
        head.load_state_dict(load_file(str(args.init)))
    val = SpanCorpus("val")

    # The same article-held-out queries and cached decoder states are used at every measurement.
    # This tuning set is separate from the published XQuAD transfer test.
    fixed = np.random.default_rng(9559)
    validation = []
    for lang in ("en", "tr"):
        idx = val.files[lang][2]
        picks = np.r_[
            fixed.choice(np.flatnonzero(idx[:, 3] >= 0), 192, replace=False),
            fixed.choice(np.flatnonzero(idx[:, 3] < 0), 64, replace=False),
        ]
        picks = sorted(picks, key=lambda i: idx[i, 2])
        for j in range(0, len(picks), 12):
            h, legal, meta = encode(model, [val.at(lang, k) for k in picks[j : j + 12]])
            validation.append((lang, h.half().cpu(), legal.cpu(), meta.cpu()))

    @torch.no_grad()
    def evaluate():
        rows = []
        total_loss = 0.0
        for lang, h, legal, meta in validation:
            h, legal, meta = (
                h.to(device=device, dtype=torch.float32),
                legal.to(device),
                meta.to(device),
            )
            logits = head(h, meta[:, 0], meta[:, 1], legal)
            truth = labels(meta, logits.shape[1], head.max_span)
            top, pred = logits[:, :-1].max(1)
            total_loss += float(F.cross_entropy(logits, truth, reduction="sum"))
            for margin, empty, exact in zip(
                (top - logits[:, -1]).tolist(),
                (meta[:, 2] < 0).tolist(),
                (pred == truth).tolist(),
            ):
                rows.append((lang, margin, empty, exact))
        # Tune a single confidence margin for both languages against a fixed, public utility.
        grid = np.arange(-7.0, 7.01, 0.25)
        choices = [(t, select(rows, t)) for t in [-100.0, *grid.tolist()]]
        # A threshold that abstains on almost every question is unhelpful even if it
        # recognizes most impossible examples.
        threshold, selected = max(
            (
                (t, m)
                for t, m in choices
                if all(v["answered"] >= 0.70 * v["answerable"] for v in m.values())
            ),
            key=lambda x: utility(x[1]),
        )
        return dict(
            nll=total_loss / len(rows),
            threshold=float(threshold),
            languages=selected,
        )

    history = []
    best = float("inf")

    def save():
        nonlocal best
        result = evaluate()
        result.update(step=state.steps, examples=state.seen, tokens=state.tokens)
        history.append(result)
        if result["nll"] < best:
            best = result["nll"]
            state.snapshot(args.out)
            (args.out / "config.json").write_text(
                json.dumps(
                    dict(
                        format="franok-span-v1",
                        width=cfg.width,
                        hidden=head.text.out_features,
                        max_span=head.max_span,
                        threshold=result["threshold"],
                        input="bos language user question eos user passage assistant",
                        validation=result,
                    ),
                    indent=2,
                )
                + "\n"
            )
        (args.out / "training.json").write_text(
            json.dumps(
                dict(
                    seed=18435,
                    history=history,
                    sampled={l: state.seen // 2 for l in ("en", "tr")},
                    elapsed_seconds=time.time() - start,
                ),
                indent=2,
            )
            + "\n"
        )
        print(result, flush=True)

    save()
    started = last = time.time()
    while (args.steps is None or state.steps < args.steps) and (
        args.deadline is None or time.time() < args.deadline - 7
    ):
        frac = (
            state.steps / max(1, args.steps)
            if args.steps
            else (time.time() - started) / max(1, args.deadline - started)
        )
        rate = (
            0.00060
            * min(1, (state.steps + 1) / 12)
            * (0.3 + 0.7 * (1 + math.cos(math.pi * min(1, frac))) / 2)
        )
        loss = state.update(args.tokens, rate)
        if state.steps % 100 == 0:
            print(dict(step=state.steps, examples=state.seen, nll=loss), flush=True)
        if time.time() - last >= 75:
            save()
            last = time.time()
    save()
    if state.steps == 0:
        raise RuntimeError("Reader deadline expired before a parameter update.")


if __name__ == "__main__":
    main()
