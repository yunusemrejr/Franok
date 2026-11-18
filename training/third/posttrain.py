"""Isolated bilingual instruction tuning, dialogue LoRA and paired evidence LoRA."""

import argparse
import gc
import json
import math
import time
from collections import Counter

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from franok.model import Config, Franok
from franok.lora import attach, merge, save_adapter
from training.extended.train import save_net
from .batches import BOUNDARIES, Conversations
from .controls import ControlledChoices
from .pretrain import LANGUAGES, OUT, Shards


def log(**fields):
    fields = {"utc": time.time(), **fields}
    with (OUT / "post.jsonl").open("a") as f:
        f.write(json.dumps(fields) + "\n")
    print(fields, flush=True)


def losses(model, x, y):
    device = model.embed.weight.device
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device)
    with torch.autocast(
        device.type, enabled=device.type == "cuda", dtype=torch.bfloat16
    ):
        flat = F.cross_entropy(
            model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="none"
        )
        token = flat.view_as(y)
        counts = (y != -100).sum(1)
        per = token.sum(1) / counts.clamp_min(1)
        # A terse correct answer gets its own training weight alongside a long reply.
        positive = 0.5 * (per.mean() + token.sum() / counts.sum().clamp_min(1))
    return positive, per


@torch.no_grad()
def validate(model, val):
    model.eval()
    scores = {}
    for lang in LANGUAGES:
        g = np.random.default_rng(22741)
        ce = []
        wins = []
        for upper in (64, 128, 256, 512, 1024):
            for _ in range(2):
                x, y, _ = val.batch(g, upper, 3072, lang=lang)
                ce.append(losses(model, x, y)[0].item())
        for upper in (256, 512):
            for _ in range(4):
                x, y, _ = val.batch(g, upper, 4096, paired=True, lang=lang)
                good, bad = losses(model, x, y)[1].chunk(2)
                wins.extend((good < bad).tolist())
        scores[lang] = {
            "instruction_loss": float(np.mean(ce)),
            "preferred": float(np.mean(wins)),
            "pairs": len(wins),
        }
    model.train()
    return scores


def quality(scores):
    return float(
        np.mean(
            [s["instruction_loss"] - 0.20 * s["preferred"] for s in scores.values()]
        )
    )


def set_scale(model, factor):
    for m in model.modules():
        if hasattr(m, "scale") and hasattr(m, "a") and hasattr(m, "b"):
            m.scale = 2.0 * factor


def main():
    p = argparse.ArgumentParser()
    for name in ("sft", "dialogue", "evidence"):
        p.add_argument("--" + name + "-end", type=float, required=True)
    p.add_argument("--batch-tokens", type=int, default=12288)
    p.add_argument(
        "--skip-reader", action="store_true", help="Run only language-model stages."
    )
    args = p.parse_args()
    if not args.sft_end < args.dialogue_end < args.evidence_end:
        p.error("The phase deadlines must increase")
    torch.set_num_threads(3)
    torch.manual_seed(73719)
    rng = np.random.default_rng(73719)
    cfg = Config(window=1024)
    model = Franok(cfg)
    model.load_state_dict(load_file(str(OUT / "continued.safetensors")))
    model.cuda()
    train, val = Conversations("train"), Conversations("val")
    formats = ControlledChoices(train)
    lm = Shards("train")
    reader = None
    if not args.skip_reader and (OUT.parent / "spans/receipt.json").is_file():
        from .reader import ReaderTraining

        # The separate answer-span head consumes detached decoder states. Its optimizer
        # never changes the language model, either LoRA, or the completion objective.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(6221)
            reader = ReaderTraining(model, 6221)
        warm = OUT / "reader-warm/head.safetensors"
        if warm.exists():
            reader.head.load_state_dict(load_file(str(warm)))
    report = {}
    schedule = [
        ("sft", args.sft_end, 1.6e-4, None, (0.055, 0.19, 0.30, 0.30, 0.155)),
        ("dialogue", args.dialogue_end, 4.2e-4, 32, (0.085, 0.23, 0.33, 0.245, 0.11)),
        ("evidence", args.evidence_end, 2.1e-4, 16, (0, 0.09, 0.33, 0.41, 0.17)),
    ]
    capacity = args.batch_tokens
    for stage, deadline, peak, rank, bands in schedule:
        path = (
            OUT / "base" / "model.safetensors"
            if rank is None
            else OUT / (stage + "-merged.safetensors")
        )
        progress = OUT / (stage + ".json")
        if progress.exists() and path.exists():
            model.load_state_dict(load_file(str(path)))
            report[stage] = json.loads(progress.read_text())
            continue
        if time.time() > deadline - 25:
            raise RuntimeError(f"Insufficient time for {stage}")
        if rank:
            attach(model, rank, 2 * rank)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=peak,
            betas=(0.9, 0.95),
            weight_decay=0.025 if not rank else 0.01,
            fused=True,
        )
        starting = validate(model, val)
        log(stage=stage, initial=starting)
        best_score = quality(starting)
        best_values = starting
        step = targets = positions = 0
        draw_counts = Counter()
        started = last = time.time()
        while time.time() < deadline - 24:
            frac = min(1, (time.time() - started) / max(1, deadline - started))
            rate = (
                peak
                * min(1, (step + 1) / max(18, 60 if rank is None else 28))
                * (0.18 + 0.82 * (1 + math.cos(math.pi * frac)) / 2)
            )
            opt.param_groups[0]["lr"] = rate
            paired = (
                (stage == "evidence" and step % 4 != 0)
                or (stage == "sft" and step % 8 == 3)
                or (stage == "dialogue" and step % 5 == 2)
            )
            upper = int(rng.choice(BOUNDARIES, p=bands))
            if step % (16 if stage != "evidence" else 12) == 7:
                paired = True
                x, y, names = formats.batch(rng, 512, capacity, paired=True)
            elif stage == "sft" and step % 12 == 11 and not paired:
                length = 512
                rows = max(1, capacity // (2 * length))
                data = np.concatenate(
                    [lm.batch(rng, l, rows, length) for l in LANGUAGES]
                ).astype(np.int64)
                x, y = data[:, :-1], data[:, 1:]
                names = ["raw-text"] * (rows * 2)
            else:
                x, y, names = train.batch(rng, upper, capacity, paired)
            opt.zero_grad(set_to_none=True)
            try:
                loss, per = losses(model, x, y)
                if paired:
                    good, bad = per.chunk(2)
                    loss = (
                        good.mean()
                        - 0.40 * F.logsigmoid(2 * (bad - good) - 0.25).mean()
                    )
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
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
                continue
            targets += int((y != -100).sum())
            positions += x.size
            draw_counts.update(names)
            step += 1
            if reader is not None and step % 5 == 0:
                reader_loss = reader.update(
                    min(8192, capacity), 0.0008 * min(1.0, (reader.steps + 1) / 16)
                )
                if reader.steps % 150 == 0:
                    log(
                        stage=stage,
                        reader_step=reader.steps,
                        reader_examples=reader.seen,
                        reader_loss=reader_loss,
                    )
            if step % 100 == 0 or step == 1:
                log(
                    stage=stage,
                    step=step,
                    loss=float(loss),
                    positions=positions,
                    targets=targets,
                    lr=rate,
                )
            if time.time() - last > 300 and rank is None:
                values = validate(model, val)
                log(stage=stage, step=step, validation=values)
                if quality(values) < best_score:
                    best_score, best_values = quality(values), values
                    save_net(model, path)
                last = time.time()
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        candidates = {}
        if rank is None:
            values = validate(model, val)
            if quality(values) < best_score:
                save_net(model, path)
                best_values = values
            if not path.is_file():
                raise RuntimeError(
                    "No instruction checkpoint improved on the continued model"
                )
            (path.parent / "config.json").write_text((OUT / "config.json").read_text())
        else:
            for factor in (0, 0.35, 0.70, 1.0):
                set_scale(model, factor)
                candidates[str(factor)] = validate(model, val)
            factor = min(
                (0, 0.35, 0.70, 1.0), key=lambda f: quality(candidates[str(f)])
            )
            set_scale(model, factor)
            best_values = candidates[str(factor)]
            save_adapter(
                model,
                OUT / "adapters" / stage,
                rank,
                2 * rank * factor,
                "base" if stage == "dialogue" else "base+dialogue",
            )
            merge(model)
            save_net(model, path)
        model.load_state_dict(load_file(str(path)))
        result = dict(
            steps=step,
            positions=positions,
            targets=targets,
            sampled_records=dict(draw_counts),
            initial=starting,
            validation=best_values,
            candidates=candidates,
            scale=factor if rank else None,
            elapsed_seconds=time.time() - started,
        )
        progress.write_text(json.dumps(result, indent=2) + "\n")
        if reader is not None:
            reader.snapshot(OUT / "reader-warm")
            (OUT / "reader-warm/training.json").write_text(
                json.dumps(
                    dict(
                        steps=reader.steps, examples=reader.seen, tokens=reader.tokens
                    ),
                    indent=2,
                )
                + "\n"
            )
        report[stage] = result
        log(stage=stage, finished=result)
    (OUT / "post-finished.json").write_text(
        json.dumps(
            dict(
                parent=json.loads((OUT / "recovery.json").read_text())["parent"],
                stages=report,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
