"""Single-GPU continuation, instruction tuning and two mergeable LoRA stages."""

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from franok.model import Config, Franok
from franok.lora import attach, merge, save_adapter
from .batches import Corpus
from .prepare import ROOT, PARENT

OUT = ROOT.parent / "run"
STAGES = [
    ("lm", 0.63, 0.00046),
    ("sft", 0.19, 0.00017),
    ("dialogue", 0.11, 0.00065),
    ("evidence", 0.07, 0.00028),
]


def save_net(m, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {
        k: v.detach().to("cpu", torch.float16).contiguous()
        for k, v in m.state_dict().items()
    }
    tmp = path.with_suffix(".tmp")
    save_file(weights, str(tmp))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deadline", type=float, help="Absolute Unix time to end training")
    p.add_argument("--minutes", type=float, default=100)
    p.add_argument("--batch-tokens", type=int, default=12288)
    p.add_argument(
        "--post-from",
        type=Path,
        help="Continue at the supervised stage from the completed LM phase",
    )
    p.add_argument(
        "--parent-from",
        type=Path,
        help="Use the initial random-lineage training result from this checkout",
    )
    p.add_argument(
        "--only-lm",
        action="store_true",
        help="Save continued.safetensors after continued pretraining",
    )
    args = p.parse_args()
    if args.only_lm and args.post_from:
        p.error("--only-lm and --post-from cannot be combined")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Select a BF16-capable GPU, such as the L4.")
    torch.manual_seed(51627)
    rng = np.random.default_rng(51627)
    torch.set_num_threads(4)
    cfg = Config(window=1024)
    OUT.mkdir(exist_ok=True, parents=True)
    out_cfg = json.dumps(asdict(cfg), indent=2)
    (OUT / "config.json").write_text(out_cfg)
    model = Franok(cfg)
    base = (
        args.post_from
        or args.parent_from
        or hf_hub_download("yunusemrejr/Franok", "model.safetensors", revision=PARENT)
    )
    parent_id = PARENT
    if args.parent_from:
        with args.parent_from.open("rb") as source:
            parent_id = "sha256:" + hashlib.file_digest(source, "sha256").hexdigest()
    if args.post_from:
        parent_id = json.loads((OUT / "progress.json").read_text())["parent"]
    model.load_state_dict(load_file(base))
    model.cuda()
    train, val = Corpus("train"), Corpus("val")
    log_file = OUT / "metrics.jsonl"

    def log(**values):
        values = {"utc": time.time(), **values}
        print(values, flush=True)
        with log_file.open("a") as f:
            f.write(json.dumps(values) + "\n")

    def nll(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            return F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)

    @torch.no_grad()
    def validate(phase):
        model.eval()
        scores = {}
        for kind in ("lm", "dialogue"):
            vrng = np.random.default_rng(31337)
            scores[kind] = (
                sum(nll(*val.batch(vrng, 8, 512, kind)).item() for _ in range(5)) / 5
            )
        log(phase=phase, validation=scores)
        model.train()
        return scores

    def preference(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            losses = F.cross_entropy(
                logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="none"
            ).reshape_as(y)
            seq = -losses.sum(-1) / (y != -100).sum(-1).clamp_min(1)
            good, bad = seq.chunk(2)
            return -good.mean() - 0.45 * F.logsigmoid(2.0 * (good - bad) - 0.35).mean()

    start = time.time()
    deadline = args.deadline or start + args.minutes * 60
    duration = deadline - start - 65
    if duration < 180:
        raise ValueError("Insufficient time left for four stages")
    log(
        parent=parent_id,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        config=asdict(cfg),
        seed=51627,
        deadline=deadline,
        phase_seconds=duration,
    )
    validate("parent")
    seen = targets = steps = 0
    report = {}
    capacity = args.batch_tokens
    stages = [("lm", 1.0, 0.00046)] if args.only_lm else STAGES
    if args.post_from:
        prior = json.loads((OUT / "progress.json").read_text())
        if set(prior["phases"]) != {"lm"}:
            raise ValueError("Expected the finished LM checkpoint")
        report = prior["phases"]
        seen = prior["positions"]
        targets = prior["targets"]
        steps = report["lm"]["steps"]
        stages = [
            ("sft", 0.53, 0.00017),
            ("dialogue", 0.29, 0.00065),
            ("evidence", 0.18, 0.00028),
        ]
    boundary = start
    for stage, fraction, peak in stages:
        begin = time.time()
        boundary += duration * fraction
        if stage in ("dialogue", "evidence"):
            rank = 32 if stage == "dialogue" else 16
            attach(model, rank, rank * 2)
        params = [a for a in model.parameters() if a.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=peak,
            betas=(0.9, 0.95),
            weight_decay=0.04 if stage in ("lm", "sft") else 0.01,
            fused=True,
        )
        step = 0
        start_seen = seen
        start_targets = targets
        last_save = last_eval = begin
        while time.time() < boundary or step == 0:
            progress = min(1, (time.time() - begin) / max(1, boundary - begin))
            # Sequence-length curriculum expands the trained window fourfold.
            if stage == "lm":
                length = 256 if progress < 0.24 else 512 if progress < 0.78 else 1024
            else:
                length = 1024 if step % 6 == 0 else 512
            pref = stage == "evidence" and step % 3 != 0
            b = max(2, (capacity // length // (2 if pref else 1)) // 2 * 2)
            x, y = (
                train.pairs(rng, b, length)
                if pref
                else train.batch(rng, b, length, stage)
            )
            lr = (
                peak
                * min(1, (step + 1) / 55)
                * (0.20 + 0.80 * (1 + math.cos(math.pi * progress)) / 2)
            )
            opt.param_groups[0]["lr"] = lr
            opt.zero_grad(set_to_none=True)
            try:
                loss = preference(x, y) if pref else nll(x, y)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(params, 1)
                if not torch.isfinite(grad):
                    raise RuntimeError("Non-finite gradient")
                opt.step()
            except torch.OutOfMemoryError:
                if capacity <= 4096:
                    raise
                capacity = capacity // 2
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                log(phase=stage, reduced_batch_tokens=capacity)
                continue
            seen += x.numel()
            targets += int((y != -100).sum())
            step += 1
            steps += 1
            if step == 1 or step % 100 == 0:
                log(
                    phase=stage,
                    step=step,
                    length=length,
                    lr=lr,
                    loss=loss.item(),
                    positions=seen,
                    targets=targets,
                    tokens_per_second=round(
                        (seen - start_seen) / max(1, time.time() - begin)
                    ),
                    peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                )
            if time.time() - last_eval > 420:
                validate(stage)
                last_eval = time.time()
            if stage in ("lm", "sft") and time.time() - last_save > 700:
                save_net(model, OUT / "recovery.safetensors")
                last_save = time.time()
        metrics = validate(stage)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        if stage == "lm":
            save_net(model, OUT / "continued.safetensors")
        elif stage == "sft":
            save_net(model, OUT / "base" / "model.safetensors")
            (OUT / "base" / "config.json").write_text(out_cfg)
            model.load_state_dict(load_file(str(OUT / "base" / "model.safetensors")))
        else:
            save_adapter(
                model,
                OUT / "adapters" / stage,
                rank,
                rank * 2,
                "base" if stage == "dialogue" else "base+dialogue",
            )
            merge(model)
            if stage == "dialogue":
                # The on-disk boundary is FP16, including for the second adapter.
                for v in model.parameters():
                    v.data.copy_(v.data.half())
            else:
                save_net(model, OUT / "model.safetensors")
        report[stage] = {
            "steps": step,
            "positions": seen - start_seen,
            "targets": targets - start_targets,
            "elapsed_seconds": round(time.time() - begin, 2),
            "validation": metrics,
        }
        tmp = OUT / "progress.tmp"
        tmp.write_text(
            json.dumps(
                dict(phases=report, positions=seen, targets=targets, parent=parent_id),
                indent=2,
            )
        )
        tmp.replace(OUT / "progress.json")
        log(finished=stage, **report[stage])
    if args.only_lm:
        return
    with (OUT / "model.safetensors").open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    (OUT / "finished.json").write_text(
        json.dumps(dict(sha256=sha, steps=steps, elapsed_seconds=time.time() - start))
    )
    print("Finished", sha, flush=True)


if __name__ == "__main__":
    main()
