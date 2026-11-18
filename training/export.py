"""Export grouped symmetric four- or eight-bit weights for the Franok WASM decoder."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def quantize_grouped(group, bits, optimize, importance=None):
    """Minimize squared weight error over several scale candidates per group of 32."""
    metric = 1.0 if importance is None else importance
    limit = 2 ** (bits - 1) - 1
    initial = np.maximum(np.max(np.abs(group), axis=1, keepdims=True) / limit, 1e-12)
    scale = initial
    ints = np.rint(group / scale).clip(-limit, limit)
    if bits == 4 and optimize:
        best = np.sum(metric * (group - scale * ints) ** 2, axis=1, keepdims=True)
        for fraction in (1.0, 0.88, 0.76):
            step = initial * fraction
            for _ in range(2):
                candidate = np.rint(group / step).clip(-limit, limit)
                # Least-squares scale for a fixed set of codes. Zero groups retain a finite scale.
                step = np.maximum(
                    np.sum(metric * candidate * group, axis=1, keepdims=True)
                    / np.maximum(
                        1, np.sum(metric * candidate**2, axis=1, keepdims=True)
                    ),
                    1e-12,
                )
                error = np.sum(
                    metric * (group - step * candidate) ** 2, axis=1, keepdims=True
                )
                take = error < best
                scale = np.where(take, step, scale)
                ints = np.where(take, candidate, ints)
                best = np.minimum(best, error)
    return scale.astype("<f4"), ints.astype("int8")


def export(run, out, tokenizer, bits=4, optimize=True, calibration=None):
    if bits not in (4, 8):
        raise ValueError("Supported weight widths are 4 and 8")
    if calibration is not None and (bits != 4 or not optimize):
        raise ValueError("Activation weighting requires a least-squares Q4 export")
    out.mkdir(parents=True, exist_ok=True)
    weights = out / f"model.q{bits}"
    catalog = {}
    with (
        safe_open(run / "model.safetensors", framework="numpy") as model,
        weights.open("wb") as dest,
    ):
        for key in sorted(model.keys()):
            a = model.get_tensor(key)
            offset = dest.tell()
            if a.ndim == 1:
                dest.write(a.astype("<f4").tobytes())
                typ = "f32"
            else:
                if a.shape[-1] % 32:
                    raise ValueError(f"Unaligned {key}")
                blocks = a.reshape(-1, 32)
                groups_per_row = a.shape[-1] // 32
                metric = None if calibration is None else calibration[key]
                if metric is not None:
                    banks = a.shape[0] if a.ndim == 3 else 1
                    if (
                        metric.shape != (banks, a.shape[-1])
                        or not np.isfinite(metric).all()
                        or np.min(metric) <= 0
                    ):
                        raise ValueError("Invalid channel importance for " + key)
                    metric = metric.reshape(banks, groups_per_row, 32)
                for first in range(0, len(blocks), 32768):
                    group = blocks[first : first + 32768].astype("float32")
                    weights_for_groups = None
                    if metric is not None:
                        indices = np.arange(first, first + len(group))
                        bank = indices // (a.shape[-2] * groups_per_row)
                        weights_for_groups = metric[bank, indices % groups_per_row]
                    scale, q = quantize_grouped(
                        group, bits, optimize, weights_for_groups
                    )
                    if bits == 4:
                        biased = (q + 8).astype("uint8")
                        payload = biased[:, :16] | (biased[:, 16:] << 4)
                    else:
                        payload = q.view("uint8")
                    dest.write(
                        np.concatenate([scale.view("uint8"), payload], axis=1).tobytes()
                    )
                typ = f"q{bits}"
            catalog[key] = dict(
                offset=offset, bytes=dest.tell() - offset, shape=list(a.shape), type=typ
            )
    cfg = json.loads((run / "config.json").read_text())
    spec = json.loads(tokenizer.read_text())
    spec["decoder"] = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": False,
    }
    (out / "tokenizer.json").write_text(
        json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    )
    with weights.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = "model.json" if bits == 4 else "model-q8.json"
    (out / manifest).write_text(
        json.dumps(
            dict(
                format=f"franok-q{bits}-v1",
                scale_method=(
                    "activation-weighted-least-squares"
                    if calibration is not None
                    else ("least-squares" if bits == 4 and optimize else "peak")
                ),
                config=cfg,
                tensors=catalog,
                bytes=weights.stat().st_size,
                sha256=digest,
            ),
            indent=2,
        )
    )
    print(weights, weights.stat().st_size, digest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("work/run"))
    p.add_argument("--out", type=Path, default=Path("work/export"))
    p.add_argument("--tokenizer", type=Path, default=Path("work/data/tokenizer.json"))
    p.add_argument("--bits", type=int, choices=[4, 8], default=4)
    p.add_argument(
        "--scale",
        choices=["least-squares", "peak"],
        default="least-squares",
        help="Q4 per-block scale search; peak reproduces v0.1",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        help="Per-channel importance measured by training.calibration",
    )
    args = p.parse_args()
    calibration = None
    if args.calibration:
        with np.load(args.calibration) as archive:
            calibration = {k: archive[k] for k in archive.files}
    export(
        args.run,
        args.out,
        args.tokenizer,
        args.bits,
        args.scale == "least-squares",
        calibration,
    )
