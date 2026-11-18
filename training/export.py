"""Export grouped symmetric four- or eight-bit weights for the Franok WASM decoder."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open


def quantize_grouped(group, bits, optimize, importance=None, full_range=False):
    """Minimize squared weight error over several scale candidates per group of 32."""
    metric = 1.0 if importance is None else importance
    limit = 2 ** (bits - 1) - 1
    # The signed scale can place the extra negative code on the larger extreme.
    # Existing runtimes already read four-bit codes as -8 through +7.
    sign = 1.0
    low = -limit
    if full_range:
        if bits != 4 or not optimize:
            raise ValueError("Full-range fitting requires Q4 least squares")
        minimum = group.min(1, keepdims=True)
        maximum = group.max(1, keepdims=True)
        sign = np.where(-minimum >= maximum, np.float32(1), np.float32(-1))
        group = group * sign
        low = -8
    initial = np.maximum(np.max(np.abs(group), axis=1, keepdims=True) / limit, 1e-12)
    scale = initial
    ints = np.rint(group / scale).clip(low, limit)
    if bits == 4 and optimize:
        best = np.sum(metric * (group - scale * ints) ** 2, axis=1, keepdims=True)
        for fraction in (1.0, 0.88, 0.76):
            step = initial * fraction
            for _ in range(2):
                candidate = np.rint(group / step).clip(low, limit)
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
    return (scale * sign).astype("<f4"), ints.astype("int8")


def refine_block(group, scale, codes, covariance, low):
    """Minimize estimated output error using correlations within each 32-weight group."""
    q = codes.astype("float32")
    error = group / scale - q
    gradient = np.einsum("bi,bij->bj", error, covariance)
    for column in range(group.shape[1]):
        ideal = q[:, column] + gradient[:, column] / covariance[:, column, column]
        moved = np.rint(ideal).clip(low, 7) - q[:, column]
        indices = np.flatnonzero(moved)
        q[indices, column] += moved[indices]
        gradient[indices] -= moved[indices, None] * covariance[indices, column]
    projected = np.einsum("bi,bij->bj", q, covariance)
    denominator = (projected * q).sum(1, keepdims=True)
    np.divide(
        (projected * group).sum(1, keepdims=True),
        denominator,
        out=scale,
        where=denominator > 1e-12,
    )
    return scale, q.astype("int8")


def refine_pairs(rows, scales, codes, covariance):
    """Refit each half of a 64-channel block given its neighbour's remaining error."""
    pairs = len(covariance)
    # Each input group shares its covariance across all output rows of this bank.
    weights = rows.reshape(-1, pairs, 64).transpose(1, 0, 2)
    scale = scales.reshape(-1, pairs, 2).transpose(1, 0, 2).copy()
    q = codes.reshape(-1, pairs, 64).transpose(1, 0, 2).copy()
    n_rows = weights.shape[1]

    for side in (1, 0):
        own = slice(side * 32, side * 32 + 32)
        other = slice((1 - side) * 32, (1 - side) * 32 + 32)
        block = covariance[:, own, own]
        projection = np.linalg.solve(block, covariance[:, own, other]).transpose(0, 2, 1)
        residual = weights[:, :, other] - scale[:, :, 1 - side, None] * q[:, :, other]
        target = (weights[:, :, own] + residual @ projection).reshape(-1, 32)
        metrics = np.broadcast_to(block[:, None], (pairs, n_rows, 32, 32)).reshape(-1, 32, 32)
        diagonal = np.diagonal(metrics, axis1=1, axis2=2)
        step, candidate = quantize_grouped(target, 4, True, diagonal, True)
        step, candidate = refine_block(target, step, candidate, metrics, -8)
        old_scale = scale[:, :, side].reshape(-1, 1)
        old_codes = q[:, :, own].reshape(-1, 32)

        def error(s, integers):
            d = target - s * integers
            return np.einsum("bi,bij,bj->b", d, metrics, d)

        take = error(step, candidate) < error(old_scale, old_codes)
        scale[:, :, side] = np.where(take, step[:, 0], old_scale[:, 0]).reshape(pairs, n_rows)
        q[:, :, own] = np.where(take[:, None], candidate, old_codes).reshape(pairs, n_rows, 32)
    return scale.transpose(1, 0, 2).reshape(-1, 1), q.transpose(1, 0, 2).reshape(-1, 32)


def export(
    run,
    out,
    tokenizer,
    bits=4,
    optimize=True,
    calibration=None,
    full_range=False,
    covariance=False,
    paired_groups=False,
):
    if bits not in (4, 8):
        raise ValueError("Supported weight widths are 4 and 8")
    if calibration is not None and (bits != 4 or not optimize):
        raise ValueError("Activation weighting requires a least-squares Q4 export")
    if covariance and calibration is None:
        raise ValueError("Supply a measured block-covariance calibration")
    if paired_groups and not (covariance and full_range):
        raise ValueError("Fitting neighbouring groups requires full-range Q4 and covariance calibration")
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
                cov = None
                if covariance:
                    cov = calibration[key + ".cov32"]
                    if (
                        cov.shape != (banks, groups_per_row, 32, 32)
                        or not np.isfinite(cov).all()
                    ):
                        raise ValueError("Invalid block covariance for " + key)
                chunk = 32768
                if paired_groups:
                    pair_cov = calibration[key + ".cov64"]
                    if pair_cov.shape != (banks, groups_per_row // 2, 64, 64) or groups_per_row % 2:
                        raise ValueError("Invalid adjacent-group calibration for " + key)
                    # Keep a chunk within a single bank, with complete output rows.
                    chunk = math.gcd(128, a.shape[-2]) * groups_per_row
                for first in range(0, len(blocks), chunk):
                    group = blocks[first : first + chunk].astype("float32")
                    weights_for_groups = None
                    if metric is not None:
                        indices = np.arange(first, first + len(group))
                        bank = indices // (a.shape[-2] * groups_per_row)
                        weights_for_groups = metric[bank, indices % groups_per_row]
                    scale, q = quantize_grouped(
                        group, bits, optimize, weights_for_groups, full_range
                    )
                    if cov is not None:
                        metric_blocks = cov[bank, indices % groups_per_row]
                        scale, q = refine_block(
                            group, scale, q, metric_blocks, -8 if full_range else -7
                        )
                    if paired_groups:
                        scale, q = refine_pairs(group, scale, q, pair_cov[bank[0]])
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
                full_range=full_range,
                block_covariance=covariance,
                paired_groups=paired_groups,
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
    p.add_argument(
        "--full-range",
        action="store_true",
        help="Fit all sixteen Q4 codes with signed scales",
    )
    p.add_argument(
        "--covariance",
        action="store_true",
        help="Fit Q4 codes with measured within-block activation covariance",
    )
    p.add_argument("--paired-groups", action="store_true", help="Correct errors across adjacent Q4 groups.")
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
        args.full_range,
        args.covariance,
        args.paired_groups,
    )
