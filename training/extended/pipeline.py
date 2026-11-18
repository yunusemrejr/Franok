"""Train the second release from a local scratch run or its exact published first checkpoint."""

import argparse, json, os, subprocess, sys
from pathlib import Path
from .prepare import ROOT, tokenizer_path
from .train import OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scratch-run",
        type=Path,
        help="Initial Franok run directory with model.safetensors. Pair with its own --tokenizer.",
    )
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--lm-minutes", type=float, default=63)
    p.add_argument("--post-minutes", type=float, default=34)
    p.add_argument("--rl-minutes", type=float, default=11)
    args = p.parse_args()
    if bool(args.scratch_run) != bool(args.tokenizer):
        p.error(
            "Supply both the local parent and its tokenizer, or use the published first checkpoint."
        )
    if any(v < 5 for v in (args.lm_minutes, args.post_minutes, args.rl_minutes)):
        p.error("Use at least five minutes per stage.")
    if (OUT / "progress.json").exists():
        p.error(
            "A training run already exists in work/extended/run. Keep that run and start in a clean checkout."
        )
    env = os.environ.copy()
    if args.tokenizer:
        env["FRANOK_TOKENIZER"] = str(args.tokenizer.resolve())

    def call(module, *values):
        subprocess.run(
            [sys.executable, "-u", "-m", module, *map(str, values)], check=True, env=env
        )

    for module in ("prepare", "dialogue", "conversation_data"):
        call("training.extended." + module)
    parent = (
        ["--parent-from", args.scratch_run / "model.safetensors"]
        if args.scratch_run
        else []
    )
    call("training.extended.train", "--only-lm", "--minutes", args.lm_minutes, *parent)
    call(
        "training.extended.train",
        "--post-from",
        OUT / "continued.safetensors",
        "--minutes",
        args.post_minutes,
    )
    call("training.extended.rl", "--minutes", args.rl_minutes)
    # Export the RL merge while keeping the pre-RL merge for comparisons.
    import shutil

    final = ROOT.parent / "final"
    final.mkdir(exist_ok=True)
    for src, dest in [
        ("rl-model.safetensors", "model.safetensors"),
        ("config.json", "config.json"),
    ]:
        shutil.copy2(OUT / src, final / dest)
    tok = args.tokenizer or tokenizer_path()
    for bits in (4, 8):
        call(
            "training.export",
            "--run",
            final,
            "--tokenizer",
            tok,
            "--out",
            ROOT.parent / "export",
            "--bits",
            bits,
        )


if __name__ == "__main__":
    main()
