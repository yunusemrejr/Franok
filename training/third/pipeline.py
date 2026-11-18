"""Run the third training recipe on one Colab L4 from a local or published Franok parent."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

from training.extended import prepare, dialogue, conversation_data
from .batches import WEIGHTS
from .data import tokenizer
from .followups import build as prepare_followups


def legacy():
    needed = {name for table in WEIGHTS.values() for name in table}
    prepare.SOURCES = {k: v for k, v in prepare.SOURCES.items() if k in needed}
    prepare.prepare()
    dialogue.prepare()
    conversation_data.main()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=240)
    p.add_argument(
        "--parent",
        type=Path,
        help="A model.safetensors from the previous complete Franok run",
    )
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--prepare-legacy", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.prepare_legacy:
        legacy()
        return
    if bool(args.parent) != bool(args.tokenizer):
        p.error("Supply the local parent weights and their tokenizer together.")
    if args.minutes <= 0:
        p.error("Set a positive training allocation.")
    if args.parent:
        os.environ["FRANOK_PARENT"] = str(args.parent.resolve())
        os.environ["FRANOK_TOKENIZER"] = str(args.tokenizer.resolve())
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    os.environ.setdefault("FRANOK_DATA_WORKERS", "2")
    root = Path("work/third/run")
    root.mkdir(exist_ok=True, parents=True)
    if (root / "rl-finished.json").exists() and (
        root / "reader/training.json"
    ).exists():
        print("This run already contains its completed reinforcement checkpoint.")
        return
    # Cache the actual Franok tokenizer before child preparation processes start.
    tokenizer()
    start = time.time()
    minute = args.minutes * 60 / 240

    def call(module, *values):
        cmd = [sys.executable, "-u", "-m", module, *map(str, values)]
        remaining = start + 240 * minute - time.time()
        if remaining <= 0:
            raise TimeoutError("Training allocation expired")
        subprocess.run(cmd, check=True, timeout=remaining)

    def text(lang):
        for corpus, size in [
            ("web", 180000000),
            ("stories", 32000000),
            ("web-more", 180000000),
        ]:
            call(
                "training.third.data",
                lang,
                "--collection",
                corpus,
                "--tokens",
                size,
                "--stream",
            )

    def instructions():
        call("training.third.pipeline", "--prepare-legacy")
        call("training.third.instructions")
        call("training.third.instructions", "--collection", "tasks")
        call("training.third.instructions", "--collection", "short-replies")
        prepare_followups()
        call("training.third.spans")

    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(text, lang) for lang in ("en", "tr")]
        jobs.append(pool.submit(instructions))
        if not (root / "pretrain.done").exists():
            call("training.third.pretrain", "--deadline", start + 160 * minute)
        for job in jobs:
            job.result()
    if not (root / "post-finished.json").exists():
        call(
            "training.third.posttrain",
            "--sft-end",
            start + 201 * minute,
            "--dialogue-end",
            start + 213 * minute,
            "--evidence-end",
            start + 221 * minute,
        )
    if not (root / "rl-finished.json").exists():
        call("training.third.reinforce", "--deadline", start + 232 * minute)
    call(
        "training.third.reader",
        "--init",
        root / "reader-warm/head.safetensors",
        "--deadline",
        start + 236 * minute,
    )


if __name__ == "__main__":
    main()
