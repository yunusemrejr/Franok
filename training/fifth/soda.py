"""Write SODA dialogues as plain-text shards for the Franok raw-text stream.

Each dialogue becomes one document of ``Speaker: utterance`` lines, wrapped in
the same special tokens ``training/third/data.py`` writes around web text:
``<bos> <en> ... <end>``. Shards are little-endian uint16 token files that
``Shards("train")`` in ``training/third/pretrain.py`` picks up through its
``train-*.bin`` glob.
"""

import argparse
import json
import os
from array import array
from pathlib import Path
import re
import sys

from tokenizers import Tokenizer

DATASET = "allenai/soda"
LICENCE = "CC-BY-4.0"
BOS, EN, END = 1, 5, 2
SHARD_TOKENS = 32_000_000
VOCAB_LIMIT = 8192
BATCH_DOCS = 512


def tokenizer():
    path = Path(os.environ.get("FRANOK_TOKENIZER", "web/tokenizer.json"))
    if not path.exists():
        raise SystemExit(f"Tokenizer not found at {path}; set FRANOK_TOKENIZER.")
    return Tokenizer.from_file(str(path))


def document(row):
    """Return the dialogue as one document, or None when it must be skipped."""
    dialogue, speakers = row["dialogue"], row["speakers"]
    if not isinstance(dialogue, list) or not isinstance(speakers, list):
        raise ValueError(f"SODA row has non-list dialogue or speakers: {row!r:.200}")
    if len(dialogue) != len(speakers) or len(dialogue) < 4:
        return None
    lines = []
    for speaker, utterance in zip(speakers, dialogue):
        text = str(utterance).strip()
        if not text:
            return None
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def soda_rows(limit_rows=None):
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train", streaming=True)
    if limit_rows is not None:
        dataset = dataset.take(limit_rows)
    yield from dataset


def token_ids(tok, texts):
    for encoding in tok.encode_batch(texts):
        ids = encoding.ids
        if not ids or min(ids) < 0 or max(ids) >= VOCAB_LIMIT:
            raise ValueError(
                f"Tokenizer produced token ids outside [0, {VOCAB_LIMIT}): {ids}"
            )
        yield ids


def write_shards(rows, out_dir, tokens, name="soda", tok=None):
    """Stream rows into ``train-<name>-NNNNN.bin`` and return the receipt."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / f"{name}.json"
    existing = sorted(out.glob(f"train-{name}-*.bin"))
    if existing or receipt_path.exists():
        present = [p.name for p in existing]
        if receipt_path.exists():
            present.append(receipt_path.name)
        raise SystemExit(
            f"Refusing to run: {name} data already exists in {out} ({present}). "
            "Delete it first if you really mean to regenerate."
        )

    tok = tok or tokenizer()
    buffer = array("H")
    flushed = shard = kept = read = 0
    files = []
    stopped = False

    def flush():
        nonlocal buffer, flushed, shard
        if not buffer:
            return
        if sys.byteorder != "little":
            buffer.byteswap()
        file = out / f"train-{name}-{shard:05d}.bin"
        tmp = file.with_suffix(".part")
        tmp.write_bytes(buffer.tobytes())
        if file.exists():
            tmp.unlink()
            raise SystemExit(f"Refusing to overwrite existing shard {file}")
        tmp.replace(file)
        files.append(file.name)
        flushed += len(buffer)
        buffer = array("H")
        shard += 1

    pending = []

    def drain(texts):
        nonlocal stopped
        for ids in token_ids(tok, texts):
            buffer.extend((BOS, EN, *ids, END))
            if len(buffer) >= SHARD_TOKENS:
                flush()
            if flushed + len(buffer) >= tokens:
                stopped = True
                return

    for row in rows:
        read += 1
        text = document(row)
        if text is None:
            continue
        kept += 1
        pending.append(text)
        if len(pending) >= BATCH_DOCS:
            drain(pending)
            pending.clear()
        if stopped:
            break
    if pending and not stopped:
        drain(pending)
    flush()

    if not kept or flushed == 0:
        raise RuntimeError(f"No usable SODA documents were read ({read} rows)")
    receipt = {
        "dataset": DATASET,
        "licence": LICENCE,
        "rows_read": read,
        "rows_kept": kept,
        "tokens": flushed,
        "shards": files,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Write SODA dialogues as Franok raw-text shards."
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--name", default="soda")
    p.add_argument("--limit-rows", type=int)
    args = p.parse_args(argv)
    if args.tokens <= 0:
        p.error("--tokens must be positive")
    if args.limit_rows is not None and args.limit_rows <= 0:
        p.error("--limit-rows must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        p.error("--name may contain only letters, digits, '_' and '-'")
    receipt = write_shards(
        soda_rows(args.limit_rows), args.out_dir, args.tokens, args.name
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
