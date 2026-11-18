"""Fit a compact bilingual session-search encoder on open parallel questions and evidence."""

import argparse
from collections import Counter
import hashlib
import gzip
import json
from pathlib import Path
import time
import unicodedata

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tokenizers import Tokenizer
from training.extended.prepare import SOURCES
from training.third.instructions import read_squad


def normalize(text):
    return unicodedata.normalize("NFKC", text).replace("İ", "i").lower().strip()


def number(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def squad(articles):
    return {
        q["id"]: (a["title"], p["context"], q["question"])
        for a in articles
        for p in a["paragraphs"]
        for q in p["qas"]
        if not q["is_impossible"] and q["answers"]
    }


def corpus(args):
    en = squad(read_squad("en", args.en))
    tr = squad(read_squad("tr", args.tr))
    for qid, (title, ep, eq) in en.items():
        if qid not in tr:
            continue
        _, tp, tq = tr[qid]
        yield "squad", number("squad:" + title.casefold()) % 40 == 0, eq, tq, ep, tp
    repo, rev, name = SOURCES["parallel"]
    file = hf_hub_download(repo, name, revision=rev, repo_type="dataset")
    seen = set()
    for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
        for r in batch.to_pylist():
            row = [
                r.get(k)
                for k in (
                    "question",
                    "question-turkish",
                    "response",
                    "response-turkish",
                )
            ]
            if any(not s or not 8 <= len(s) <= 1100 for s in row):
                continue
            key = normalize(row[0])
            if key in seen:
                continue
            seen.add(key)
            yield "parallel", number("parallel:" + key) % 40 == 0, *row
            if len(seen) >= 45000:
                break
        if len(seen) >= 45000:
            break

    if args.translations:
        source = hf_hub_download(
            "sentence-transformers/parallel-sentences",
            "Tatoeba/Tatoeba-en-tr-train.tsv.gz",
            repo_type="dataset",
            revision="767e169873643dc102321cbb83bf5798efdc7d0f",
        )
        known = set()
        with gzip.open(source, "rt", encoding="utf8") as parallel:
            for line in parallel:
                parts = line.rstrip().split("\t")
                if len(parts) < 2:
                    continue
                en, tr = parts[0], parts[1]
                if not all(8 <= len(x) <= 440 for x in (en, tr)):
                    continue
                key = normalize(en)
                if key in known:
                    continue
                known.add(key)
                yield "tatoeba", number("tatoeba:" + key) % 40 == 0, en, tr, en, tr
                if len(known) >= 220000:
                    break


def prepare(args):
    args.data.mkdir(exist_ok=True, parents=True)
    ids = {}
    texts = []
    counts = Counter()
    records = []

    def entry(text):
        k = normalize(text)
        if k not in ids:
            ids[k] = len(texts)
            texts.append(k)
        return ids[k]

    for kind, val, eq, tq, ep, tp in corpus(args):
        counts[kind + "-" + ("val" if val else "train")] += 1
        records.append([int(val), entry(eq), entry(tq), entry(ep), entry(tp)])
    t = Tokenizer.from_file(str(args.tokenizer))
    table = np.lib.format.open_memmap(
        args.data / "tokens.npy", mode="w+", dtype="<i2", shape=(len(texts), 160)
    )
    lengths = np.zeros(len(texts), dtype="<i2")
    df = np.ones(t.get_vocab_size(), dtype=np.int64)
    training_texts = set(v for row in records if not row[0] for v in row[1:])
    for i in range(0, len(texts), 256):
        for j, example in enumerate(t.encode_batch(texts[i : i + 256])):
            tokens = example.ids
            if len(tokens) > 160:
                tokens = tokens[:96] + tokens[-64:]
            table[i + j, : len(tokens)] = tokens
            lengths[i + j] = len(tokens)
            if i + j in training_texts:
                df[np.unique(tokens).astype(np.int64)] += 1
    np.save(args.data / "lengths.npy", lengths)
    np.save(args.data / "records.npy", np.array(records, dtype="<i4"))
    idf = np.clip(np.log((len(training_texts) + 2) / df), 0.35, 6.0).astype("<f4")
    idf[0] = 0
    np.save(args.data / "idf.npy", idf)
    (args.data / "receipt.json").write_text(
        json.dumps(dict(counts=counts, distinct_texts=len(texts)), indent=2) + "\n"
    )


class Encoder(nn.Module):
    def __init__(self, width, idf):
        super().__init__()
        self.emb = nn.Embedding(len(idf), width, padding_idx=0)
        self.register_buffer("idf", torch.from_numpy(idf.copy()))

    def forward(self, x):
        weights = self.idf[x].unsqueeze(-1)
        return F.normalize((self.emb(x) * weights).sum(1), dim=-1)


def train(args):
    rng = np.random.default_rng(66033)
    torch.set_num_threads(args.threads)
    torch.manual_seed(66033)
    table = np.load(args.data / "tokens.npy", mmap_mode="r")
    length = np.load(args.data / "lengths.npy")
    records = np.load(args.data / "records.npy")
    idf = np.load(args.data / "idf.npy")
    train, held = records[records[:, 0] == 0], records[records[:, 0] != 0]
    model = Encoder(96, idf)
    if args.init:
        with safe_open(str(args.init), framework="pt") as f:
            embedding = f.get_tensor("embed.weight").float()
        projection = torch.randn(embedding.shape[1], 96) / embedding.shape[1] ** 0.5
        with torch.no_grad():
            model.emb.weight.copy_(F.normalize(embedding @ projection, dim=-1))
            model.emb.weight[0].zero_()
    else:
        with torch.no_grad():
            model.emb.weight.copy_(F.normalize(model.emb.weight, dim=-1))
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)

    def encode(rows):
        n = max(1, int(length[rows].max()))
        return model(torch.from_numpy(table[rows, :n].astype(np.int64)))

    def sample(ids, mode):
        a, b = mode
        left, right = ids[:, a], ids[:, b].copy()
        identical = left == right
        # A translation-only row has no separate context. Always align its two
        # languages, including when the rest of the batch uses monolingual QA.
        right[identical] = ids[identical, 2 if a in (1, 3) else 1]
        return encode(left), encode(right)

    modes = [(1, 3), (2, 4), (1, 4), (2, 3), (1, 2), (3, 4)]
    fixed = np.random.default_rng(66943)
    val = held[fixed.choice(len(held), min(1024, len(held)), replace=False)]

    # In-batch duplicates from one paragraph are positives, never negatives.
    def objective(a, b, ids):
        same = torch.as_tensor(ids[:, None, 3] == ids[None, :, 3])
        s = a @ b.T / 0.07
        loss = -(s.masked_fill(~same, -1e4).logsumexp(1) - s.logsumexp(1)).mean()
        return loss, same.gather(1, s.argmax(1)[:, None]).float().mean()

    @torch.no_grad()
    def validate():
        losses = []
        acc = []
        for mode in modes[:5]:
            for i in range(0, len(val), 128):
                chunk = val[i : i + 128]
                l, a = objective(*sample(chunk, mode), chunk)
                losses.append(float(l))
                acc.append(float(a))
        return dict(nll=float(np.mean(losses)), recall_at_one=float(np.mean(acc)))

    args.out.mkdir(parents=True, exist_ok=True)
    before = validate()
    print("initial", before, flush=True)
    history = []
    best = 1e9
    start = time.time()
    for step in range(1, args.steps + 1):
        picks = train[rng.integers(len(train), size=args.batch)]
        mode = modes[rng.choice(6, p=[0.20, 0.20, 0.19, 0.19, 0.16, 0.06])]
        a, b = sample(picks, mode)
        left, _ = objective(a, b, picks)
        right, _ = objective(b, a, picks)
        loss = (left + right) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        opt.param_groups[0]["lr"] = 0.003 * (
            0.25 + 0.75 * (1 + np.cos(step / args.steps * np.pi)) / 2
        )
        if step % 300 == 0 or step == args.steps:
            val_score = validate()
            val_score.update(step=step)
            history.append(val_score)
            print(val_score, flush=True)
            if val_score["nll"] < best:
                best = val_score["nll"]
                # Fold IDF into each token vector so inference needs only a lookup, sum and normalization.
                w = (model.emb.weight.detach() * model.idf[:, None]).numpy()
                scale = np.maximum(np.abs(w).max(1) / 127, 1e-12).astype("<f4")
                quant = np.clip(np.rint(w / scale[:, None]), -127, 127).astype("i1")
                (args.out / "recall.bin").write_bytes(scale.tobytes() + quant.tobytes())
                (args.out / "recall.json").write_text(
                    json.dumps(
                        dict(
                            format="franok-recall-i8-v1",
                            vocab=len(idf),
                            width=96,
                            maxTokens=160,
                            sha256=hashlib.sha256(
                                (args.out / "recall.bin").read_bytes()
                            ).hexdigest(),
                            validation=val_score,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            (args.out / "training.json").write_text(
                json.dumps(
                    dict(
                        initial=before,
                        history=history,
                        source_counts=json.loads(
                            (args.data / "receipt.json").read_text()
                        ),
                        elapsed_seconds=time.time() - start,
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("work/third/semantic"))
    p.add_argument("--out", type=Path, default=Path("work/v3/semantic"))
    p.add_argument("--tokenizer", type=Path, default=Path("web/tokenizer.json"))
    p.add_argument("--en", type=Path)
    p.add_argument("--tr", type=Path)
    p.add_argument("--init", type=Path)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument(
        "--translations",
        action="store_true",
        help="Add open English-Turkish Tatoeba training sentence pairs",
    )
    args = p.parse_args()
    if not (args.data / "receipt.json").exists():
        prepare(args)
    train(args)
