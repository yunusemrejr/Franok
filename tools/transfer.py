"""Evaluate aligned English/Turkish passage reading and free answer extraction."""

import argparse
from collections import Counter
import hashlib
import json
import re
import string
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer, decoders

from franok.incremental import Decoder
from franok.model import Franok, Config
from .reading import BASE, REV

XQUAD = "51adfef1c1287aab1d2d91b5bead9bcfb9c68583"
SYSTEM = {
    "en": "Read the passage and answer the question in English.",
    "tr": "Metni oku ve soruyu Türkçe yanıtla.",
}
EXTRACT = {
    "en": "Use the shortest phrase from the passage that answers the question.",
    "tr": "Soruyu yanıtlayan en kısa ifadeyi metinden aynen yaz.",
}


def cases(suite, ids):
    for lang, dialect in (("en", "eng_Latn"), ("tr", "tur_Latn")):
        if suite == "belebele":
            file = hf_hub_download(
                "facebook/belebele",
                "data/" + dialect + ".jsonl",
                revision=REV,
                repo_type="dataset",
            )
            data = {
                (r["link"], r["question_number"]): r
                for r in map(json.loads, Path(file).read_text().splitlines())
            }
            for key in ids:
                row = data[tuple(key)]
                yield dict(
                    lang=lang,
                    key=key,
                    prompt=row["flores_passage"] + "\n\n" + row["question"],
                    choices=[row[f"mc_answer{j}"] for j in range(1, 5)],
                    gold=int(row["correct_answer_num"]) - 1,
                )
        else:
            file = hf_hub_download(
                "google/xquad",
                f"xquad.{lang}/validation-00000-of-00001.parquet",
                revision=XQUAD,
                repo_type="dataset",
            )
            data = {r["id"]: r for r in pq.read_table(file).to_pylist()}
            for key in ids:
                row = data[key]
                yield dict(
                    lang=lang,
                    key=key,
                    passage=row["context"],
                    question=row["question"],
                    prompt=row["context"]
                    + "\n\n"
                    + row["question"]
                    + "\n"
                    + EXTRACT[lang],
                    answers=row["answers"]["text"],
                )


def words(text, lang):
    text = unicodedata.normalize("NFKC", text)
    if lang == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.lower()
    text = "".join(
        c
        for c in text
        if not unicodedata.category(c).startswith("P") and c not in string.punctuation
    )
    if lang == "en":
        text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def grade(text, refs, lang):
    pred = words(text, lang)
    exact = 0
    f1 = 0
    for ref in refs:
        gold = words(ref, lang)
        overlap = sum((Counter(pred) & Counter(gold)).values())
        exact = max(exact, int(pred == gold))
        f1 = max(f1, 2 * overlap / max(1, len(pred) + len(gold)))
    return dict(exact=exact, f1=f1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("belebele", "xquad"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path)
    group.add_argument("--peer", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    ids = json.loads(Path("tests/transfer_ids.json").read_text())[args.suite]
    if args.peer:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        repo = "HuggingFaceTB/SmolLM2-360M-Instruct"
        tok = AutoTokenizer.from_pretrained(repo, revision=BASE, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=BASE, local_files_only=True, dtype=torch.float32
        ).eval()
        identity = BASE

        def prefix(item):
            turns = [
                {"role": "system", "content": SYSTEM[item["lang"]]},
                {"role": "user", "content": item["prompt"]},
            ]
            return tok.apply_chat_template(
                turns, tokenize=True, add_generation_prompt=True
            )

        def encode(text):
            return tok.encode(text, add_special_tokens=False)

        def generate(head):
            tokens = []
            first = model(torch.tensor([head]), use_cache=True, logits_to_keep=1)
            cache, logits = first.past_key_values, first.logits[0, -1]
            for _ in range(24):
                token = int(logits.argmax())
                if token == tok.eos_token_id:
                    break
                tokens.append(token)
                nxt = model(
                    torch.tensor([[token]]), past_key_values=cache, use_cache=True
                )
                cache, logits = nxt.past_key_values, nxt.logits[0, -1]
            return tok.decode(tokens, skip_special_tokens=True).strip()

        def option_scores(head, tails):
            first = model(torch.tensor([head]), use_cache=True, logits_to_keep=1)
            cache = first.past_key_values
            cache.batch_repeat_interleave(len(tails))
            hs = first.logits[:, -1:].expand(len(tails), 1, -1)
            more = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(t[:-1], dtype=torch.long) for t in tails],
                batch_first=True,
            )
            if more.shape[1]:
                more_out = model(more, past_key_values=cache, use_cache=True).logits
                return torch.cat([hs, more_out], dim=1)
            return hs

    else:
        cfg = Config(**json.loads((args.run / "config.json").read_text()))
        model = Franok(cfg).eval()
        file = args.run / "model.safetensors"
        model.load_state_dict(load_file(str(file)))
        with file.open("rb") as f:
            identity = hashlib.file_digest(f, "sha256").hexdigest()
        tok = Tokenizer.from_file("work/data/tokenizer.json")
        tok.decoder = decoders.ByteLevel()
        decoder = Decoder(model)

        def encode(text):
            return tok.encode(text).ids

        def prefix(item):
            lang = item["lang"]
            return [
                1,
                5 if lang == "en" else 6,
                *encode(SYSTEM[lang]),
                2,
                3,
                *encode(item["prompt"]),
                4,
            ]

        def generate(head):
            tokens = []
            logits = decoder.prefill(head)
            for _ in range(24):
                logits[0:2] = -torch.inf
                logits[3:7] = -torch.inf
                token = int(logits.argmax())
                if token == 2:
                    break
                tokens.append(token)
                logits = decoder.step(token)
            return tok.decode(tokens).strip()

        def option_scores(head, tails):
            batch = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(head + t[:-1]) for t in tails], batch_first=True
            )
            return model(batch)[:, len(head) - 1 :]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.with_suffix(".jsonl")
    rows = []
    if log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    for i, item in enumerate(cases(args.suite, ids)):
        key, lang = item["key"], item["lang"]
        if i < len(rows):
            saved = rows[i]
            if (saved["key"], saved["lang"], saved["weights"]) != (key, lang, identity):
                raise ValueError("Partial results refer to a different model or sample")
            continue
        head = prefix(item)
        result = dict(weights=identity, lang=lang, key=key, prompt_tokens=len(head))
        if args.suite == "belebele":
            tails = [encode(choice) for choice in item["choices"]]
            logits = option_scores(head, tails).float().log_softmax(-1)
            scores = [
                float(logits[j, torch.arange(len(t)), torch.tensor(t)].sum())
                / len(item["choices"][j])
                for j, t in enumerate(tails)
            ]
            prediction = int(np.argmax(scores))
            result.update(
                gold=item["gold"],
                prediction=prediction,
                correct=prediction == item["gold"],
                scores=scores,
            )
        else:
            text = generate(head)
            result.update(answer=text, **grade(text, item["answers"], lang))
        rows.append(result)
        with log.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if (i + 1) % 20 == 0:
            print(args.suite, i + 1, flush=True)
    summary = {}
    for lang in ("en", "tr"):
        subset = [r for r in rows if r["lang"] == lang]
        metrics = ("correct",) if args.suite == "belebele" else ("exact", "f1")
        summary[lang] = {
            "count": len(subset),
            **{k: float(np.mean([r[k] for r in subset])) for k in metrics},
        }
    result = dict(
        suite=args.suite,
        dataset_revision=REV if args.suite == "belebele" else XQUAD,
        weights=identity,
        summary=summary,
        items=rows,
    )
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
