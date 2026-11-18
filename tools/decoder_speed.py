"""Compare single-request, warm CPU inference on identical bilingual text."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from franok.model import Config, Franok
from franok.incremental import Decoder
from tools.reading import BASE

TEXT = {
    "en": (
        "We used the public library for our evening meeting. Deniz brought a notebook and Ayla wrote down what we agreed. ",
        "We will meet again on Friday. Everyone can bring a book to discuss.",
    ),
    "tr": (
        "Akşam toplantımızı mahalle kütüphanesinde yaptık. Deniz bir defter getirdi ve Ayla aldığımız kararları yazdı. ",
        "Cuma günü yeniden buluşacağız. Herkes tartışmak için bir kitap getirebilir.",
    ),
}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path)
    g.add_argument("--peer", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.peer:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = "HuggingFaceTB/SmolLM2-360M-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=BASE, dtype=torch.float32, local_files_only=True
        ).eval()
        tok = AutoTokenizer.from_pretrained(repo, revision=BASE, local_files_only=True)
        ident = BASE

        def encode(s):
            return tok.encode(s, add_special_tokens=False)

        def pre(ids):
            return model(
                torch.tensor([ids]), use_cache=True, logits_to_keep=1
            ).past_key_values

        def step(t, past):
            return model(
                torch.tensor([[t]]),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            ).past_key_values
    else:
        cfg = Config(**json.loads((a.run / "config.json").read_text()))
        file = a.run / "model.safetensors"
        with file.open("rb") as f:
            ident = hashlib.file_digest(f, "sha256").hexdigest()
        model = Franok(cfg).eval()
        model.load_state_dict(load_file(str(file)))
        dec = Decoder(model)
        tok = Tokenizer.from_file("web/tokenizer.json")

        def encode(s):
            return tok.encode(s).ids

        def pre(ids):
            dec.prefill(ids)

        def step(t, _):
            dec.step(t)

    rows = []
    for lang, (base, cont) in TEXT.items():
        tail = encode(cont)
        for repeats in (2, 12):
            text = (base * repeats).strip()
            ids = encode(text)
            assert len(ids) + len(tail) <= 1024
            small = ids[: min(16, len(ids))]
            # Warm both the matrix prefill and the single-token path.
            seed = pre(small)
            step(tail[0], seed)
            pf, decoding = [], []
            for _ in range(5):
                begin = perf_counter()
                cache = pre(ids)
                split = perf_counter()
                for t in tail:
                    cache = step(t, cache)
                pf.append(split - begin)
                decoding.append(perf_counter() - split)
            rows.append(
                dict(
                    lang=lang,
                    repeats=repeats,
                    prompt_tokens=len(ids),
                    prompt_bytes=len(text.encode()),
                    continuation_tokens=len(tail),
                    continuation_bytes=len(cont.encode()),
                    prefill_seconds=pf,
                    decode_seconds=decoding,
                    median_prefill_ms=1000 * median(pf),
                    median_decode_tokens_per_second=len(tail) / median(decoding),
                    median_decode_bytes_per_second=len(cont.encode())
                    / median(decoding),
                )
            )
    cpu = next(
        (
            l.split(":", 1)[1].strip()
            for l in Path("/proc/cpuinfo").read_text().splitlines()
            if l.startswith("model name")
        ),
        platform.machine(),
    )
    data = dict(
        model="SmolLM2-360M-Instruct" if a.peer else "Franok",
        weights=ident,
        cpu=cpu,
        threads=a.threads,
        dtype="float32",
        torch=torch.__version__,
        cases=rows,
    )
    a.out.parent.mkdir(exist_ok=True, parents=True)
    a.out.write_text(json.dumps(data, indent=2) + "\n")
    print(
        [
            (r["lang"], r["repeats"], round(r["median_decode_bytes_per_second"]))
            for r in rows
        ]
    )


if __name__ == "__main__":
    main()
