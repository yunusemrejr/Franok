"""Run the same conversation script on an unquantized Franok checkpoint or its peer."""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer, decoders
from safetensors.torch import load_file
from franok.model import Config, Franok
from franok.incremental import Decoder
from training.extended.dialogue import DEFAULT
from .reading import PEERS


def normal(text):
    s = unicodedata.normalize("NFKC", text).replace("I", "ı").replace("İ", "i").lower()
    return re.sub(r"[^\w:.]+", " ", s).strip().rstrip(".")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path)
    p.add_argument("--peer", nargs="?", const="smollm2", choices=PEERS)
    p.add_argument("--cases", type=Path, default=Path("tests/conversations.json"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.peer:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo, revision = PEERS[args.peer]
        tk = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, local_files_only=True
        ).eval()
        stop = tk.eos_token_id
        invalid = set(tk.all_special_ids) - {stop}

        def prefix(turns, lang, system):
            return tk.apply_chat_template(
                [{"role": "system", "content": system}, *turns],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )

        def decode(x):
            return tk.decode(x, skip_special_tokens=True)

        def step(ids, prev):
            o = model(torch.tensor([ids]), past_key_values=prev, use_cache=True)
            return o.logits[0, -1].float().numpy().copy(), o.past_key_values

    else:
        cfg = Config(**json.loads((args.run / "config.json").read_text()))
        model = Franok(cfg).eval()
        weights = args.run / "model.safetensors"
        tensors = load_file(str(weights))
        storage = sorted(
            {str(t.dtype).removeprefix("torch.") for t in tensors.values()}
        )
        model.load_state_dict(tensors)
        del tensors
        with weights.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        tk = Tokenizer.from_file("work/data/tokenizer.json")
        tk.decoder = decoders.ByteLevel()
        stop = 2
        invalid = {0, 1, 3, 4, 5, 6}

        def prefix(turns, lang, system):
            ids = [1, 5 if lang == "en" else 6]
            if cfg.window >= 1024:
                ids.extend(tk.encode(system).ids + [2])
            for t in turns:
                ids.extend(
                    ([3] if t["role"] == "user" else [4])
                    + tk.encode(t["content"]).ids
                    + ([] if t["role"] == "user" else [2])
                )
            ids.append(4)
            return (
                ids
                if len(ids) < cfg.window - 64
                else ids[:2] + ids[-(cfg.window - 66) :]
            )

        def decode(x):
            return tk.decode(x)

        def step(ids, prev):
            if prev is None:
                prev = Decoder(model)
                next_logits = prev.prefill(ids)
            else:
                next_logits = prev.step(ids[0])
            return next_logits.float().numpy().copy(), prev

    out = []
    for sc in json.loads(args.cases.read_text()):
        turns = []
        checks = []
        seed = 17941
        lang = sc["lang"]
        temp = sc.get("temperature", 0.58)
        for j, q in enumerate(sc["turns"]):
            turns.append(dict(role="user", content=q))
            guide = DEFAULT[lang]
            if args.peer:
                name = "SmolLM" if args.peer == "smollm2" else "LFM2"
                guide = guide.replace("Franok'sun", f"{name}'sin").replace("Franok", name)
            ids = prefix(turns, lang, sc.get("system", guide))
            cache = None
            generated = []
            for _ in range(sc.get("maxNew", 64)):
                logits, cache = step(ids, cache)
                logits[list(invalid)] = -np.inf
                if generated:
                    logits[list(set(generated[-32:]))] -= 0.55
                if len(generated) >= 3:
                    for k in range(3, len(generated)):
                        if generated[k - 3 : k] == generated[-3:]:
                            logits[generated[k]] = -np.inf
                if temp == 0:
                    token = int(np.argmax(logits))
                else:
                    best = np.argpartition(-logits, 48)[:48]
                    best = best[np.argsort(-logits[best])]
                    prob = np.exp(
                        (logits[best] - logits[best[0]]) / temp, dtype="float64"
                    )
                    total = prob.sum()
                    n = 1 + np.searchsorted(prob.cumsum(), 0.89 * total)
                    prob = prob[:n]
                    seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
                    token = int(
                        best[
                            np.searchsorted(
                                prob.cumsum(), (seed / 4294967296) * prob.sum()
                            )
                        ]
                    )
                if token == stop:
                    break
                generated.append(token)
                ids = [token]
            answer = decode(generated).strip()
            turns.append(dict(role="assistant", content=answer))
            criteria = sc["expect"][j]
            if args.peer and criteria == ["franok"]:
                criteria = ["smollm", "smol lm"] if args.peer == "smollm2" else ["lfm2", "lfm 2"]
            passed = (
                None
                if criteria is None
                else any(
                    (
                        answer == c
                        if sc.get("verbatim")
                        else normal(answer) == normal(c)
                        if sc.get("exact")
                        else normal(c) in normal(answer)
                    )
                    for c in criteria
                )
            )
            checks.append(
                dict(criterion=criteria, passed=passed, tokens=len(generated))
            )
        out.append(dict(id=sc["case"], lang=lang, turns=turns, checks=checks))
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(
        json.dumps(
            dict(
                model=repo.split("/")[-1] if args.peer else "Franok",
                **(
                    {"revision": revision}
                    if args.peer
                    else {"weights_sha256": digest, "storage": storage}
                ),
                conversations=out,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
