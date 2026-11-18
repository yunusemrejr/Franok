"""Interactive CPU or GPU reference chat with your downloaded Franok files."""

import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, decoders
from .model import Config, Franok


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "directory",
        type=Path,
        help="Directory with config, tokenizer and model.safetensors",
    )
    p.add_argument("--lang", choices=["en", "tr"], default="en")
    p.add_argument("--max-new", type=int, default=60)
    p.add_argument(
        "--system", help="Session system prompt for the 1,024-position release"
    )
    p.add_argument("--temperature", type=float, default=0.58)
    p.add_argument("--top-p", type=float, default=0.89)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    if not 0 <= args.temperature <= 1.5 or not 0.15 <= args.top_p <= 1:
        p.error("Temperature must be 0–1.5 and top-p 0.15–1")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(str(args.directory / "tokenizer.json"))
    tok.decoder = decoders.ByteLevel()
    cfg = Config(**json.loads((args.directory / "config.json").read_text()))
    m = Franok(cfg).eval().to(device)
    m.load_state_dict(load_file(str(args.directory / "model.safetensors")))
    limit = max(1, min(cfg.window // 2, args.max_new))
    history = []
    prefix = [1, 5 if args.lang == "en" else 6]
    if cfg.window >= 1024:
        guide = args.system or (
            "You are Franok. Reply in English. Answer the latest message directly and use earlier turns when relevant."
            if args.lang == "en"
            else "Sen Franok'sun. Türkçe yanıt ver. Son mesaja doğrudan cevap ver ve gerektiğinde önceki konuşmayı kullan."
        )
        prefix += tok.encode(guide).ids[:180] + [2]
    while True:
        try:
            line = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        prompt = [
            3,
            *tok.encode(line).ids[-(cfg.window - limit - len(prefix) - 2) :],
            4,
        ]
        room = cfg.window - limit - len(prompt) - len(prefix)
        context = [*prefix, *(history[-room:] if room > 0 else []), *prompt]
        output = []
        for _ in range(limit):
            logits = m(torch.tensor([context + output], device=device))[0, -1]
            logits[:2] = -torch.inf
            logits[3:7] = -torch.inf
            if output:
                logits[list(set(output[-32:]))] -= 0.55
            if len(output) >= 3:
                for j in range(3, len(output)):
                    if output[j - 3 : j] == output[-3:]:
                        logits[output[j]] = -torch.inf
            if args.temperature == 0:
                token = logits.argmax().item()
            else:
                values, indices = logits.topk(48)
                probs = (values / args.temperature).softmax(0)
                probs[(probs.cumsum(0) - probs) >= args.top_p] = 0
                token = indices[torch.multinomial(probs, 1)].item()
            if token == 2:
                break
            output.append(token)
        print("Franok ›", tok.decode(output).strip())
        history = (history + prompt + output + [2])[-cfg.window :]


if __name__ == "__main__":
    main()
