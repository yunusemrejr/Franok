"""Four-way retrieval of an early fact under controlled unrelated dialogue at three depths."""

import argparse, json
from pathlib import Path
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from franok.model import Franok, Config
from training.extended.dialogue import DEFAULT
from .reading import BASE

FILLERS = {
    "en": [
        ("I put the spare chairs away.", "Thanks for the update."),
        ("Can you suggest a quiet activity?", "You could read a short story."),
        ("The weather is clear today.", "That sounds like a good day for a walk."),
        ("I will tidy the desk after lunch.", "That sounds like a plan."),
    ],
    "tr": [
        ("Yedek sandalyeleri kaldırdım.", "Haber verdiğin için teşekkürler."),
        ("Sessiz bir etkinlik önerebilir misin?", "Kısa bir öykü okuyabilirsin."),
        ("Bugün hava açık.", "Yürüyüş için güzel bir gün."),
        ("Öğle yemeğinden sonra masayı toplayacağım.", "Güzel bir plan."),
    ],
}
WORDS = {
    "en": ["cedar", "maple", "willow", "birch"],
    "tr": ["sedir", "kavak", "söğüt", "çam"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peer", action="store_true")
    p.add_argument("--run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ft = Tokenizer.from_file("work/data/tokenizer.json")

    def ids_for(turns, lang):
        ids = [1, 5 if lang == "en" else 6, *ft.encode(DEFAULT[lang]).ids, 2]
        for role, text in turns:
            ids += [3 if role == "user" else 4, *ft.encode(text).ids] + (
                [2] if role == "assistant" else []
            )
        return ids + [4]

    if args.peer:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = "HuggingFaceTB/SmolLM2-360M-Instruct"
        tk = AutoTokenizer.from_pretrained(repo, revision=BASE, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=BASE, local_files_only=True
        ).eval()
        encode = lambda s: tk.encode(s, add_special_tokens=False)
    else:
        cfg = Config(**json.loads((args.run / "config.json").read_text()))
        model = Franok(cfg).eval()
        model.load_state_dict(load_file(str(args.run / "model.safetensors")))
        encode = lambda s: ft.encode(s).ids
    rows = []
    with torch.inference_mode():
        for lang in ("en", "tr"):
            pool = WORDS[lang]
            for depth in (128, 384, 960):
                for j in range(8):
                    label = ["K17", "J42", "B31", "T54", "N28", "M60", "R58", "V74"][j]
                    word = pool[j % 4]
                    fact = (
                        f"The word assigned to folder {label} is {word}. Please keep it for later."
                        if lang == "en"
                        else f"{label} klasörüne verilen kelime {word}. Bunu daha sonrası için hatırla."
                    )
                    q = (
                        f"Which word did I assign to folder {label}? Give only the word."
                        if lang == "en"
                        else f"{label} klasörüne hangi kelimeyi vermiştim? Sadece kelimeyi yaz."
                    )
                    turns = [
                        ("user", fact),
                        ("assistant", "All right." if lang == "en" else "Tamam."),
                    ]
                    n = 0
                    while True:
                        a, b = FILLERS[lang][(j + n) % 4]
                        pair = [("user", a), ("assistant", b)]
                        if len(ids_for(turns + pair + [("user", q)], lang)) > depth:
                            break
                        turns.extend(pair)
                        n += 1
                    turns.append(("user", q))
                    head = ids_for(turns, lang)
                    if args.peer:
                        messages = [
                            dict(
                                role="system",
                                content="Answer in English."
                                if lang == "en"
                                else "Türkçe cevap ver.",
                            )
                        ] + [dict(role=r, content=t) for r, t in turns]
                        head = tk.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=True
                        )
                    tails = [encode(w) for w in pool]
                    seq = [head + t[:-1] for t in tails]
                    w = max(map(len, seq))
                    batch = torch.tensor([s + [0] * (w - len(s)) for s in seq])
                    logits = (
                        model(batch, use_cache=False).logits
                        if args.peer
                        else model(batch)
                    )
                    vals = []
                    for branch, tail in enumerate(tails):
                        lp = (
                            logits[branch, len(head) - 1 : len(head) - 1 + len(tail)]
                            .float()
                            .log_softmax(-1)
                        )
                        vals.append(
                            lp.gather(1, torch.tensor(tail)[:, None]).mean().item()
                        )
                    pred = pool[max(range(4), key=vals.__getitem__)]
                    rows.append(
                        dict(
                            lang=lang,
                            budget=depth,
                            prompt_tokens=len(head),
                            intervening_turns=n * 2,
                            label=label,
                            word=word,
                            pred=pred,
                            correct=pred == word,
                        )
                    )
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
