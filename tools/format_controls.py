"""Change only the system instruction on held-out, bilingual two-choice questions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from training.third.batches import Conversations
from training.third.controls import ControlledChoices


INSTRUCTIONS = {
    "en": (
        "Return a single capital letter indicating your answer.",
        "Respond with the words from your selected answer. Omit its label.",
    ),
    "tr": (
        "Cevabını tek bir büyük harfle belirt.",
        "Seçtiğin yanıtın metnini aynen ver. Şıkkın harfini ekleme.",
    ),
}


def cases(count=24):
    corpus = ControlledChoices(Conversations("val"))
    rng = np.random.default_rng(81104)
    output = []
    for lang, prompts in INSTRUCTIONS.items():
        seen = set()
        records = corpus.sources[lang]
        for index in rng.permutation(corpus.indices[lang]):
            a, m, e = records.index[index]
            f, g = records.foil_index[index]
            prefix = records.data[a:m]
            context = prefix[np.flatnonzero(prefix == 3)[-1] + 1 :]
            user, correct, foil = (
                corpus.tokenizer.decode(v.astype(int).tolist()).strip()
                for v in (
                    context,
                    records.data[m + 1 : e - 1],
                    records.foils[f : g - 1],
                )
            )
            if (
                not correct
                or not foil
                or correct.casefold() == foil.casefold()
                or user in seen
            ):
                continue
            second = len(seen) % 2
            options = (foil, correct) if second else (correct, foil)
            question = f"{user}\nA) {options[0]}\nB) {options[1]}"
            key = hashlib.sha256(question.encode()).hexdigest()[:16]
            for mode, instruction, target in zip(
                ("letter", "words"), prompts, ("AB"[second], correct)
            ):
                output.append(
                    dict(
                        case=f"{lang}-{key}-{mode}",
                        lang=lang,
                        system=instruction,
                        turns=[question],
                        temperature=0,
                        maxNew=32,
                        expect=[[target]],
                        exact=True,
                        verbatim=True,
                    )
                )
            seen.add(user)
            if len(seen) == count:
                break
        if len(seen) != count:
            raise ValueError("Too few distinct held-out questions for " + lang)
    return output


def summarize(report):
    grouped = {}
    for row in report["conversations"]:
        lang, key, mode = row["id"].split("-")
        pair = grouped.setdefault(lang, {}).setdefault(key, {})
        if mode in pair:
            raise ValueError("Conversations contain a duplicate format")
        pair[mode] = bool(row["checks"][0]["passed"])
    scores = {}
    for lang, questions in grouped.items():
        if any(set(pair) != {"letter", "words"} for pair in questions.values()):
            raise ValueError("Each question needs both forms")
        scores[lang] = dict(
            questions=len(questions),
            letter=sum(pair["letter"] for pair in questions.values()),
            words=sum(pair["words"] for pair in questions.values()),
            both=sum(all(pair.values()) for pair in questions.values()),
        )
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.score.read_text())) if args.score else cases()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
