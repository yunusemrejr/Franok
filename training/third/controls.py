"""Use the same passage question to teach distinct, system-directed answer formats."""

import numpy as np
from tokenizers import decoders
from .batches import Conversations
from .data import tokenizer

GUIDES = {
    "en": (
        (
            "Answer with the option letter only.",
            "Give only the letter of the correct choice.",
        ),
        (
            "Write only the text of the correct option.",
            "Answer with the chosen phrase and leave out its letter.",
        ),
        (
            "Give the letter, a closing parenthesis and the correct option text.",
            "Copy the entire correct option, including its letter and closing parenthesis.",
        ),
    ),
    "tr": (
        (
            "Yalnızca doğru seçeneğin harfini yaz.",
            "Cevap olarak sadece şıkkın harfini ver.",
        ),
        (
            "Yalnızca doğru seçeneğin metnini yaz.",
            "Cevap olarak seçtiğin ifadeyi yaz. Başına harf ekleme.",
        ),
        (
            "Doğru şıkkı, harfi ve kapanış paranteziyle birlikte aynen yaz.",
            "Seçenek harfini, kapanış parantezini ve doğru seçeneğin metnini yaz.",
        ),
    ),
}


class ControlledChoices:
    batch = Conversations.batch

    def __init__(self, conversations):
        self.sources = {l: conversations.records[l, "passages"] for l in GUIDES}
        self.tokenizer = tokenizer()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.indices = {}
        for lang, source in self.sources.items():
            a, m, e = source.index.T
            f, g = source.foil_index.T
            # Reserve ample room for the role, system format, both options and target.
            valid = (
                (e - m > 2)
                & (e - m <= 24)
                & (g - f > 1)
                & (g - f <= 24)
                & (m - a < 375)
            )
            self.indices[lang] = np.flatnonzero(valid)
            if not len(self.indices[lang]):
                raise ValueError("Missing passage choices in " + lang)

    def sample(self, rng, lang, upper, paired):
        if upper != 512:
            raise ValueError("System-choice batches use the 512-position bucket")
        source = self.sources[lang]
        index = int(rng.choice(self.indices[lang]))
        a, m, e = source.index[index]
        f, g = source.foil_index[index]
        prefix = source.data[a:m]
        user = int(np.flatnonzero(prefix == 3)[-1])
        choices = [
            self.tokenizer.decode(x.astype(int).tolist()).strip()
            for x in (source.data[m + 1 : e - 1], source.foils[f : g - 1])
        ]
        correct = int(rng.integers(2))
        if correct:
            choices.reverse()
        options = [chr(65 + j) + ") " + v for j, v in enumerate(choices)]
        style = int(rng.integers(3))
        system = GUIDES[lang][style][int(rng.integers(2))]

        def render(j, mode):
            return (chr(65 + j), choices[j], options[j])[mode]

        gold = render(correct, style)
        if rng.random() < 0.5:
            foil = render(1 - correct, style)
        else:
            foil = render(correct, (style + int(rng.integers(1, 3))) % 3)
        tokens = self.tokenizer.encode_batch(
            [system, "\n" + "\n".join(options), gold, foil], add_special_tokens=False
        )
        pre = [
            1,
            5 if lang == "en" else 6,
            *tokens[0].ids,
            2,
            *prefix[user:],
            *tokens[1].ids,
            4,
        ]
        yes = tokens[2].ids + [2]
        no = tokens[3].ids + [2]
        if len(pre) + max(len(yes), len(no)) - 1 > upper:
            raise ValueError("Controlled choice exceeded the input bucket.")
        return pre, yes, no if paired else None, "system-choice"
