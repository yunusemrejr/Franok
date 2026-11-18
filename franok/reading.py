"""Quote a response directly from UTF-8 source text using the optional trained span readout."""

import numpy as np
import torch
from .span import pack, boundaries, pieces


class PassageReader:
    def __init__(self, model, head, tokenizer, threshold):
        self.model, self.head, self.tokenizer = model, head, tokenizer
        self.table = pieces(tokenizer)
        self.threshold = threshold

    @torch.inference_mode()
    def read(self, question, passage, lang):
        q = self.tokenizer.encode(question.strip()).ids
        ids = self.tokenizer.encode(passage).ids
        if not q or not ids:
            raise ValueError("Supply a question and a passage.")
        prefix = [1, 5 if lang == "en" else 6, 3, *q, 2, 3]
        capacity = self.model.c.window - len(prefix) - 1
        if capacity < self.head.max_span + 8:
            raise ValueError("Question leaves too little room for a passage.")
        raw, ends, legal = boundaries(passage, ids, self.table)
        byte_offsets = np.r_[0, ends]
        best = None
        start = windows = 0
        while start < len(ids):
            stop = min(len(ids), start + capacity)
            while stop > start and not legal[stop]:
                stop -= 1
            row = (
                prefix + ids[start:stop] + [4],
                len(prefix),
                legal[start : stop + 1],
                -1,
                -1,
            )
            batch, flags, meta = pack([row])
            # A supported window is divisible by the bank count (as in the released configs).
            device = self.model.embed.weight.device
            h = self.model.hidden(batch.to(device))
            logits = self.head(
                h, meta[:, 0].to(device), meta[:, 1].to(device), flags.to(device)
            )[0]
            index = int(logits[:-1].argmax())
            margin = float(logits[index] - logits[-1])
            relative_end, delta = divmod(index, self.head.max_span)
            j = start + relative_end - len(prefix)
            i = j - delta
            lo, hi = int(byte_offsets[i]), int(byte_offsets[j + 1])
            quote = raw[lo:hi].decode()
            # Strip only outside whitespace and retain offsets into the unmodified source bytes.
            leading = len(quote.encode()) - len(quote.lstrip().encode())
            lo += leading
            hi = lo + len(quote.strip().encode())
            if hi > lo and (best is None or margin > best["margin"]):
                best = dict(
                    quote=raw[lo:hi].decode(), start_byte=lo, end_byte=hi, margin=margin
                )
            windows += 1
            if stop == len(ids):
                break
            start = stop - self.head.max_span
            while not legal[start]:
                start -= 1
        accepted = bool(best and best["margin"] >= self.threshold)
        return dict(
            answer=best["quote"] if accepted else "",
            accepted=accepted,
            windows=windows,
            **(best or dict(quote="", start_byte=None, end_byte=None, margin=None))
        )
