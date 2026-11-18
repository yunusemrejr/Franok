"""Validate extractable UTF-8 boundaries, question conditioning, padding and abstention."""

import torch
from franok.span import SpanReadout, labels, pack
from franok.span import boundaries, pieces
from tokenizers import Tokenizer


def main():
    torch.manual_seed(1308)
    tokenizer = Tokenizer.from_file("web/tokenizer.json")
    text = "IŞIK numarası 73. 😊 Aşı önemlidir."
    tok = tokenizer.encode(text).ids
    raw, ends, legal = boundaries(text, tok, pieces(tokenizer))
    assert 0 in legal and raw[int(ends[-1]) :] == b""
    head = SpanReadout(16, 12, 9)
    prefix = [1, 6, 3, 8, 9, 2, 3]
    rows = [
        (prefix + tok + [4], len(prefix), legal, -1, -1),
        (prefix + [8, 9, 10, 4], len(prefix), [1, 1, 1, 1], 1, 2),
    ]
    ids, mask, meta = pack(rows)
    states = torch.randn(2, ids.shape[1], 16, requires_grad=True)
    scores = head(states, meta[:, 0], meta[:, 1], mask)
    gold = labels(meta, scores.shape[1], head.max_span)
    assert gold[0] == scores.shape[-1] - 1 and gold[1] == (len(prefix) + 2) * 9 + 1
    assert scores[1, gold[1]] > -1e8
    for j in range(len(tok)):
        for delta in range(9):
            valid = j >= delta and legal[j - delta] and legal[j + 1]
            assert (scores[0, (len(prefix) + j) * 9 + delta].item() > -1e8) == bool(
                valid
            )
    longer = torch.cat([states, torch.randn(2, 9, 16)], dim=1)
    padded = head(longer, meta[:, 0], meta[:, 1], torch.nn.functional.pad(mask, (0, 9)))
    torch.testing.assert_close(padded[:, : scores.shape[-1] - 1], scores[:, :-1])
    torch.testing.assert_close(padded[:, -1], scores[:, -1])
    torch.nn.functional.cross_entropy(scores, gold).backward()
    assert torch.isfinite(states.grad).all() and states.grad[:, :4].abs().sum() > 0
    print("Unicode spans, row padding, training labels and null score passed.")


if __name__ == "__main__":
    main()
