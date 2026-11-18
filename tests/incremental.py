"""Reference decoding parity across padding, anchor rotations, reset, and nonzero adapters."""

import torch
from franok.model import Config, Franok
from franok.incremental import Decoder
from franok.lora import attach


@torch.inference_mode()
def main():
    torch.manual_seed(2761)
    cfg = Config(
        vocab=256,
        width=48,
        layers=5,
        heads=6,
        kv_heads=2,
        hidden=64,
        banks=8,
        window=48,
    )
    model = Franok(cfg).eval()
    model.norm.weight.fill_(1.32)
    decoder = Decoder(model)
    for adapted in (False, True):
        if adapted:
            attach(model, 4, 8)
            for name, p in model.named_parameters():
                if ".lora_" in name and ".b.weight" in name:
                    p.normal_(std=0.03)
        for length in (1, 7, 8, 9, 15, 17, 31, 39):
            tokens = torch.randint(0, cfg.vocab, (1, 48))
            ids = tokens[0].tolist()
            expected = model(tokens)[0]
            actual = [decoder.prefill(ids[:length])] + [
                decoder.step(t) for t in ids[length:]
            ]
            torch.testing.assert_close(
                torch.stack(actual), expected[length - 1 :], rtol=1e-4, atol=3e-5
            )
            try:
                decoder.step(2)
            except ValueError:
                pass
            else:
                raise AssertionError("context overflow")
    print(
        "Matched full causal logits at every tested position with and without adapters."
    )


if __name__ == "__main__":
    main()
