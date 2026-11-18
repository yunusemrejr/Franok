"""Compare batched, independently routed cached samples with full causal passes."""

import torch
from franok.lora import attach
from franok.model import Config, Franok
from franok.rollout import GroupDecoder


@torch.no_grad()
def main():
    torch.manual_seed(47193)
    torch.set_num_threads(2)
    model = Franok(
        Config(
            vocab=256,
            width=48,
            heads=6,
            kv_heads=2,
            layers=3,
            banks=8,
            hidden=128,
            window=128,
        )
    ).eval()
    attach(model, 4, 8)
    for block in model.blocks:
        for name in ("qkv", "out", "up", "down"):
            getattr(block, "lora_" + name).b.weight.normal_(0, 0.03)
    gen = torch.Generator().manual_seed(4417)
    checks = 0
    for prefix in (1, 7, 8, 9, 16, 19, 30, 97):
        ids = torch.randint(7, 256, (1, prefix), generator=gen).expand(6, -1)
        decoder = GroupDecoder(model, ids[0], len(ids))
        for _ in range(19):
            cached = decoder.logits()
            full = model(ids)[:, -1]
            torch.testing.assert_close(cached, full, rtol=2e-5, atol=3e-5)
            v = torch.randint(7, 256, (6,), generator=gen)
            decoder.advance(v)
            ids = torch.cat((ids, v[:, None]), 1)
            checks += len(ids)
    print(
        f"Matched {checks} continuation logits across diverging anchors and nonzero LoRAs."
    )


if __name__ == "__main__":
    main()
