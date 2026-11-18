"""Assistant targets and routing must be independent for two conversations in one batch."""

import numpy as np
import torch
from training.third.batches import Conversations
from franok.model import Franok, Config


class ExampleBatch:
    batch = Conversations.batch

    @staticmethod
    def sample(rng, lang, upper, paired):
        if lang == "en":
            pre, good, bad = [1, 5, 3, 12, 4, 21, 2, 3, 40, 4], [19, 22, 2], [99, 2]
        else:
            pre, good, bad = [1, 6, 3, 71, 4], [27, 2], [101, 31, 2]
        return (
            np.array(pre),
            np.array(good),
            np.array(bad) if paired else None,
            "example",
        )


@torch.inference_mode()
def main():
    torch.manual_seed(82)
    m = Franok(
        Config(
            vocab=256,
            width=32,
            heads=4,
            kv_heads=2,
            hidden=64,
            banks=8,
            layers=2,
            window=32,
        )
    ).eval()
    rng = np.random.default_rng(71)
    data = ExampleBatch()
    x, y, _ = data.batch(rng, 16, 64, paired=True)
    for i, (first, good, bad) in enumerate(
        [(9, [19, 22, 2], [99, 2]), (4, [27, 2], [101, 31, 2])]
    ):
        for row, answer in ((i, good), (i + 2, bad)):
            assert x[row, first] == 4 and np.all(y[row, :first] == -100)
            assert list(y[row, first : first + len(answer)]) == answer and np.all(
                y[row, first + len(answer) :] == -100
            )
        assert np.array_equal(x[i, : first + 1], x[i + 2, : first + 1])
    ref = m(torch.tensor(x))[0]
    x[1:] = 143
    torch.testing.assert_close(ref, m(torch.tensor(x))[0], atol=0, rtol=0)
    print(
        "Complete prompts, assistant-only targets, choice pairs and row isolation passed."
    )


if __name__ == "__main__":
    main()
