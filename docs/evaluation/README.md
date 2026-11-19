# Initial Colab checkpoint

This page records the original 256-token checkpoint. The [extended release evaluation](v2/README.md) covers the 1,024-token version and its training stages.

The first release consumed 56,524,800 token positions, including masked positions. Training used 53,859,151 next-token targets. The GPU was an NVIDIA L4. The first 3,200 steps took about 11 minutes and the remaining 6,000 steps took about 18 minutes. Dataset preparation, saving and transfers add wall time.

| Evaluation | Franok EN | Comparison EN | Franok TR | Comparison TR |
|---|---:|---:|---:|---:|
| Paired Belebele passages, 4 choices | 13/48 | 19/48 | 10/48 | 15/48 |
| Authored dialogue probes, 2 choices | 10/16 | 11/16 | 7/16 | 9/16 |

The comparison is [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) at revision a10cc1512eabd3dde888204e902eca88bddb4951. Both models used their own tokenizer and chat formatting. The [Belebele](https://huggingface.co/datasets/facebook/belebele) test is a reproducible subset. We hash-order aligned questions and take the first 48 for which the passage, prompt and every candidate fit the 256-position limit for both tokenizers in both languages. We choose the highest continuation log likelihood per candidate character. The authored probes score mean log likelihood per candidate token. Each JSON file includes individual scores. These small evaluations do not establish size-class superiority. The generation samples include incorrect factual answers.

Across 192 validation positions, the four-bit WASM output tracked the unquantized reference closely. Per-slice results and top-token agreement are in wasm-parity.json. On the development laptop, one 20-token Electron reply decoded at 154.7 tokens per second after prefill. This observation excludes startup, transfer and prompt processing. The session test reopens a saved conversation after restarting Electron. A separate test retrieves and attaches a turn from a 1,000-turn history.

Reproduce scores with `python -m tools.probe` and `python -m tools.reading`. Evaluation against the independent baseline requires transformers 4.57.6. Run `python -m tests.check_release && node tests/check_release.mjs` for full-size Q4 parity.

Later reports: [Franok 0.4 results](v4/final.md) and [Franok 0.5 results](v5/final.md).
