---
license: mit
language:
- en
- tr
pipeline_tag: text-generation
tags:
- from-scratch
- custom-architecture
- webassembly
datasets:
- wikimedia/wikipedia
- OpenAssistant/oasst1
- databricks/databricks-dolly-15k
- merve/turkish_instructions
- GoktugD/turkish-extractive-qa-1.5m
---

# Franok

Franok is an experimental English–Turkish language model by Yunus Emre Vurgun. The 314,061,312 parameters were initialized randomly. Every layer dispatches each token to one of eight independent SwiGLU banks using a causal anchor from the preceding group. About 49.82M parameters participate per token including the tied output head. It has a 256-token neural window.

[Chat locally in the browser](https://franok.yunusemrevurgun.com/) · [Public source](https://github.com/yunusemrejr/Franok) · [Colab notebook](https://colab.research.google.com/github/yunusemrejr/Franok/blob/main/training/Franok.ipynb)

## Weights and inference

| File | Purpose |
|---|---|
| `model.safetensors` | FP16 tensors for the Franok PyTorch reference |
| `config.json`, `tokenizer.json` | Architecture dimensions and from-scratch 8,192-token byte BPE |
| `model.q4`, `model.json` | Four-bit SIMD decoder weights, byte offsets, dimensions and SHA-256 |

Use the included implementation for this custom architecture. Download the three PyTorch files into a directory, install the requirements from the source repository and run `python -m franok.chat DIRECTORY --lang en`. Use `--lang tr` for Turkish. The browser and Electron app share the same incremental WASM decoder. In both interfaces, users can recover a saved turn and attach it to a new prompt. Attached context and persistent sessions are application features.

## Training and capabilities

A single NVIDIA L4 on Google Colab ran 3,200 initial training steps, 4,800 continuation steps and 1,200 assistant-supervision steps. Each step had 24 sequences of length 256. Across all phases 53,859,151 next-token targets contributed to loss. The prepared data contains English and Turkish Wikipedia, instructions and question-answer material. A hash split reserves validation groups before tokenizer fitting. See the pinned [data sources](https://github.com/yunusemrejr/Franok/blob/main/docs/data.md) and [math](https://github.com/yunusemrejr/Franok/blob/main/docs/architecture.md).

Validation assistant-token negative log likelihood fell from 4.183 at the end of the initial phase to 3.546 after refinement. This is a short training run. Generated answers can be repetitive, factually wrong, off-topic or mixed-language. A language selector conditions the model; fluency and factual accuracy vary. It is unsuitable for high-stakes advice.

MIT covers Franok's original code and weights. Source corpora have their own licenses. The comparison model, where reported, is used only for evaluation.

## Comparative evaluation

On 48 matched Belebele reading questions in each language, Franok scored 13/48 in English and 10/48 in Turkish. SmolLM2-360M-Instruct scored 19/48 and 15/48 under the same protocol. In 32 short two-choice probes the totals were 17/32 and 20/32 respectively. The peer model led on both tests. Methods, item scores and unedited outputs are in the [evaluation report](https://github.com/yunusemrejr/Franok/tree/main/docs/evaluation).
