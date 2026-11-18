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
- lora
- reinforcement-learning
datasets:
- wikimedia/wikipedia
- OpenAssistant/oasst1
- databricks/databricks-dolly-15k
- merve/turkish_instructions
- GoktugD/turkish-extractive-qa-1.5m
- HuggingFaceTB/cosmopedia
- HuggingFaceTB/smol-smoltalk
- malhajar/OpenOrca-tr
- CohereForAI/aya_dataset
- rajpurkar/squad
- openai/gsm8k
- HuggingFaceH4/ultrachat_200k
- MarulAI/Marul-SFT-TR
- HuggingFaceTB/everyday-conversations-llama3.1-2k
- SoAp9035/everyday-conversations-tur
- serhatkagans/turkish-chat-sft-v3
- yunusemrejr/Franok-Session-Tasks
---

# Franok

Franok is an experimental English–Turkish language model by Yunus Emre Vurgun. Its **314,061,312** parameters began at random initialization. Franok's own byte-level tokenizer was fitted to the original training split. Twelve causal decoder layers each route a token to one of eight SwiGLU banks using an anchor from the preceding eight-token group. **49,820,160 parameters** participate per token, including the tied vocabulary head. The extended release trains on sequences up to **1,024 tokens**.

[Browser chat](https://franok.yunusemrevurgun.com/) · [Ubuntu installer](https://franok.yunusemrevurgun.com/downloads/franok_0.2.0_amd64.deb) · [Architecture film](https://youtu.be/XtNThi_pBjM) · [Source on GitHub](https://github.com/yunusemrejr/Franok) · [Colab notebook](https://colab.research.google.com/github/yunusemrejr/Franok/blob/main/training/Franok.ipynb)

## Files

| Files | Use |
|---|---|
| `model.safetensors`, `config.json`, `tokenizer.json` | Native FP16 checkpoint and bilingual vocabulary |
| `model.q4`, `model.json` | Browser-default Q4 weights with a per-group scale search, offsets, shapes and SHA-256 |
| `model.q8`, `model-q8.json` | Desktop-default Q8 weights and manifest |
| `fp32/` | Unrounded final FP32 adapter merge, configuration and reconstruction check |
| `adapters/base/` | Native model immediately before the first low-rank adaptation |
| `adapters/dialogue/`, `adapters/evidence/`, `adapters/rl/` | Rank-32 dialogue tuning, rank-16 contrastive/supervised tuning and rank-16 reinforcement tuning, in merge order |
| `training-record/` | Step counts, training loss, internal validation and rollout receipts |
| `source/` | Architecture, math, notebook, trainers, kernels, frontend, video and evaluation sources |

The architecture uses its own runtime. To use PyTorch, download the root configuration, tokenizer and FP16 checkpoint, install the dependencies from the source repository and run `python -m franok.chat DIRECTORY --lang en`. Use `--lang tr` for Turkish and `--system` for a session instruction. For the FP32 variant, put the same tokenizer beside its model and configuration. The Ubuntu application caches the verified weights for offline generation. Both apps support saved history, attached turns, labeled local arithmetic and a switchable conversation helper. For a standalone greeting, the app normalizes its spelling to “Hello!” or “Merhaba!” and decodes the neural model at temperature zero. It labels helper-written clarifications and can retry a substantive question without a prior confusing reply. Direct model evaluations bypass these features.

## How it was trained

One Colab L4 trained the first release in three phases from scratch for 53,859,151 next-token targets. A second L4 allocation continued this project's own [original checkpoint](https://huggingface.co/yunusemrejr/Franok/tree/7633d06180ebbc836af02c590f96f8ac28a4829a) on open English and Turkish corpora, expanded the context, then performed full supervised tuning. The dialogue adapter trained on assistant replies. The evidence adapter combined assistant supervision with a matching-answer versus distractor loss. The final adapter learned from six sampled completions per prompt using a group-relative policy gradient and a symbolic reward.

The original parameters and code are MIT licensed. The linked datasets retain their original licenses. Turkish conversational training includes [Marul-SFT-TR by Ahmet Karakuş (Marul AI)](https://huggingface.co/datasets/MarulAI/Marul-SFT-TR). Corpus selection, synthetic sources, attribution, splits and exact pinned revisions are documented in the [data guide](https://github.com/yunusemrejr/Franok/blob/main/docs/data.md). [Training objectives](https://github.com/yunusemrejr/Franok/blob/main/docs/training.md) and [architecture equations](https://github.com/yunusemrejr/Franok/blob/main/docs/architecture.md) are public.

## Evaluation

<!-- measured -->
Extended pretraining and supervised stages consumed 183,459,702 target tokens. Reinforcement learning sampled 11,880 replies and made 1,060 optimizer updates.

| Fixed test | Franok EN | Franok TR | SmolLM2-360M-Instruct EN | SmolLM2-360M-Instruct TR |
|---|---:|---:|---:|---:|
| Paired Belebele reading | 12/48 | 11/48 | 19/48 | 15/48 |
| Short two-choice tasks | 10/16 | 8/16 | 11/16 | 9/16 |

The native FP32 model passed 2/6 English and 0/6 Turkish checks in direct multi-turn generation. Its answers frequently lose the topic. The broader conversational and reading results fall below the comparison model on these tests.

Q4 chose the same answer as FP32 on 122/128 paired questions. Across 2,048 bilingual next-token predictions its top token matched FP32 on 83.2%, and mean negative log likelihood changed by +0.05113.

Q8 chose the same answer as FP32 on 128/128 paired questions. Across 2,048 bilingual next-token predictions its top token matched FP32 on 98.4%, and mean negative log likelihood changed by +0.00036.
<!-- end measured -->

The direct transcripts include incorrect and off-topic replies. Both interfaces condition on English or Turkish and let users customize the system prompt. The neural model can miss instructions, repeat itself or mix languages. A saved session can be longer than the neural context, and attaching a turn supplies its text without guaranteeing a correct answer. Scores and raw outputs are available in the [extended evaluation report](https://github.com/yunusemrejr/Franok/tree/main/docs/evaluation/v2).
