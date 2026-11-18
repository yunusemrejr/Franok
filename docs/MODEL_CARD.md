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
- HuggingFaceFW/fineweb-edu
- HuggingFaceFW/fineweb-2
- roneneldan/TinyStories
- esat-krky/TinyStories_Turkish
- rajpurkar/squad_v2
- OpenAssistant/oasst2
- tau/commonsense_qa
- cisimcik/turkish-chat-max-25k
- sentence-transformers/parallel-sentences
---

# Franok

Franok is an experimental English–Turkish language model by Yunus Emre Vurgun. Its **314,061,312** parameters began at random initialization. Franok's own byte-level tokenizer was fitted to the original training split. Twelve causal decoder layers each route a token to one of eight SwiGLU banks using an anchor from the preceding eight-token group. **49,820,160 parameters** participate per token, including the tied vocabulary head. The extended release trains on sequences up to **1,024 tokens**.

[Browser chat](https://franok.yunusemrevurgun.com/) · [Ubuntu installer](https://franok.yunusemrevurgun.com/downloads/ubuntu) · [Architecture film](https://youtu.be/XtNThi_pBjM) · [Source on GitHub](https://github.com/yunusemrejr/Franok) · [Colab notebook](https://colab.research.google.com/github/yunusemrejr/Franok/blob/main/training/Franok.ipynb)

## Files

| Files | Use |
|---|---|
| `model.safetensors`, `config.json`, `tokenizer.json` | Native FP16 checkpoint and bilingual vocabulary |
| `model.q4`, `model.json` | Browser-default Q4 weights with a per-group scale search, offsets, shapes and SHA-256 |
| `model.q8`, `model-q8.json` | Desktop-default Q8 weights and manifest |
| `desktop/franok_0.3.0_amd64.deb` | Ubuntu installer with the same SHA-256 as the GitHub and website downloads |
| `fp32/` | Unrounded final FP32 adapter merge, configuration and reconstruction check |
| `pretrained/` | Decoder before instruction tuning, plus its tokenizer and six held-out language-modeling measurements |
| `adapters/base/` | Native model immediately before the first low-rank adaptation |
| `adapters/dialogue/`, `adapters/evidence/`, `adapters/rl/` | Rank-32 dialogue tuning, rank-16 contrastive/supervised tuning and rank-16 reinforcement tuning, in merge order |
| `adapters/pre-rl/` and `adapters/rl/updates.safetensors` | Rounded input checkpoint and exact training-host matrix products for reproducing the unrounded final merge |
| `reader/`, `search/` | Optional question-conditioned source-span reader and an 8-bit bilingual session-search encoder |
| `dialogue-reward/` | Training-only reply-preference head, training counts and held-out validation |
| `training-record/` | Step counts, training loss, internal validation and rollout receipts |
| `source/` | Architecture, math, notebook, trainers, kernels, frontend, video and evaluation sources |

The architecture uses its own runtime. To use PyTorch, download the root configuration, tokenizer and FP16 checkpoint, install the dependencies from the source repository and run `python -m franok.chat DIRECTORY --lang en`. Use `--lang tr` for Turkish and `--system` for a session instruction. For the FP32 variant, put the same tokenizer beside its model and configuration. The Ubuntu application caches the verified weights for offline generation. Both apps support saved history, attached turns, labeled local arithmetic and a switchable conversation helper. For a standalone greeting, the app normalizes its spelling to “Hello!” or “Merhaba!” and decodes the neural model at temperature zero. It labels helper-written clarifications and can retry a substantive question without a prior confusing reply. Direct model evaluations bypass these features.

## How it was trained

One Colab L4 trained the first release in three phases from scratch for 53,859,151 next-token targets. A second L4 allocation continued this project's own [original checkpoint](https://huggingface.co/yunusemrejr/Franok/tree/7633d06180ebbc836af02c590f96f8ac28a4829a) on open English and Turkish corpora, expanded the context, then performed full supervised tuning. The dialogue adapter trained on assistant replies. The evidence adapter combined assistant supervision with a matching-answer versus distractor loss. The final adapter learned from six sampled completions per prompt using a group-relative policy gradient and a symbolic reward.

The third run continued Franok’s [second checkpoint](https://huggingface.co/yunusemrejr/Franok/tree/db3e8a10bd3fcefde40acdc25ba7c6dc59cde32b). Another 383,385,600 pretraining targets came from educational and general web documents and short stories. Supervised batches included human English dialogues, Turkish conversations, questions about passages, same-conversation negative replies and examples that change format when the system instruction changes. A separate readout learns where a short source quotation begins and ends. A 96-dimensional encoder provides optional cross-language search over saved messages. Each readout has its own weights and evaluation.

The original parameters and code are MIT licensed. The linked datasets retain their original licenses. Turkish conversational training includes [Marul-SFT-TR by Ahmet Karakuş (Marul AI)](https://huggingface.co/datasets/MarulAI/Marul-SFT-TR). Corpus selection, synthetic sources, attribution, splits and exact pinned revisions are documented in the [data guide](https://github.com/yunusemrejr/Franok/blob/main/docs/data.md). [Training objectives](https://github.com/yunusemrejr/Franok/blob/main/docs/training.md) and [architecture equations](https://github.com/yunusemrejr/Franok/blob/main/docs/architecture.md) are public.

## Evaluation

<!-- measured -->
On 16 held-out two-turn conversations, Franok 0.3 met **0/32** turn criteria. Franok 0.2 met 0, SmolLM2-360M-Instruct met 8 and LFM2-350M met 6. Each follow-up includes the model's actual first reply.

| Test | Franok 0.3 | Franok 0.2 | SmolLM2-360M-Instruct | LFM2-350M |
|---|---:|---:|---:|---:|
| Reading four choices, English and Turkish / 240 | 66 | 59 | 70 | 78 |
| Free exact English answers / 64 | 1 | 0 | 12 | 8 |
| Free exact Turkish answers / 64 | 3 | 0 | 0 | 0 |

With a source attached, the separate passage reader found 12/64 exact English answers and 7/64 Turkish answers. Browser Q4 matched the full-precision reading choice on 227/240 questions and Ubuntu Q8 on 238/240. Average next-token loss changed by +0.04334 nats for Q4 and +0.00282 for Q8 across 1,344 positions. Individual generated replies can differ. [Measurements, rubric and transcripts](https://github.com/yunusemrejr/Franok/blob/main/docs/evaluation/v3/final.md)
<!-- end measured -->

The direct transcripts include incorrect and off-topic replies. Both interfaces condition on English or Turkish and let users customize the system prompt. The neural model can miss instructions, repeat itself or mix languages. A saved session can be longer than the neural context, and attaching a turn supplies its text without guaranteeing a correct answer. Scores and raw outputs are available in the [extended evaluation report](https://github.com/yunusemrejr/Franok/blob/main/docs/evaluation/v3/final.md).
