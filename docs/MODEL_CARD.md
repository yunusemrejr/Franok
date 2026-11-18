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
| `model.safetensors`, `config.json`, `tokenizer.json` | Native FP32 checkpoint of the 0.4 fine-tune and bilingual vocabulary |
| `model.q4`, `model.json` | Browser-default Q4 weights with a per-group scale search, offsets, shapes and SHA-256 |
| `model.q8`, `model-q8.json` | Desktop-default Q8 weights and manifest |
| `desktop/franok_0.4.0_amd64.deb` | Ubuntu installer with the same SHA-256 as the GitHub and website downloads |
| `pretrained/` | Decoder before instruction tuning, plus its tokenizer and six held-out language-modeling measurements |
| `reader/`, `search/` | Optional question-conditioned source-span reader and an 8-bit bilingual session-search encoder |
| `training-record/` | Step counts, training loss, internal validation and data-filter counts for each release |
| `source/` | Architecture, math, notebook, trainers, kernels, frontend, video and evaluation sources |

The 0.3 adapters, its FP16 checkpoint and its unrounded FP32 merge remain at the [0.3 revision](https://huggingface.co/yunusemrejr/Franok/tree/v0.3.0) of this repository.

The architecture uses its own runtime. To use PyTorch, download the root configuration, tokenizer and checkpoint, install the dependencies from the source repository and run `python -m franok.chat DIRECTORY --lang en`. Use `--lang tr` for Turkish and `--system` for a session instruction. The Ubuntu application caches the verified weights for offline generation. Both apps support saved history, attached turns, labeled local arithmetic and a switchable conversation helper. For a standalone greeting, the app normalizes its spelling to “Hello!” or “Merhaba!” and decodes the neural model at temperature zero. It labels helper-written clarifications and can retry a substantive question without a prior confusing reply. Direct model evaluations bypass these features.

## How it was trained

One Colab L4 trained the first release in three phases from scratch for 53,859,151 next-token targets. A second L4 allocation continued this project's own [original checkpoint](https://huggingface.co/yunusemrejr/Franok/tree/7633d06180ebbc836af02c590f96f8ac28a4829a) on open English and Turkish corpora, expanded the context, then performed full supervised tuning. The dialogue adapter trained on assistant replies. The evidence adapter combined assistant supervision with a matching-answer versus distractor loss. The final adapter learned from six sampled completions per prompt using a group-relative policy gradient and a symbolic reward.

The third run continued Franok’s [second checkpoint](https://huggingface.co/yunusemrejr/Franok/tree/db3e8a10bd3fcefde40acdc25ba7c6dc59cde32b). Another 383,385,600 pretraining targets came from educational and general web documents and short stories. Supervised batches included human English dialogues, Turkish conversations, questions about passages, same-conversation negative replies and examples that change format when the system instruction changes. A separate readout learns where a short source quotation begins and ends. A 96-dimensional encoder provides optional cross-language search over saved messages. Each readout has its own weights and evaluation.

Release 0.4 fine-tunes the third release's FP32 merge in full on 35,672 conversations: 19,834 English and 15,838 Turkish. Qwen3-8B and Qwen3-14B wrote teacher conversations, a grading pass removed weak ones, and a contamination filter removed overlap with the test sets. Programmatic dialogues with computed answers, short-span reading from SQuAD and SQuAD-TR, replayed 0.3 conversations and raw text make up the rest of the mixture. [Fourth-run stage](https://github.com/yunusemrejr/Franok/blob/main/docs/training.md#fourth-run-teacher-written-conversations)

The original parameters and code are MIT licensed. The linked datasets retain their original licenses. Teacher-written conversations come from Qwen3-8B and Qwen3-14B, both Apache 2.0. Turkish conversational training includes [Marul-SFT-TR by Ahmet Karakuş (Marul AI)](https://huggingface.co/datasets/MarulAI/Marul-SFT-TR). Corpus selection, synthetic sources, attribution, splits and exact pinned revisions are documented in the [data guide](https://github.com/yunusemrejr/Franok/blob/main/docs/data.md). [Training objectives](https://github.com/yunusemrejr/Franok/blob/main/docs/training.md) and [architecture equations](https://github.com/yunusemrejr/Franok/blob/main/docs/architecture.md) are public.

## Evaluation

<!-- measured -->
On 16 held-out two-turn conversations, Franok 0.4 met **3/32** turn criteria. Franok 0.3 met 0, Franok 0.2 met 0, SmolLM2-360M-Instruct met 8 and LFM2-350M met 6. A second set of 16 two-turn cases, created before any training conversation was generated, gave Franok 0.4 2/32 and LFM2-350M 8/32. Each follow-up includes the model's actual first reply. The 0.4 grades come from one non-blind rubric review by the assistant that ran the training.

| Test | Franok 0.4 | Franok 0.3 | Franok 0.2 | SmolLM2-360M-Instruct | LFM2-350M |
|---|---:|---:|---:|---:|---:|
| Reading four choices, English and Turkish / 240 | 66 | 66 | 59 | 70 | 78 |
| Free exact English answers / 64 | 10 | 1 | 0 | 12 | 8 |
| Free exact Turkish answers / 64 | 5 | 3 | 0 | 0 | 0 |
| Conversation criteria, published set / 32 | 3 | 0 | 0 | 8 | 6 |
| Conversation criteria, new held-out set / 32 | 2 | not run | not run | not run | 8 |

For 0.4, the browser Q4 scored 67/240 on Belebele and 14/128 exact on XQuAD, and the Ubuntu Q8 scored 66/240 and 15/128, against 66/240 and 15/128 for FP32. Average next-token loss changed by +0.04985 nats for Q4 and +0.00013 for Q8 across 1,344 positions. The optional passage reader is the unchanged 0.3 head; with FP32 weights it found 11/64 English and 5/64 Turkish answers exactly. [Browser and Ubuntu results](https://github.com/yunusemrejr/Franok/blob/main/docs/evaluation/v4/final.md#browser-and-ubuntu)
<!-- end measured -->

The direct transcripts include incorrect and off-topic replies. Both interfaces condition on English or Turkish and let users customize the system prompt. The neural model can miss instructions, repeat itself or mix languages. A saved session can be longer than the neural context, and attaching a turn supplies its text without guaranteeing a correct answer. Scores and raw outputs are available in the [extended evaluation report](https://github.com/yunusemrejr/Franok/blob/main/docs/evaluation/v4/final.md).
