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
- allenai/soda
---

# Franok

Franok is an experimental English–Turkish language model by Yunus Emre Vurgun. Its **314,061,312** parameters began at random initialization. Franok's own byte-level tokenizer was fitted to the original training split. Twelve causal decoder layers each route a token to one of eight SwiGLU banks using an anchor from the preceding eight-token group. **49,820,160 parameters** participate per token, including the tied vocabulary head. The extended release trains on sequences up to **1,024 tokens**.

[Browser chat](https://franok.yunusemrevurgun.com/) · [Ubuntu installer](https://franok.yunusemrevurgun.com/downloads/ubuntu) · [Architecture film](https://youtu.be/XtNThi_pBjM) · [Source on GitHub](https://github.com/yunusemrejr/Franok) · [Colab notebook](https://colab.research.google.com/github/yunusemrejr/Franok/blob/main/training/Franok.ipynb)

## Files

| Files | Use |
|---|---|
| `model.safetensors`, `config.json`, `tokenizer.json` | Native FP32 checkpoint of the 0.5 fine-tune and bilingual vocabulary |
| `model.q4`, `model.json` | Q4 weights with a per-group scale search, offsets, shapes and SHA-256, loaded by the website on phones and smaller machines |
| `model.q8`, `model-q8.json` | Q8 weights and manifest, loaded by the Ubuntu app and by the website on desktop-class browsers |
| `desktop/franok_0.5.0_amd64.deb` | Ubuntu installer with the same SHA-256 as the GitHub and website downloads |
| `pretrained/` | Decoder before instruction tuning, plus its tokenizer and six held-out language-modeling measurements |
| `reader/`, `search/` | Optional question-conditioned source-span reader and an 8-bit bilingual session-search encoder |
| `training-record/` | Step counts, training loss, internal validation and data-filter counts for each release |
| `source/` | Architecture, math, notebook, trainers, kernels, frontend, video and evaluation sources |

The 0.3 adapters, its FP16 checkpoint and its unrounded FP32 merge remain at the [0.3 revision](https://huggingface.co/yunusemrejr/Franok/tree/v0.3.0) of this repository. The 0.4 weights and files remain at the [0.4 revision](https://huggingface.co/yunusemrejr/Franok/tree/v0.4.0).

The architecture uses its own runtime. To use PyTorch, download the root configuration, tokenizer and checkpoint, install the dependencies from the source repository and run `python -m franok.chat DIRECTORY --lang en`. Use `--lang tr` for Turkish and `--system` for a session instruction. The Ubuntu application caches the verified weights for offline generation. The website loads the Q8 weights (353 MB) on desktop-class browsers, meaning no phone user agent and at least 8 GB of device memory, or at least 8 cores where the browser does not report memory, and the Q4 weights (196 MB) on phones and smaller machines. The Ubuntu app loads Q8. Both apps detect whether each message is English or Turkish and switch the reply language to match. They support saved history, attached turns, labeled local arithmetic and a switchable conversation helper. For a standalone greeting, the app normalizes its spelling to “Hello!” or “Merhaba!” and decodes the neural model at temperature zero, which is the app's default sampling temperature. It labels helper-written clarifications and can retry a substantive question without a prior confusing reply. Direct model evaluations bypass these features.

## How it was trained

One Colab L4 trained the first release in three phases from scratch for 53,859,151 next-token targets. A second L4 allocation continued this project's own [original checkpoint](https://huggingface.co/yunusemrejr/Franok/tree/7633d06180ebbc836af02c590f96f8ac28a4829a) on open English and Turkish corpora, expanded the context, then performed full supervised tuning. The dialogue adapter trained on assistant replies. The evidence adapter combined assistant supervision with a matching-answer versus distractor loss. The final adapter learned from six sampled completions per prompt using a group-relative policy gradient and a symbolic reward.

The third run continued Franok’s [second checkpoint](https://huggingface.co/yunusemrejr/Franok/tree/db3e8a10bd3fcefde40acdc25ba7c6dc59cde32b). Another 383,385,600 pretraining targets came from educational and general web documents and short stories. Supervised batches included human English dialogues, Turkish conversations, questions about passages, same-conversation negative replies and examples that change format when the system instruction changes. A separate readout learns where a short source quotation begins and ends. A 96-dimensional encoder provides optional cross-language search over saved messages. Each readout has its own weights and evaluation.

Release 0.4 fine-tuned the third release's FP32 merge in full on 35,672 practical-advice conversations written by Qwen3-8B and Qwen3-14B, with a grading pass and a contamination filter.

Release 0.5 fine-tunes the 0.4 FP32 weights in full. Architecture, tokenizer, reader head and search data are unchanged. Version 0.4 failed at greetings, small talk, questions about itself, casually typed messages, repair turns and requests for a poem, a short Python example or help with maths, so the new dialogues cover those cases in thirteen families, seven user typing styles and both languages. google/gemma-4-26B-A4B-it wrote 63,000 dialogues (48,000 from ten general families and 15,000 from create, code and math) and graded each one twice, and 37,145 survived the filter. 1,500 more Turkish dialogues came from google/gemma-4-31B-it without grading, and 1,377 were kept. Code computed 12,778 further dialogues about the model's name, its maker, confirming its name and everyday basics. The fifth-run set holds 51,300 dialogues. The 35,672 conversations of 0.4 stay as a secondary stream, and 150 million tokens of SODA transcripts join the English raw text. Stage A trained on every token of a dialogue for 15 minutes and stage B on assistant tokens only for 5 minutes, both on one A100 80 GB. The kept checkpoint has the lowest assistant-token loss on held-out development dialogues, 1.489 against 2.948 for 0.4, and SHA-256 `419f3ad443df455110a8e0bbfbd080c040fb5ee5321f85d0e8a0f89def2f620f`. [Fifth-run stage](https://github.com/yunusemrejr/Franok/blob/main/docs/training.md#fifth-run-everyday-conversation)

The original parameters and code are MIT licensed. The linked datasets retain their original licenses. Teacher-written conversations come from Qwen3-8B and Qwen3-14B for 0.4 and from google/gemma-4-26B-A4B-it and google/gemma-4-31B-it for 0.5, all Apache 2.0. The SODA dialogues by Kim et al. (2023) are CC BY 4.0. Turkish conversational training includes [Marul-SFT-TR by Ahmet Karakuş (Marul AI)](https://huggingface.co/datasets/MarulAI/Marul-SFT-TR). Corpus selection, synthetic sources, attribution, splits and exact pinned revisions are documented in the [data guide](https://github.com/yunusemrejr/Franok/blob/main/docs/data.md). [Training objectives](https://github.com/yunusemrejr/Franok/blob/main/docs/training.md) and [architecture equations](https://github.com/yunusemrejr/Franok/blob/main/docs/architecture.md) are public.

## Evaluation

<!-- measured -->
Franok 0.5 was tested on 24 held-out cases (50 turns, 12 cases per language) and on 10 cases (22 turns, 11 per language) built from failed chats of the project's author. Transcripts are FP32 at temperature 0. A blind judge, google/gemma-4-31B-it, saw each reply without the system name, in shuffled order, and gave two verdicts: RUBRIC (the reply meets the case's written requirement) and SENSIBLE (coherent, on topic, right language). Franok 0.4 and LFM2-350M answered the same scripts. Each cell shows English, then Turkish.

| Test | Franok 0.5 | Franok 0.4 | LFM2-350M |
|---|---:|---:|---:|
| Held-out set, RUBRIC / 25 per language | 7, 7 | 0, 2 | 10, 0 |
| Held-out set, SENSIBLE / 25 per language | 11, 14 | 3, 6 | 13, 0 |
| Held-out set, keyword criteria / 7 per language | 3, 3 | 0, 1 | 2, 0 |
| Author's chats, RUBRIC / 11 per language | 6, 5 | 4, 3 | 5, 0 |
| Author's chats, SENSIBLE / 11 per language | 7, 5 | 3, 3 | 8, 0 |

On 1,344 positions the Q4 weights changed average next-token loss by +0.05006 nats and matched the FP32 top token at 86.8% of positions. For the Q8 weights the figures are +0.00105 and 98.4%. Belebele, XQuAD and reader scores were not re-measured on the Q4 and Q8 runtimes for 0.5. Franok 0.5 answered 61 of 240 Belebele questions, against 66 for 0.4. The optional passage reader is the unchanged 0.3 head. [Browser and Ubuntu results](https://github.com/yunusemrejr/Franok/blob/main/docs/evaluation/v5/final.md#browser-and-ubuntu)
<!-- end measured -->

The test prompts are short everyday chat, close to the training distribution by design, and a judge model can be wrong, so the transcripts are published next to the scores. The model has 50 million active parameters, so factual knowledge, long answers, real programming and multi-step maths remain out of reach. The direct transcripts include incorrect and off-topic replies. Greetings, questions about itself, thanks and simple repair turns usually work in both languages. Facts, sums, rewriting requests and step-by-step advice are often wrong: it answered "There are 120 days in a week" and gave 72 for "12 plus 30". Poems and code are short and often wrong, and it can lose the topic on a follow-up. Advice from the model must not be relied on. Both interfaces detect English or Turkish and let users customize the system prompt. The neural model can miss instructions, repeat itself or mix languages. A saved session can be longer than the neural context, and attaching a turn supplies its text without guaranteeing a correct answer. Scores and raw outputs are available in the [extended evaluation report](https://github.com/yunusemrejr/Franok/blob/main/docs/evaluation/v5/final.md).
