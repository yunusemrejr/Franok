# Franok 0.3 results

Both languages use the same 314,061,312-parameter decoder and tokenizer. The 16 two-turn conversations feed each model’s actual first answer back into its second prompt. The passage and natural-text tests use fixed bilingual inputs. [Test methods](README.md)

## Training

| Stage | Optimizer steps | Training targets | Selected adapter scale |
|---|---:|---:|---:|
| Continued pretraining | 31,200 | 383,385,600 | — |
| sft | 8,190 | 25,312,438 | — |
| dialogue | 2,545 | 7,228,724 | 0.35 |
| evidence | 1,714 | 3,322,925 | 0.7 |

One Colab L4 was allocated for 236.6 minutes, including preparation and transfer. The reinforcement stage sampled 7,338 replies and 83,187 tokens. It made 825 updates and selected adapter scale 1.0. Supervised target counts include answer positions used to compare preferred and rejected responses. A sampled position can recur during training. [Training receipts](post.jsonl) · [Reinforcement record](rl-finished.json) · [Allocation](allocation.json)

## Direct model comparisons

| Model | Conversation criteria / 32 | Belebele / 240 | XQuAD exact EN / 64 | XQuAD exact TR / 64 | English bits/byte ↓ | Turkish bits/byte ↓ |
|---|---:|---:|---:|---:|---:|---:|
| [Franok 0.2](parent-open.json) | 0 | 59 | 0 | 0 | 1.578 | 1.658 |
| [SmolLM2-360M-Instruct](peer-open.json) | 8 | 70 | 12 | 0 | 0.828 | 2.000 |
| [LFM2-350M](lfm2-open.json) | 6 | 78 | 8 | 0 | 0.976 | 2.060 |
| [Franok 0.3 FP32](final-open.json) | 0 | 66 | 1 | 3 | 1.405 | 1.360 |

Conversation criteria were graded by one model-assisted reviewer against the published rubric. Belebele ranks four supplied choices, while XQuAD generates an answer freely. The paired 95% bootstrap interval for Franok 0.3 minus SmolLM2 on Belebele is -7.9 to +4.6 percentage points. Both translations of each passage stay together during resampling. [Rubric judgments](final-rubric.json) · [Paired differences](paired.json)

## Browser and Ubuntu

| Runtime | Belebele / 240 | Choices matching FP32 / 240 | Free XQuAD exact / 128 | Passage-reader exact / 128 | Top token matching FP32 | Change in target NLL |
|---|---:|---:|---:|---:|---:|---:|
| [Q4 WASM](final-q4-quant.json) | 66 | 227 | 3 | 19 | 88.5% | +0.04334 |
| [Q8 WASM](final-q8-quant.json) | 67 | 238 | 4 | 19 | 98.9% | +0.00282 |

The optional FP32 passage reader found 12/64 exact English answers and 7/64 Turkish answers. It returned a quotation on 56/64 English and 40/64 Turkish questions. Abstentions count as misses on this answerable set. [Native reader output](final-reader.json) · [Q4 outputs](final-q4-reader.json) · [Q8 outputs](final-q8-reader.json)

The distribution check covers 1,344 target positions on 96 validation prefixes. A small average loss change can still alter a particular sampled sentence. The earlier 12 automatic dialogue checks also have full transcripts. [Q4 conversations](final-q4-conversations.json) · [Q8 conversations](final-q8-conversations.json)

## Changing the system prompt

Each language has 24 distinct held-out passage questions with two choices. Each question is asked twice at temperature zero. Only the system instruction changes. One version requires the correct letter and the other requires the exact answer text. Each question balances the same choices across the two requests, and the correct letter is balanced across questions.

| Runtime | EN letter / 24 | EN words / 24 | EN both / 24 | TR letter / 24 | TR words / 24 | TR both / 24 |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 Q4 | 0 | 0 | 0 | 0 | 0 | 0 |
| SmolLM2-360M-Instruct | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.3 Q4 | 11 | 3 | 1 | 11 | 1 | 1 |
| 0.3 Q8 | 10 | 3 | 2 | 11 | 2 | 2 |

## Long sessions

The native memory task asks the model which of four words was assigned to a folder. Sixteen cases per length cover eight labels in each language. These are forced choices with conversation history inside the model window.

| Model | 128-position budget / 16 | 384-position budget / 16 | 960-position budget / 16 |
|---|---:|---:|---:|
| Franok 0.2 | 6 | 5 | 6 |
| SmolLM2-360M-Instruct | 12 | 12 | 8 |
| Franok 0.3 FP32 | 8 | 4 | 6 |

The workspace also searches earlier saved messages outside that window. Users can inspect and attach a source or let bilingual search select earlier context. These operations have their own retrieval measurements. [Native memory cases](final-long.json) · [Bilingual session search](meaning-search.json)
