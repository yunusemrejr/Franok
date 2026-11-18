# Franok 0.4 results

Franok 0.4 keeps the 0.3 architecture: 314,061,312 parameters, 49,820,160 of them on one token's path, 12 layers, an 8,192-piece tokenizer and a 1,024-token window. The conversation tests feed each model's actual first answer back into its second prompt. The passage and natural-text tests use fixed bilingual inputs. [Test methods](../v3/README.md)

## Training

Franok 0.4 is a full-parameter fine-tune of the 0.3 FP32 merge, SHA-256 `b7d97329…`. It uses no low-rank adapter and no reinforcement stage. The optimizer is AdamW with betas 0.9 and 0.95 and weight decay 0.01. The peak learning rate is 4e-5, reached after 100 warm-up steps, then decayed on a cosine schedule to 5% of the peak. Loss is computed on assistant tokens only, including the end marker.

Each micro-batch comes from one of four components. The component furthest behind its target share supplies the next one.

| Component | Target tokens | Share |
|---|---:|---:|
| New conversations: teacher-written, programmatic and round-1 | 9,654,347 | 55% |
| Short-span reading | 3,509,459 | 20% |
| Replay of 0.3 conversations | 1,753,884 | 10% |
| Raw English and Turkish text | 2,637,824 | 15% |

The run took 3,940 optimizer steps in 1,400 seconds. The released checkpoint is step 768, SHA-256 `8259f95e…`.

Candidates were scored on development conversations (teacher and arithmetic rows), development reading questions and held-out raw text in each language. The selected checkpoint has the lowest development conversation loss among candidates whose raw-text loss rose by no more than 0.02 nats over the starting model in both languages. Step 768 qualifies with conversation loss 2.443. Later steps reached 2.31, but each one raised raw-text loss past the limit. At step 3,940, English raw-text loss was 3.622 against a limit of 3.594.

Average loss in nats per target token, at the start and at step 768: conversations 3.605 to 2.443, reading 2.398 to 1.065, programmatic 3.039 to 0.806, raw English 3.574 to 3.590 and raw Turkish 3.742 to 3.749. The programmatic development set is diagnostic and the selection does not use it.

## Training data

| Source | Kept, English / Turkish | Total |
|---|---:|---:|
| Round-1 set: teacher-written conversations and 3,000 arithmetic conversations from code templates | 8,839 / 9,546 | 18,385 |
| Round-2 teacher-written conversations | 8,389 / 3,493 | 11,882 |
| Programmatic dialogues | 2,606 / 2,799 | 5,405 |
| All conversations | 19,834 / 15,838 | 35,672 |

Teacher-written conversations come from Qwen3-8B (revision `b968826d9c46dd6066d109eabc6255188de91218`) and Qwen3-14B (revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, FP8). Both are Apache-2.0. Round 1 used Qwen3-8B for English and Qwen3-14B for Turkish. Round 2 used Qwen3-14B for both languages. Thinking was disabled. Each seed fixes a topic, a task type, a persona, a style and a twist, and the teacher writes one conversation per seed at temperature 0.8.

Each conversation gets two grading passes from the same teacher: a general check and a stricter one. A conversation is kept only when both passes return GOOD.

Programmatic dialogues are computed from templates and never written by a model. They cover object tracking across two turns (1,202 kept), one-word label choices with the options given (2,390), clarifying questions for vague requests (493), and counting or unit arithmetic with a "just the number" follow-up (1,320). Each answer is computed from the state the user described.

The filter removes malformed output, wrong-language turns, duplicates and overlong turns. It also removes a conversation when one of its user turns has a content-word Jaccard similarity of at least 0.3 with a held-out test turn, or shares any 6-gram with one. The test files are the four held-out conversation files in `tests/`. The XQuAD and Belebele questions count as test items too. Round 2 kept 11,882 of 26,000 teacher conversations and 5,405 of 6,000 programmatic dialogues.

Short-span reading examples are answerable questions from the checksum-verified SQuAD 2.0 training file, which holds the SQuAD v1.1 training questions, plus the SQuAD-TR training file. Passages that share an 8-gram with an XQuAD passage are removed. The reading set has 138,493 training and 600 development examples.

Replay uses the 0.3 conversations. Raw text comes from the pretraining shards used for 0.3.

## Results

Results are FP32 unless stated.

| Measure | Franok 0.4 | Franok 0.3 | Franok 0.2 | SmolLM2-360M-Instruct | LFM2-350M |
|---|---:|---:|---:|---:|---:|
| Belebele, /240 | 66 | 66 | 59 | 70 | 78 |
| XQuAD exact EN, /64 | 10 | 1 | 0 | 12 | 8 |
| XQuAD exact TR, /64 | 5 | 3 | 0 | 0 | 0 |
| XQuAD F1 EN | 22.3% | 3.2% | 4.6% | 38.1% | 39.9% |
| XQuAD F1 TR | 13.7% | 8.2% | 3.5% | 3.6% | 6.8% |
| Bits per byte EN ↓ | 1.417 | 1.405 | 1.578 | 0.828 | 0.976 |
| Bits per byte TR ↓ | 1.371 | 1.360 | 1.658 | 2.000 | 2.060 |
| Conversation criteria, published set, /32 | 3 | 0 | 0 | 8 | 6 |
| Conversation criteria, new held-out set, /32 | 2 | not run | not run | not run | 8 |

Belebele gives four supplied choices per question, and XQuAD asks the model to generate the answer. Bits per byte measures how well a model predicts held-out text, so lower is better. Belebele was answered correctly 31 times in English and 35 times in Turkish, out of 120 questions per language. XQuAD exact answers were 10 of 64 in English and 5 of 64 in Turkish. [Belebele](franok04r2-belebele.json) · [XQuAD](franok04r2-xquad.json) · [Bits per byte](franok04r2-byte.json)

Franok 0.4 gives more exact XQuAD answers than LFM2-350M in both languages, and its Turkish bits per byte is lower than LFM2's. It matches 0.3 on Belebele. Its exact XQuAD gain over 0.3 is 9 answers in English and 2 in Turkish, from 64 questions per language. Its bits per byte is about 0.01 higher than 0.3's in each language. LFM2-350M is better on English bits per byte.

## Conversation

The published transfer set has 16 two-turn cases in English and Turkish, with 32 criteria. Franok 0.4 passed 3 of them. Franok 0.3 passed 0, LFM2-350M passed 6 and SmolLM2-360M-Instruct passed 8. [Franok 0.4 transcripts](franok04r2-transfer-conversations.json)

The new held-out set, `tests/v4_conversations.json`, has 16 two-turn cases in English and Turkish. Its file was created before any training conversation was generated, and it has not changed since. Franok 0.4 passed 2 of 32 criteria, both in Turkish. LFM2-350M passed 8, all in English. [Franok 0.4 transcripts](franok04r2-v4-conversations.json) · [LFM2-350M transcripts](lfm2-v4-conversations.json)

The 0.4 grades come from a single, non-blind rubric review by the assistant that ran the training. The reviewer saw each reply truncated at 170 characters. [Rubric judgments](rubric-r2.json). The 0.2, 0.3, SmolLM2 and LFM2 figures on the published set come from the [v3 report](../v3/final.md), which was graded by one model-assisted reviewer.

## Browser and Ubuntu

| Runtime | Belebele / 240 | Choices matching FP32 / 240 | Free XQuAD exact / 128 | Passage-reader exact / 128 | Top token matching FP32 | Change in target NLL |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | [66](franok04r2-belebele.json) | 240 | [15](franok04r2-xquad.json) | [16](franok04-reader.json) | 100% | 0 |
| [Q4 WASM](franok04-q4-quant.json) | [67](franok04-q4-belebele.json) | 175 | [14](franok04-q4-xquad.json) | [16](franok04-q4-reader.json) | 88.4% | +0.04985 |
| [Q8 WASM](franok04-q8-quant.json) | [66](franok04-q8-belebele.json) | 178 | [15](franok04-q8-xquad.json) | [18](franok04-q8-reader.json) | 99.1% | +0.00013 |

The browser loads Q4 and the Ubuntu app loads Q8. Choices matching FP32 counts Belebele items where the quantized runtime picked the same letter as FP32, right or wrong. Franok 0.4 scores near the 60/240 that chance would give on Belebele, so small numeric changes flip many of its choices. The distribution check covers 1,344 target positions on 96 validation prefixes.

The optional passage reader is the 0.3 head, unchanged. On Franok 0.4 decoder states it found 28/192 English and 35/192 Turkish validation spans exactly, against 26 and 34 on 0.3. A 20-step refit lowered both, so the head was kept as it was. With FP32 weights it found 11/64 English and 5/64 Turkish XQuAD answers exactly, close to the 15/128 that Franok 0.4 now produces without it.

## Reproduce

The commands below assume a clean checkout at the project root. The teacher step needs a GPU that vLLM can run with FP8 weights. Time budgets are wall-clock limits, so step counts depend on the hardware.

```sh
# Round-2 teacher conversations: 30,000 seeds, each graded twice
python -m training.fourth.teacher --model Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 --count 30000 --seed 2024 --id-prefix r2 --chunk 1000 --quantization fp8 --max-seqs 256 --gpu-util 0.92 --grade --max-minutes 66 --out data/r2.jsonl

# Short-span reading examples; the SQuAD and SQuAD-TR files are downloaded and checksum-verified
python -m training.fourth.reading --out data/reading.jsonl

# Parse, grade, de-contaminate and add 6,000 programmatic dialogues to the round-1 set
python -m training.fourth.filter data/r2.jsonl --grades data/r2.grades.jsonl --existing data/conversations-r1.jsonl --programmatic 6000 --arithmetic 0 --out data/conversations-r2.jsonl --report data/filter-report-r2.json

# Full-parameter fine-tune from the 0.3 FP32 merge
python -m training.fourth.finetune --weights weights/fp32/model.safetensors --expect-sha b7d97329fb1ea8250a1818c168217d9b163178640138c11069ca8209a4552530 --conversations data/conversations-r2.jsonl --reading data/reading.jsonl --out run2 --minutes 40 --epochs 6 --eval-minutes 2 --max-raw-rise 0.02
```

The round-1 file `data/conversations-r1.jsonl` comes from the earlier round. Its teacher runs used `training.fourth.teacher --lang en --count 12000 --seed 40404 --chunk 2048 --grade` with Qwen3-8B, and `--lang tr --count 12000 --seed 11 --model Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 --quantization fp8 --chunk 1000 --max-seqs 256 --grade` with Qwen3-14B. The release evaluation ran these commands on the selected checkpoint:

```sh
python -m tools.transfer belebele --run RUN --threads 4 --out franok04r2-belebele.json
python -m tools.transfer xquad --run RUN --threads 4 --out franok04r2-xquad.json
python -m tools.byte_fluency --run RUN --data DATA --threads 4 --out franok04r2-byte.json
python -m tools.compare_chat --run RUN --cases tests/transfer_conversations.json --out franok04r2-transfer-conversations.json
python -m tools.compare_chat --run RUN --cases tests/v4_conversations.json --out franok04r2-v4-conversations.json
python -m tools.compare_chat --peer lfm2 --cases tests/v4_conversations.json --out lfm2-v4-conversations.json
```

`RUN` is the selected checkpoint folder. `DATA` is the prepared validation shards from the third run. The training receipts are described in the [training guide](../../training.md#fourth-run-teacher-written-conversations).

## Limits

The conversations were written by Qwen3 models and graded by the same teacher. A reply can pass both grading passes and still contain a factual error or unnatural Turkish. The contamination check covers the held-out files and the XQuAD and Belebele questions. Overlap between training sources, and between training data and internet benchmarks beyond the listed test files, has not been exhaustively audited.

All conversation grades for 0.4 come from one non-blind review. Each held-out set has 16 cases, so a difference of one or two criteria is not a reliable signal.

Franok 0.4 still trails LFM2-350M on conversation criteria, Belebele and English bits per byte. Its Belebele score equals 0.3's. Its bits per byte is slightly higher than 0.3's in both languages.

These tests were not rerun for 0.4: the system-prompt choice test, long-session memory, bilingual session search and the passage reader.
