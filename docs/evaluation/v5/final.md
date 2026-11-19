# Franok 0.5 results

Franok 0.5 keeps the 0.4 architecture: 314,061,312 parameters, 49,820,160 of them on one token's path, 12 layers, an 8,192-piece tokenizer and a 1,024-token window. The tokenizer, the passage-reader head and the search data are unchanged. The conversation tests feed each model's actual first answer back into its second prompt. [Test methods](../v3/README.md)

## Training

Franok 0.5 is a full-parameter fine-tune of the 0.4 FP32 weights, SHA-256 `8259f95e933c5e021dc29258bb21982532229e574bd7bc29cc83417a14b67738`. Release 0.4 was trained only on practical-advice dialogues. In real use it failed at greetings and small talk, questions about itself, casually typed messages, "what do you mean" repair turns, and requests for a poem, a tiny Python example or help with maths. The optimizer is AdamW with betas 0.9 and 0.95, weight decay 0.01 and gradient clip 1.0. Each micro-batch holds 32,768 padded positions, two micro-batches make a step, and the cosine schedule is set by wall-clock time on one A100 80 GB.

Each micro-batch comes from one of four components. The component furthest behind its target share supplies the next one.

| Stage | Loss on | Peak learning rate | Minutes | Steps | Fifth-run dialogues | Round one | Passage reading | Raw text |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | every token of a dialogue | 1.5e-4 | 15 | 3,401 | 47% | 10% | 6% | 37% |
| B | assistant tokens only | 3e-5 | 5 | 1,130 | 60% | 10% | 6% | 24% |

A first attempt ran stage A for 68 minutes on a set of 30,323 dialogues. Held-out conversation loss was lowest at the first checkpoint, 1.471 at nine minutes, and rose to 1.486 and then 1.640, so the model was memorising the set. That run was discarded, the set was enlarged to 51,300 dialogues and the schedule was shortened.

In both stages the checkpoint with the lowest assistant-token loss on held-out development dialogues is kept, whatever the training mask, so the stages are compared on one scale. Stage B starts from the best stage-A checkpoint with a 50-step warm-up. Average assistant-token loss in nats on the development dialogues, at each evaluation:

| Stage | Evaluations, every 2.5 minutes in A and 1.25 minutes in B |
|---|---|
| A | 2.948 at the start, then 1.919, 1.702, 1.598, 1.554, 1.534, 1.531 |
| B | 1.531 at the start, then 1.508, 1.492, 1.490, 1.489 |

The released checkpoint is step 1,130 of stage B, with loss 1.489 (English 1.416, Turkish 1.553) and SHA-256 `419f3ad443df455110a8e0bbfbd080c040fb5ee5321f85d0e8a0f89def2f620f`. Over both stages the held-out passage-reading loss went from 1.065 to 1.060 and raw-text loss from 3.590 to 3.578 in English and from 3.749 to 3.673 in Turkish, in nats per token.

## Training data

Fifth-run dialogues are written from seeds in `training/fifth/seeds.py`. They cover thirteen families: smalltalk, facts, repair, explain, howto, feelings, opinions, followup, writing, unknowns, create, code and math. Create covers four-line poems, two-sentence stories, jokes, riddles and short messages. Code is one to four lines of plain-text Python plus one sentence. Math is an area explained in one or two sentences with one small worked example. Seven user typing styles occur: clean, lowercase, typos, abbreviations, Turkish typed without Turkish letters, very short and mixed. Assistant replies are always clean. Turkish prompts are written in Turkish, and a dialogue has two to five turns.

The teacher is google/gemma-4-26B-A4B-it (revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, Apache-2.0), run with vLLM 0.34.0 on one A100 80 GB. It was asked for 63,000 dialogues in three batches: 24,000 from the ten general families, 15,000 from create, code and math, and 24,000 more from the ten general families (requested as 60,000 and stopped by a 24-minute cap). The same model graded every parsed dialogue twice, with a standard and a strict prompt, and a dialogue is kept only when both say GOOD. Of the 63,000, 1,939 were malformed and the grader flagged 8,479. A further 1,500 Turkish dialogues came from google/gemma-4-31B-it (revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, Apache-2.0, FP8). They were rule-filtered but not graded, and 1,377 were kept.

`training/fifth/filter.py` removes a dialogue when a user turn overlaps a held-out file, when a reply has markdown or an emoji, when a reply runs over 60 words or three sentences, when the dialogue duplicates another, or when a reply is in the wrong language. The held-out files are `tests/transfer_conversations.json`, `tests/v4_conversations.json`, `tests/conversations.json`, `tests/dialogue.json`, `tests/v5_conversations.json` and `tests/v5_user_conversations.json`. The XQuAD and Belebele questions count as test items too, so the filter compared against 4,368 held-out test turns. Of the 63,000 dialogues, 12,387 were duplicates and 1,582 overlapped a test turn: 884 exactly, 640 by Jaccard similarity and 58 by six-gram. Poems and code use a line-based rule instead: code lines must parse as Python and may not use `import`, `open(` or `while True`.

| Source | Kept, English / Turkish | Total |
|---|---:|---:|
| Dialogues from the 26B teacher, after the filter | 18,451 / 18,694 | 37,145 |
| Programmatic dialogues, after the filter | 6,393 / 6,385 | 12,778 |
| Dialogues from the 31B model, after the filter | 0 / 1,377 | 1,377 |
| Fifth-run set | 24,844 / 26,456 | 51,300 |
| Round-one conversations from 0.4, secondary stream (18,385 rows after preparation) | 19,834 / 15,838 | 35,672 |

Programmatic dialogues are computed in code, 13,500 in all. `training/fifth/identity.py` writes 6,000 covering the name, the maker (Yunus Emre Vurgun), what Franok is, its languages, its limits, greetings and thanks. `training/fifth/affirm.py` writes 1,500 in which the user insists on, doubts or repeats Franok's name and Franok confirms it. `training/fifth/basics.py` writes 6,000 about calendar facts, number order, small sums and everyday knowledge. It leaves out any question close to a held-out test turn and any sum whose two numbers appear together in a test turn, so facts the tests ask for, such as the days in a week, the day after Tuesday and 12 plus 30, are not in it. Typing noise is applied to user turns only. 51,300 dialogues are kept in total, split into 49,650 training rows and 1,650 held-out development rows.

SODA (allenai/soda, CC-BY-4.0, Kim et al. 2023) adds 150 million tokens to the English raw-text stream. `training/fifth/soda.py` writes each dialogue as plain `Speaker: utterance` lines. They are raw text and never assistant turns. Passage-reading examples are the 0.4 short-span set, 138,493 rows.

## Conversation

Two new held-out files and two older ones are scored. Franok 0.5, Franok 0.4 and LFM2-350M (revision `f37d3f5c8c5484bc01dad379a595cf4c68c4e70e`) answer the same scripts. Transcripts are made by `tools/compare_chat.py` with FP32 weights, without the app's rule layers. Every system ran at temperature 0 (`--temperature 0`), top-48, top-p 0.89, repetition penalty 0.55 over the last 32 tokens, a trigram block and at most 64 new tokens. The temperature was chosen on 60 development cases (87 turns) that `tools/dev_cases.py` sampled from the held-out development dialogues, never on the test sets. The blind judge passed 33 of 87 turns on the requirement and 38 on SENSIBLE at temperature 0.58, 37 and 41 at 0.3, and 39 and 42 at 0. [Development cases](dev-cases.json) · [Judge verdicts](judge-dev.json) · [t=0.58](franok05-dev-t58.json) · [t=0.3](franok05-dev-t30.json) · [t=0](franok05-dev-t00.json)

Each reply gets up to three scores.

1. **Keyword criteria.** The scripted keywords of the case, where a turn has them.
2. **RUBRIC.** A blind judge, google/gemma-4-31B-it, reads the conversation up to the reply and the case's written requirement. It does not see which system wrote the reply, and the replies of all systems are shuffled. It answers PASS or FAIL on whether the reply meets the requirement. The judge runs in FP8 with seed 0, at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`.
3. **SENSIBLE.** The same judge, with the same blinding, answers PASS or FAIL on whether the reply is coherent, on the topic of the user's latest message and in the right language.

`tests/v5_conversations.json` has 24 cases, 12 per language, with 50 turns: 25 English and 25 Turkish. They cover identity, small talk, repair, casual typing, follow-ups, facts and topic change. Seven turns per language have keyword criteria. [Judge verdicts](judge-v5.json)

| System | Keyword EN / 7 | Keyword TR / 7 | RUBRIC EN / 25 | RUBRIC TR / 25 | SENSIBLE EN / 25 | SENSIBLE TR / 25 |
|---|---:|---:|---:|---:|---:|---:|
| Franok 0.5 | 3 | 3 | 7 | 7 | 11 | 14 |
| Franok 0.4 | 0 | 1 | 0 | 2 | 3 | 6 |
| LFM2-350M | 2 | 0 | 10 | 0 | 13 | 0 |

Over both languages Franok 0.5 met the requirement on 14 of 50 turns (7 + 7), Franok 0.4 on 2 (0 + 2) and LFM2-350M on 10 (10 + 0). SENSIBLE totals are 25 (11 + 14), 9 (3 + 6) and 13 (13 + 0). Keyword criteria passed on 6 of 14 turns for Franok 0.5, 1 for Franok 0.4 and 2 for LFM2-350M.

Transcripts: [Franok 0.5](franok05-v5-conversations.json) · [Franok 0.4](franok04-v5-conversations.json) · [LFM2-350M](lfm2-v5-conversations.json)

`tests/v5_user_conversations.json` has 10 cases and 22 turns, 11 per language. They are built from failed chats of the project's author: help with maths, "hey how are you", a short Python script, a poem, cooking with eggs and rice, and a chat in which the user says "no you are franok". Three turns per language have keyword criteria. [Judge verdicts](judge-user.json)

| System | Keyword EN / 3 | Keyword TR / 3 | RUBRIC EN / 11 | RUBRIC TR / 11 | SENSIBLE EN / 11 | SENSIBLE TR / 11 |
|---|---:|---:|---:|---:|---:|---:|
| Franok 0.5 | 2 | 1 | 6 | 5 | 7 | 5 |
| Franok 0.4 | 1 | 0 | 4 | 3 | 3 | 3 |
| LFM2-350M | 1 | 0 | 5 | 0 | 8 | 0 |

Over both languages Franok 0.5 met the requirement on 11 of 22 turns (6 + 5), Franok 0.4 on 7 (4 + 3) and LFM2-350M on 5 (5 + 0). SENSIBLE totals are 12 (7 + 5), 6 (3 + 3) and 8 (8 + 0). Keyword criteria passed on 3 of 6 turns for Franok 0.5 and 1 of 6 for each of the others.

Transcripts: [Franok 0.5](franok05-user-conversations.json) · [Franok 0.4](franok04-user-conversations.json) · [LFM2-350M](lfm2-user-conversations.json)

The older sets, `tests/v4_conversations.json` and `tests/transfer_conversations.json`, have 16 two-turn cases each, 32 turns. Only six turns in each file, three per language, carry keyword criteria, so only those are scored here. The 0.4 report graded all 32 turns of each file by one non-blind review, and its figures are not comparable with these.

| System | v4 set EN / 3 | v4 set TR / 3 | Transfer set EN / 3 | Transfer set TR / 3 |
|---|---:|---:|---:|---:|
| Franok 0.5 | 0 | 0 | 0 | 0 |
| Franok 0.4 | 0 | 1 | 0 | 0 |
| LFM2-350M | 1 | 1 | 0 | 0 |

Franok 0.5 passed none of the six keyword turns in either file. Franok 0.4 passed 1 on the v4 set and 0 on the transfer set, and LFM2-350M passed 2 on the v4 set and 0 on the transfer set.

Transcripts: [v4 set, Franok 0.5](franok05-v4-conversations.json) · [v4 set, Franok 0.4](franok04-v4-conversations.json) · [v4 set, LFM2-350M](lfm2-v4-conversations.json) · [transfer set, Franok 0.5](franok05-transfer-conversations.json) · [transfer set, Franok 0.4](franok04-transfer-conversations.json) · [transfer set, LFM2-350M](lfm2-transfer-conversations.json)

## Reading and text

Belebele, free XQuAD and bits per byte use the settings of the 0.4 report, FP32 weights. [Belebele](franok05-belebele.json) · [XQuAD](franok05-xquad.json) · [Bits per byte](franok05-byte.json)

| Measure | Franok 0.5 | Franok 0.4 |
|---|---:|---:|
| Belebele, /240 | 61 | 66 |
| XQuAD exact EN, /64 | 9 | 10 |
| XQuAD exact TR, /64 | 6 | 5 |
| Bits per byte EN ↓ | 1.410 | 1.418 |
| Bits per byte TR ↓ | 1.338 | 1.371 |

Belebele was answered correctly 29 times in English and 32 times in Turkish, out of 120 questions per language. Franok 0.5 is within a few answers of 0.4 on Belebele and XQuAD, which is within the noise of a 64-question set. Its bits per byte is lower in both languages.

The optional passage reader is the 0.3 head, unchanged. On Franok 0.5 decoder states it found 21/192 English and 31/192 Turkish validation spans exactly, against 28 and 35 on 0.4. A 20-step refit did not raise either count, so the head was kept as it was. With FP32 weights it found 10/64 English and 5/64 Turkish XQuAD answers exactly, against 11 and 5 on 0.4. [Reader results](franok05-reader.json)

## Browser and Ubuntu

The Ubuntu app loads Q8 weights (353 MB). The website loads Q8 on desktop-class browsers, meaning no phone user agent and at least 8 GB of device memory, or at least 8 cores where the browser does not report memory (`weightFormat` in `web/config.mjs`). It loads Q4 weights (196 MB) on phones and smaller machines. Both were compared with FP32 on the same validation prefixes, 1,344 target positions in all.

| Runtime | Top token matching FP32 | Change in target NLL |
|---|---:|---:|
| [Q4 WASM](franok05-q4-quant.json) | 86.8% | +0.05006 |
| [Q8 WASM](franok05-q8-quant.json) | 98.4% | +0.00105 |

Q4 file SHA-256 `c8c434bf7b02506f2276aed4abbd1069fc11f73db749f8ea9d5151d969383751`, Q8 file SHA-256 `d473d6447db9c535e1f224152b042631e763fa7b3ac53711f745513930dbff5e`. Belebele, XQuAD and reader scores were not re-measured on the Q4 and Q8 runtimes for 0.5.

The app detects whether each message is English or Turkish (`detectLanguage` in `web/language.mjs`) and switches the reply language to match. Its default sampling temperature is 0, the value chosen on the development cases. Direct model evaluations bypass the app's rule layers.

## Reproduce

The commands below assume a clean checkout at the project root and the data and run folders from the [training guide](../../training.md#fifth-run-everyday-conversation). The judge needs a GPU that vLLM can run with a 31B model.

```sh
# Transcripts; RUN is the selected 0.5 checkpoint folder and RUN04 the 0.4 one
python -m tools.compare_chat --run RUN --cases tests/v5_conversations.json --temperature 0 --out franok05-v5-conversations.json
python -m tools.compare_chat --run RUN04 --cases tests/v5_conversations.json --temperature 0 --out franok04-v5-conversations.json
python -m tools.compare_chat --peer lfm2 --cases tests/v5_conversations.json --temperature 0 --out lfm2-v5-conversations.json
# The same three commands run again with tests/v5_user_conversations.json, tests/v4_conversations.json and tests/transfer_conversations.json

# Reading and text measures
python -m tools.transfer belebele --run RUN --threads 4 --out franok05-belebele.json
python -m tools.transfer xquad --run RUN --threads 4 --out franok05-xquad.json
python -m tools.byte_fluency --run RUN --data DATA --threads 4 --out franok05-byte.json

# Blind judge
python -m tools.judge --cases tests/v5_conversations.json --transcripts franok05=franok05-v5-conversations.json franok04=franok04-v5-conversations.json lfm2=lfm2-v5-conversations.json --model google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --quantization fp8 --seed 0 --out judge-v5.json
python -m tools.judge --cases tests/v5_user_conversations.json --transcripts franok05=franok05-user-conversations.json franok04=franok04-user-conversations.json lfm2=lfm2-user-conversations.json --model google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --quantization fp8 --seed 0 --out judge-user.json
```

## Limits

The test prompts are short everyday chat, which is close to the training distribution by design. A pass on them says the model handles that kind of chat. It does not say the model is knowledgeable.

Franok has 50 million active parameters. Factual knowledge, long answers, real programming and multi-step maths remain out of reach.

A judge model can be wrong, and it belongs to the same model family as the teachers that wrote and graded the training dialogues. The transcripts are published next to the scores so that a reader can check any verdict. The test sets have 22 to 50 turns, so a difference of one or two turns is not a reliable signal. Keyword criteria exist for only a few turns of each set.

The contamination check covers the six held-out files and the XQuAD and Belebele questions. Overlap between training sources, and between training data and internet benchmarks beyond those files, has not been exhaustively audited.

The transcripts show where the model works and where it does not. Greetings, questions about itself, abilities, thanks and simple repair turns usually work in both languages. Facts, sums, rewriting requests and step-by-step advice are often wrong: it answered "There are 120 days in a week" and gave 72 for "12 plus 30". Poems and code are short and often wrong, and it can lose the topic on a follow-up. Advice from the model must not be relied on. LFM2-350M passed no Turkish turn on either new set.

These tests were not rerun for 0.5: the system-prompt choice test, long-session memory, bilingual session search and the Q4 and Q8 runtime benchmark runs.
