# Training data

The release contains original code, tokenizer and trained parameters under MIT. Each dataset retains its original license. The loader in `training/data.py` pins the exact upstream revisions and filenames. Corpus text is downloaded directly from its publisher. The weights package distributes source references and aggregate counts.

| Source | Material used | Upstream license |
|---|---|---|
| [Wikimedia Wikipedia via Hugging Face](https://huggingface.co/datasets/wikimedia/wikipedia) | First shard of the 2023-11-01 English and Turkish dumps, up to 95M characters per language | CC BY-SA 3.0 / GFDL as listed on the snapshot |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | English and Turkish replies linked to their prompts, filtered for language, deletion, synthetic flag and rank | Apache 2.0 |
| [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | English context, instruction and response | CC BY-SA 3.0 |
| [merve/turkish_instructions](https://huggingface.co/datasets/merve/turkish_instructions) | Turkish instruction, context and response fields | Apache 2.0 on dataset card |
| [GoktugD Turkish Extractive QA](https://huggingface.co/datasets/GoktugD/turkish-extractive-qa-1.5m) | First 16,000 rows of the first training shard, including passage and question | CC0 1.0 |

Exact duplicate records are removed. A deterministic hash assigns approximately 1% of groups to validation. Groups use article IDs, conversation tree IDs, instruction text or extractive source passages, depending on the dataset. English and Turkish have equal sampling probability. Historical documents, quotations, proper names and code can include additional languages. The language selector conditions generation and users can receive mixed-language or inaccurate text.

The tokenizer uses only training partitions and starts with all 256 byte values. The first training pipeline started from random weights and called no teacher or pretrained tokenizer. The fourth run adds teacher-written conversations, described in [Fourth run](#fourth-run). The fifth run adds conversations written by two Gemma models, programmatic dialogues and SODA transcripts, described in [Fifth run](#fifth-run). The separately downloaded comparison model is used only by the evaluation script.

## Extended training

The second run adds the sources below. Its preparation scripts pin every input file and revision. The model mixes English and Turkish within training batches. Published upstream texts and translations include multiple authors and may contain mistakes. The original-data generators and processing code are MIT licensed. Upstream works retain their listed licenses.

| Source | Use | Upstream license |
|---|---|---|
| [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | New English and Turkish shards, lead-sentence prompts | CC BY-SA 3.0 / GFDL as listed on the snapshot |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | OpenStax-inspired synthetic text and stories | Apache 2.0 |
| [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | English instructions and multi-turn dialogues | Apache 2.0 |
| [OpenOrca Turkish](https://huggingface.co/datasets/malhajar/OpenOrca-tr) | Parallel English and Turkish instructions, translation | MIT as listed on the dataset card |
| [Aya](https://huggingface.co/datasets/CohereForAI/aya_dataset) | English and Turkish human-authored instructions | Apache 2.0 |
| [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) and [Turkish extractive QA](https://huggingface.co/datasets/GoktugD/turkish-extractive-qa-1.5m) | Passage-grounded short answers | CC BY-SA 4.0 and CC0 1.0 respectively |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | English training arithmetic | MIT |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | English multi-turn supervised data | MIT |
| [Marul-SFT-TR by Ahmet Karakuş (Marul AI)](https://huggingface.co/datasets/MarulAI/Marul-SFT-TR) | Native Turkish multi-turn dialogues, assistant text only | Apache 2.0. [NOTICE](licenses/marul-NOTICE.txt) and [license](licenses/marul-LICENSE.txt) |
| [Everyday Conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) and [everyday-conversations-tur](https://huggingface.co/datasets/SoAp9035/everyday-conversations-tur) | Short, ordinary English and Turkish follow-ups | Apache 2.0 |
| [Turkish Chat SFT v3](https://huggingface.co/datasets/serhatkagans/turkish-chat-sft-v3) | Filtered Turkish conversational, general knowledge, writing and culture examples | Apache 2.0 |
| [Franok Session Tasks](https://huggingface.co/datasets/yunusemrejr/Franok-Session-Tasks) | Original bilingual reasoning tasks, symbolic rewards, plus separate practice dialogues about names, corrections and changed plans | MIT |

The general text loader groups derivatives of one article or parallel example into the same internal split. Multi-turn preparation holds out groups before expanding turns, filters reasoning and tool traces from selected dialogues, and includes ordinary body text for system instructions. Deduplication operates within the processing source. Cross-source overlap and overlap with internet benchmarks have not been exhaustively audited. The arithmetic/recall task validation split has its own generator seed. The conversation benchmark is a separate set of manually specified prompts and criteria. Downloaded dataset shards and user chat histories are excluded from the public source tree.

The pinned record loaders are [general text and instructions](../training/extended/prepare.py), [multi-turn dialogue](../training/extended/dialogue.py), and [everyday conversation](../training/extended/conversation_data.py).

## Third-run sources

The third run continues the project's v0.2 weights. Equal numbers of English and Turkish prediction positions appear in every natural-text batch. The preparation code groups web pages by URL for validation and removes exact duplicate documents as it reads each source. The second web segment resumes after the last accepted document in the first segment. A source's own training split supplies the supervised examples. FineWeb and stories remain separate corpora in the receipts.

| Publisher and corpus | Use | Upstream license |
|---|---|---|
| [Hugging Face FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | English educational web text and Turkish web text filtered to language confidence of at least 0.75 | ODC-By 1.0 |
| [TinyStories by Eldan and Li](https://huggingface.co/datasets/roneneldan/TinyStories) and [Turkish TinyStories by esat-krky](https://huggingface.co/datasets/esat-krky/TinyStories_Turkish) | Short synthetic narratives in each language | CDLA-Sharing 1.0 |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) and [SQuAD-TR by Budur and colleagues](https://tulap.cmpe.bogazici.edu.tr/entities/dataset/5b23d251-715c-49fa-82fd-8acb87ea43b6) | Short passage answers, unanswerable questions and alternatives from the same paragraph. SQuAD-TR uses machine translations. | CC BY-SA 4.0 |
| [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) | English reviewed human conversation trees, including ranked alternative responses. The Turkish subset surviving the complete-tree filter was too small for the sampler. | Apache 2.0 |
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) | English questions, correct concepts and labeled alternatives | MIT as listed by the dataset publisher |
| [cisimcik Turkish Chat Max 25K](https://huggingface.co/datasets/cisimcik/turkish-chat-max-25k) | Turkish-only synthetic conversations across 13 task categories. Only the final assistant text enters the targets. | Apache 2.0 |

The supervised sampler also uses the earlier open dialogue and instruction collections listed above. For suitable follow-ups it contrasts the current reply with an earlier reply to a different question in the same conversation. The two replies come from the same language and conversation. High-overlap questions, near-duplicate answers and large answer-length disparities are excluded. These derived examples can still have ambiguous alternatives. [Source preparation](../training/third/instructions.py) · [Follow-up selection](../training/third/followups.py)

Another 30,834 English and 39,746 Turkish training turns contain concise replies of at most 100 words. The English sample comes from the second UltraChat training shard. The Turkish sample uses Marul records starting at source row 60,000. Both use the same first-question split rule as the earlier dialogue importer and keep up to three complete user/assistant exchanges. [Short-reply selection](../training/third/short_dialogue.py)

The bilingual session-search encoder also uses English–Turkish translations written by Tatoeba contributors, distributed through OPUS and Sentence Transformers. It uses only the English–Turkish Tatoeba training file at revision `767e169873643dc102321cbb83bf5798efdc7d0f`. Of 220,000 distinct English sentence keys, 214,429 went into training and 5,571 into validation. The sentence translations train the search table and never enter the decoder's language-model loss. [Tatoeba data and attribution terms](https://tatoeba.org/en/downloads) · [Pinned distribution](https://huggingface.co/datasets/sentence-transformers/parallel-sentences/tree/767e169873643dc102321cbb83bf5798efdc7d0f/Tatoeba)

## Fourth run

The fourth run adds the sources below to the 0.3 weights. Teacher-written conversations are pinned to exact model revisions. The programmatic generators and filters are MIT licensed like the rest of the repository.

| Source | Use | Upstream license |
|---|---|---|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) | English teacher-written conversations in the round-1 set, graded and filtered | Apache 2.0 |
| [Qwen3-14B, FP8](https://huggingface.co/Qwen/Qwen3-14B/tree/40c069824f4251a91eefaf281ebe4c544efd3e18) | Turkish teacher-written conversations in the round-1 set, and all teacher-written conversations in round 2 | Apache 2.0 |
| Programmatic dialogues, [`training/fourth/dialogues.py`](../training/fourth/dialogues.py) | Object tracking, one-word label choices, clarifying questions, and counting with unit arithmetic. 5,405 kept, with each answer computed in code | MIT, this repository |
| Round-1 arithmetic, [`training/fourth/arithmetic.py`](../training/fourth/arithmetic.py) | 3,000 conversations computed in code, 1,500 per language | MIT, this repository |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) training file and [SQuAD-TR](https://tulap.cmpe.bogazici.edu.tr/entities/dataset/5b23d251-715c-49fa-82fd-8acb87ea43b6) training file | Short-span reading. Answerable questions only, with the SQuAD v1.1 training questions included in the SQuAD 2.0 file. Passages sharing an 8-gram with XQuAD are removed | CC BY-SA 4.0 |

Teacher and programmatic conversations pass the same filter. Malformed output, wrong-language turns, duplicates and overlong turns are dropped. A conversation is also dropped when a user turn has a content-word Jaccard similarity of at least 0.3 with any held-out test turn, or shares any 6-gram with one. The held-out files are `tests/transfer_conversations.json`, `tests/v4_conversations.json`, `tests/conversations.json` and `tests/dialogue.json`. XQuAD and Belebele questions count as test items too. The teacher grades each conversation twice, and only conversations that both passes mark GOOD are kept.

The commands and counts are in the [evaluation report](evaluation/v4/final.md#reproduce).

## Fifth run

The fifth run adds the sources below to the 0.4 weights. Teacher-written conversations are pinned to exact model revisions. The seeds, programmatic generators and filter are MIT licensed like the rest of the repository.

| Source | Use | Upstream license |
|---|---|---|
| [Gemma 4 26B A4B instruct](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52) | Writes 63,000 dialogues in three batches (48,000 from ten general families, 15,000 from create, code and math) and grades every parsed dialogue twice. 37,145 are kept | Apache 2.0 |
| [Gemma 4 31B instruct, FP8](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475) | Writes 1,500 Turkish dialogues, rule-filtered but not graded. 1,377 are kept | Apache 2.0 |
| Identity dialogues, [`training/fifth/identity.py`](../training/fifth/identity.py) | 6,000 dialogues computed in code: name, maker, what Franok is, languages, limits, greetings and thanks, with typing noise on user turns only | MIT, this repository |
| Name confirmation, [`training/fifth/affirm.py`](../training/fifth/affirm.py) | 1,500 dialogues computed in code in which the user insists on, doubts or repeats Franok's name and Franok confirms it | MIT, this repository |
| Everyday basics, [`training/fifth/basics.py`](../training/fifth/basics.py) | 6,000 dialogues computed in code about calendar facts, number order, small sums and everyday knowledge. Questions close to a held-out test turn, and sums whose two numbers appear together in a test turn, are left out | MIT, this repository |
| [SODA](https://huggingface.co/datasets/allenai/soda) by Kim et al. (2023) | 150 million tokens of dialogue written as plain `Speaker: utterance` transcripts into the English raw-text stream by [`training/fifth/soda.py`](../training/fifth/soda.py). The transcripts are raw text and never assistant turns | CC BY 4.0 |

The 35,672 conversations of the fourth run, the SQuAD and SQuAD-TR reading examples and the raw text of earlier runs remain in the mixture under the licenses listed above.

The thirteen seed families are in [`training/fifth/seeds.py`](../training/fifth/seeds.py): smalltalk, facts, repair, explain, howto, feelings, opinions, followup, writing, unknowns, create, code and math. Seven user typing styles occur: clean, lowercase, typos, abbreviations, Turkish typed without Turkish letters, very short and mixed. Assistant replies are always written cleanly. Turkish seeds are written in Turkish. The teacher grades each dialogue twice, with a standard and a strict prompt, and a dialogue is kept only when both say GOOD. Of the 63,000 dialogues from the 26B model, 1,939 were malformed and the grader flagged 8,479.

The filter, [`training/fifth/filter.py`](../training/fifth/filter.py), drops a dialogue when a user turn overlaps a held-out test file, when a reply contains markdown or an emoji, when a reply is longer than 60 words or three sentences, when the dialogue duplicates another, or when a reply is in the wrong language. Poems and code have a line-based rule: code lines must parse as Python and may not use `import`, `open(` or `while True`. The held-out files are `tests/transfer_conversations.json`, `tests/v4_conversations.json`, `tests/conversations.json`, `tests/dialogue.json`, `tests/v5_conversations.json` and `tests/v5_user_conversations.json`. XQuAD and Belebele questions count as test items too. The filter compared against 4,368 held-out test turns and dropped 12,387 duplicates and 1,582 dialogues that overlapped a test turn (884 exactly, 640 by Jaccard similarity, 58 by six-gram). The 26B model's dialogues kept number 18,451 in English and 18,694 in Turkish, 37,145 in all. The 13,500 programmatic dialogues leave 12,778 (6,393 English, 6,385 Turkish), and the 31B model's leave 1,377 Turkish. The fifth-run set holds 51,300 dialogues, of which 49,650 are training rows and 1,650 are held-out development rows.

The commands and counts are in the [evaluation report](evaluation/v5/final.md#reproduce).
