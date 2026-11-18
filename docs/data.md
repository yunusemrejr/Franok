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

The tokenizer uses only training partitions and starts with all 256 byte values. The architecture starts from random weights. No teacher or pretrained tokenizer is called by the training pipeline. The separately downloaded comparison model is used only by the evaluation script.

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
