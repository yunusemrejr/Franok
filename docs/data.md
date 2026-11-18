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
