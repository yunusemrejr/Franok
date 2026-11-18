# Franok third-run evaluation

The third run continues the 314,061,312-parameter release from the pinned [Franok parent](https://huggingface.co/yunusemrejr/Franok/tree/db3e8a10bd3fcefde40acdc25ba7c6dc59cde32b). The comparison model is [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951). Each model uses its own vocabulary and role formatting.

## Passage reading

The public [sample manifest](../../../tests/transfer_ids.json) fixes the inputs before third-run post-training. It contains identifiers only. The evaluator retrieves the official datasets at pinned revisions.

* **Belebele.** Read 120 distinct source passages with English and Turkish questions on the same material, for 240 decisions. The model scores each of four possible answers following the same question. The winner has the highest sum of answer-token log probabilities divided by the answer's character length. One in four is the chance rate. The selection excludes the source links used in both earlier reading probes. [Dataset by Bandarkar and colleagues](https://huggingface.co/datasets/facebook/belebele)
* **XQuAD.** Answer 64 questions per language on matching passage/question pairs. The model receives a language-appropriate instruction to quote the shortest answer and generates up to 24 tokens greedily. Score normalized exact match and word overlap F1. Each question comes from a different English passage. All 1,190 XQuAD IDs were checked against the English and Turkish SQuAD training IDs with zero matches. Web-corpus overlap at the level of individual sentences remains unknown. [Dataset by Artetxe and colleagues](https://huggingface.co/datasets/google/xquad)
* **Conversation.** Run 16 two-turn conversations covering practical requests, ambiguity, preference changes, arithmetic, politeness and follow-up context. Each scenario has a stated rubric. The second prompt follows the model's actual first reply. A coherent answer must satisfy the requested task and make sense in the selected language. Keyword matches on their own can reward fluent but irrelevant text. [Cases and rubrics](../../../tests/transfer_conversations.json)
* **Quantization.** Compare actual WASM next-token distributions against the full-precision merge on 96 source- and language-mixed validation prefixes. Half of each language/length band goes into each reporting partition. The KL comparison uses the 64 highest-probability reference tokens and groups the remaining probability into a single category. This supplies a lower bound on full-vocabulary KL. The production format also receives direct dialogue and multiple-choice checks.

The evaluation scripts run locally. Pass either a Franok directory or the peer flag. Keep output paths distinct for each model. The passage program can resume from its item log.

```sh
python -m tools.transfer belebele --run work/v2/run --out work/eval/parent-reading.json
python -m tools.transfer xquad --peer --out work/eval/peer-extraction.json
python -m tools.compare_chat --cases tests/transfer_conversations.json --run work/v2/run --out work/eval/parent-dialogue.json
```

## Why the data changed

In the previous Turkish extractive training shard, all 99,502 training records and all 498 validation records contained a transaction-like identifier. The target itself was that identifier in 19,910 training examples. The new recipe uses natural passages from SQuAD-TR, answerable and unanswerable questions, multiple task types and multi-turn dialogues. FineWeb text and short stories provide varied English and Turkish next-token training. The third run retains the project's own tokenizer and the full lineage back to random initialization.

## Measurements before the final instruction stages

| Fixed sample | v0.2 parent | After 116,674,560 new natural-text targets |
|---|---:|---:|
| English educational web, first segment | 3.722 | 3.394 |
| English educational web, second segment | 3.740 | 3.399 |
| English stories | 2.969 | 2.205 |
| Turkish web, first segment | 4.383 | 3.721 |
| Turkish web, second segment | 4.215 | 3.620 |
| Turkish stories | 3.533 | 2.342 |

Each number is mean next-token loss in nats on 64 windows of 256 target tokens. These windows come from held-out shards outside the small shard used to select the pretraining checkpoint. Both models saw the same token IDs. Lower is better. The evaluation records the input seed and weight digest. [Text evaluator](../../../tools/text_validation.py)

The v0.2 parent answered 59 of 240 Belebele questions correctly. SmolLM2-360M-Instruct answered 70. On the separate 128-question XQuAD generation test, Franok had zero exact answers. SmolLM2 had twelve in English and zero in Turkish. On the 32 qualitative conversation criteria, Franok passed zero and SmolLM2 passed eight. A single model-assisted review graded the anonymized replies against the published rubrics. These small samples describe the named tests.
