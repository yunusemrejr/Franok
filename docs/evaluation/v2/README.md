# Extended Franok evaluation

The 1,024-token run starts from Franok’s original random-initialization lineage. The two available intermediate checkpoints show how supervised tuning changes its answers. The automatic keyword checks cover six turns per language and should be read alongside the complete exchanges.

| Checkpoint | English short choices / 16 | Turkish short choices / 16 | Raw dialogue |
|---|---:|---:|---|
| After 150,577,152 additional pretraining targets | 9 | 9 | [Continued pretraining](continued-chat.json) |
| After 20,390,032 supervised targets | 10 | 9 | [Full supervised tuning](supervised-chat.json) |

[Original Franok conversations](parent-conversations.json) · [SmolLM2-360M-Instruct conversations](smollm2-conversations.json) · [Comparison model’s long-context answers](smollm2-long-context.json)

The final reading, conversation, long-context and quantized-runtime measurements will be added when evaluation of the merged reinforcement checkpoint finishes.
