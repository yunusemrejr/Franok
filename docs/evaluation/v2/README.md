# Extended Colab release

Franok has 314,061,312 parameters and a 1,024-token neural context. This report uses the same 48 aligned reading questions per language as the first release, plus newly recorded conversations, controlled long-context inputs and numerical comparisons against the unrounded final FP32 merge.

## Training on one NVIDIA L4

| Stage | Optimizer steps | Input positions | Supervised or next-token targets | Elapsed |
|---|---:|---:|---:|---:|
| Continued bilingual pretraining | 12,254 | 150,577,152 | 150,577,152 | 62.7 min |
| Full supervised tuning | 3,358 | 41,263,104 | 20,390,032 | 17.2 min |
| Dialogue LoRA | 1,957 | 24,047,616 | 10,061,806 | 9.4 min |
| Contrastive and supervised LoRA | 1,237 | 15,200,256 | 2,430,712 | 5.9 min |

The extension used 231,088,128 input positions and 183,459,702 unmasked targets in the stages above. The original run contributed another 53,859,151 next-token targets. Positions can repeat across draws. The separate reinforcement stage sampled 11,880 replies containing 101,762 generated tokens and made 1,060 parameter updates in 10.4 minutes. Its selected adapter scale was 0.35.

The additional L4 allocation lasted 115.2 minutes, including setup, evaluation on the accelerator and transfer. Colab confirmed termination at 2026-11-18T11:23:51Z. [Allocation record](allocation.json)

## Reading and short tasks

| Model | Reading EN / 48 | Reading TR / 48 | Short EN / 16 | Short TR / 16 |
|---|---:|---:|---:|---:|
| SmolLM2-360M-Instruct | 19 | 15 | 11 | 9 |
| Franok v0.1 native | 13 | 10 | 10 | 7 |
| Franok v0.2 native FP16 | 12 | 11 | 10 | 8 |
| Franok v0.2 native FP32 | 12 | 11 | 10 | 8 |
| Franok Q4 WASM | 13 | 12 | 9 | 9 |
| Franok Q8 WASM | 12 | 11 | 10 | 8 |

Reading uses the mean log probability per answer character across four alternatives. The short set compares two continuations by mean token log probability. All Franok choice tests use the same short user-only template to preserve the comparison to v0.1. The peer uses its native chat template. The paired reading sample, question IDs, scorer and comparison revision are described in the [original release report](../README.md). These are small tests.

## Actual conversations

Ten scripted sessions have independent English and Turkish prompts. Each response becomes real history for the next user turn. Twelve turns have predeclared keyword or exact-match checks. The remaining turns show free-form behavior without an automatic grade. Identity questions use each model’s own name. Keyword checks can pass an incoherent answer or miss a valid paraphrase. Read the full exchanges before drawing conclusions.

| Model and decoder | English checks | Turkish checks | Transcript |
|---|---:|---:|---|
| Franok v0.1 Q4 | 2/6 | 0/6 | [Raw replies](parent-conversations.json) |
| SmolLM2-360M-Instruct | 5/6 | 0/6 | [Raw replies](smollm2-conversations.json) |
| Franok v0.2 native FP16 | 2/6 | 0/6 | [Raw replies](franok-conversations.json) |
| Franok v0.2 native FP32 | 2/6 | 0/6 | [Raw replies](fp32-conversations.json) |
| Franok v0.2 Q4 | 3/6 | 1/6 | [Raw replies](runtime-q4.json) |
| Franok v0.2 Q8 | 2/6 | 1/6 | [Raw replies](runtime-q8.json) |

### What changed during post-training

These intermediate checkpoints answered the same prompts with the same sampler and seed. The six scored turns per language cover only a few dialogue behaviors. Every row links the actual replies, including unscored turns.

| Native checkpoint | English checks / 6 | Turkish checks / 6 | Short EN / 16 | Short TR / 16 |
|---|---:|---:|---:|---:|
| [After extended pretraining](continued-chat.json) | 2/6 | 0/6 | 9 | 9 |
| [After full supervised tuning](supervised-chat.json) | 2/6 | 1/6 | 10 | 9 |
| [After dialogue and evidence LoRAs](pre-rl-chat.json) | 2/6 | 1/6 | 10 | 8 |
| [After reinforcement learning](franok-conversations.json) | 2/6 | 0/6 | 10 | 8 |

The conversational sampler uses temperature 0.58, top-p 0.89, top-k 48, a repetition penalty of 0.55 on recent tokens and a blocked fourth-order repetition. Its random generator resets to the same seed at the start of every session. Franok receives the application system prompt. The peer receives parallel language instructions under its own name. These runs bypass the local calculator and language notices.

## Precision on the actual runtime

| Format | Held-out tokens | Native FP32 NLL | WASM NLL | Top-token agreement | Mean logit cosine |
|---|---:|---:|---:|---:|---:|
| Q4 | 2,048 | 3.64022 | 3.69136 | 83.2% | 0.99092 |
| Q8 | 2,048 | 3.64022 | 3.64058 | 98.4% | 0.99992 |

| Paired choices | Same reading answer as native FP32 / 96 | Same short answer as native FP32 / 32 |
|---|---:|---:|
| Q4 | 92 | 30 |
| Q8 | 96 | 32 |

The reference reconstructs the unrounded final update from the pre-RL parent and released FP32 adapter. Rounding that reconstruction to FP16 differed from the recorded Colab FP16 checkpoint at 0 of 314,061,312 elements (maximum absolute difference 0). Validation uses 16 seeded 129-token fragments of held-out English and Turkish text and chat, with every next position scored. Generated answers can diverge even when their next-token probability distributions are close.

Rebuilding from the published supervised base and all three adapters with the CPU merge tool differed from the Colab checkpoint at 0 of 314,061,312 FP16 values. The largest absolute difference was 0. [Merge comparison](adapter-reconstruction.json)

## Neural recall over longer conversations

An early user message assigns a word to a folder. Unrelated turns intervene before the model chooses one of four words. Correct words are balanced across eight trials per language per depth. Both models receive the same dialogue content at each depth. The bounds below use Franok tokenization, so actual token counts for the peer differ. No session search or attachment participates.

| Model | Budget in Franok tokens | English / 8 | Turkish / 8 |
|---|---:|---:|---:|
| Franok v0.2 | 128 | 4 | 2 |
| Franok v0.2 | 384 | 3 | 2 |
| Franok v0.2 | 960 | 4 | 2 |
| SmolLM2-360M-Instruct | 128 | 8 | 4 |
| SmolLM2-360M-Instruct | 384 | 8 | 4 |
| SmolLM2-360M-Instruct | 960 | 6 | 2 |

The separate session-library test attaches the source from a 1,000-turn transcript. It checks whether the selected text actually reaches the model’s prompt. It makes no claim that the model can reliably recall or interpret that text.

## Reproduction

Prepare the pinned held-out data with `python -m training.data`. Use `tools.eval_inputs` to regenerate the shared choice set. `tools.numeric` generates native references. Run `node tools/measure_runtime.mjs EXPORT q4 REFERENCE OUT.json` and repeat with `q8`. The scripts `tools.reading`, `tools.probe`, `tools.compare_chat` and `tools.long_context` take checkpoint paths and output paths. The peer uses transformers 4.57.6. The [fixed conversation scripts](../../../tests/conversations.json) and all individual numerical and generation results are available alongside this report.
