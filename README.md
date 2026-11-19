# Franok

Franok is an experimental English–Turkish language model by Yunus Emre Vurgun. It began with random weights and its own 8,192-piece tokenizer. Each of its twelve layers uses one of eight feed-forward banks for each token. It has **314,061,312 total parameters** and **49,820,160 on one token's path**, including the vocabulary head.

[Chat in your browser](https://franok.yunusemrevurgun.com/) · [Download for Ubuntu](https://franok.yunusemrevurgun.com/downloads/ubuntu) · [Hugging Face](https://huggingface.co/yunusemrejr/Franok) · [Watch the 2-minute film](https://youtu.be/XtNThi_pBjM)

## Chat locally

The browser downloads the 196.3 MB Q4 checkpoint on the first message. The Ubuntu app uses the 353.4 MB Q8 checkpoint and caches it for offline use. A WebAssembly SIMD worker runs generation on the device. The Architecture tab has an interactive token-routing diagram, a narrated video and English/Turkish captions.

Choose English or Turkish beside the composer. **Controls** lets you set a session-specific system prompt, temperature, top-p, output limit, retrieval and exact local calculation. With conversation help enabled, a standalone greeting is normalized to “Hello!” or “Merhaba!” and the model generates its reply. Its context details show the normalized input. The labeled Conversation helper handles a request to clarify that greeting. After a substantive question, a clarification retries that question without the confusing answer in the prompt. A custom system prompt or disabling the helper sends these messages directly to the model. Recognized standalone arithmetic expressions have an explicit Calculator attribution. The 1,024-token neural context holds only part of a long conversation. Type `/recall term` or `/hatırla kelime` to quote matching messages you wrote, with a link to each original. Search earlier turns, attach the original passage to a question, pin up to three turns in future prompts, or export the complete transcript. Drafts, conversations and controls stay in the current app or browser profile.

For a bilingual session, turn on **Match older context across English and Turkish** under Controls. A Turkish follow-up can bring an earlier English note back into the model’s input. Expand **Context** beneath the answer to inspect or reopen each selected turn. The app also lets you search explicitly and attach a source yourself.

Install the Debian package on amd64 Ubuntu.

```sh
sudo apt install ./franok_0.5.0_amd64.deb
```

For a source installation, run `npm ci`, `npx install-electron`, then `npm run install:ubuntu -- --pin` and `gtk-launch franok`. Both installations register the same desktop identifier. A second click on the dock icon brings the running window forward. The [GitHub releases](https://github.com/yunusemrejr/Franok/releases) include installers and SHA-256 digests.

## Trace the architecture

At layer two, suppose the last token in the previous eight-token group has ID 17. That anchor gives a rotation of six. Slot three therefore uses bank one, since `(3 + 6) % 8 = 1`. Rotating all eight slots by the same amount gives each bank exactly one token in a full group. Every token still uses causal attention. [Equations and parameter counts](docs/architecture.md)

## Train or inspect the model

The [Colab notebook](https://colab.research.google.com/github/yunusemrejr/Franok/blob/main/training/Franok.ipynb) covers the lineage from random initialization. The third run starts from Franok’s own second checkpoint. It predicts another 383,385,600 English and Turkish text tokens, then trains on conversation turns, passage questions and changes of instruction. Two low-rank stages specialize in dialogue and evidence. An on-policy reinforcement stage samples six replies per prompt. The fourth run fine-tunes the third release in full on 35,672 conversations: teacher-written ones from Qwen3-8B and Qwen3-14B that a grading pass checked, programmatic dialogues with computed answers, and older conversations. A contamination filter removes overlap with the test sets. Short-span reading from SQuAD and SQuAD-TR, replayed conversations and raw text complete the mixture. The fifth run fine-tunes the fourth release in full on 51,300 everyday dialogues: 63,000 written and graded twice by Gemma 4 26B and 1,500 Turkish ones from Gemma 4 31B, of which 38,522 survive the filter, plus 12,778 computed in code. 150 million tokens of SODA transcripts join the English raw text. The architecture page shows how to trace actual tokens, and the training guide gives the loss equations and checkpoint-selection procedure.

- [Training stages, math and reproduction commands](docs/training.md)
- [Open datasets, licenses and attributions](docs/data.md)
- [Release 0.5 results, direct conversations and judge verdicts](docs/evaluation/v5/final.md)
- [Public weights, adapter base, three LoRAs and logs](https://huggingface.co/yunusemrejr/Franok/tree/main)
- [Packaging, Namecheap and Cloudflare deployment](docs/deployment.md)

For the PyTorch reference, download `config.json`, `tokenizer.json` and `model.safetensors` to one directory, install `requirements.txt`, then run `python -m franok.chat DIRECTORY --lang tr`. The reference accepts a custom system instruction and sampling settings. Build the SIMD runtime with `bash runtime/build.sh`.

Franok's code and trained weights are MIT licensed. On 50 held-out conversation turns in English and Turkish, a blind judge found that Franok 0.5 met the written requirement in 7 of 25 English and 7 of 25 Turkish turns, compared with 0 and 2 for Franok 0.4 and 10 and 0 for LFM2-350M. The model answered "12 plus 30" with 72. The prompts are short everyday chat, and with 50 million active parameters, factual knowledge, long answers, real programming and multi-step maths remain out of reach. [Results and direct transcripts](docs/evaluation/v5/final.md) Earlier reports cover the [fourth release](docs/evaluation/v4/final.md), the [first release](docs/evaluation/README.md), the [second release](docs/evaluation/v2/README.md) and the [third release](docs/evaluation/v3/final.md).
