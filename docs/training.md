# Training Franok

Franok began with an empty tokenizer vocabulary and randomly initialized decoder weights. The first Colab run fitted the byte-pair vocabulary to English and Turkish training text, trained a 256-position decoder and taught it the user/assistant turn format. The extended run resumed that exact Franok checkpoint and its tokenizer at [revision 7633d06](https://huggingface.co/yunusemrejr/Franok/tree/7633d06180ebbc836af02c590f96f8ac28a4829a). A third-party model supplied neither starting parameters nor runtime answers. Several open corpora contain previously generated synthetic examples.

## What a training example looks like

The integer markers for beginning, language, user, assistant and end are recorded in the tokenizer. A system-conditioned Turkish example follows this order.

```text
<BOS> <TR> system instruction <END>
<USER> first question <ASSISTANT> first reply <END>
<USER> follow-up <ASSISTANT> second reply <END>
```

Pretraining predicts the next piece at every included position. Instruction tuning shows the model the prefix and applies the loss to the target assistant reply and its end marker. The same prompt order is used in the web and Ubuntu chat workspaces.

## Optimization stages

1. **Build the starting model.** Fit a new 8,192-piece byte-level tokenizer using training partitions. Initialize all 314,061,312 model parameters. The first release used 3,200 initial steps, 4,800 continuation steps and 1,200 assistant-supervised steps, with 24 sequences of length 256 per step.
2. **Read more text.** Continue next-token training with equal English and Turkish rows in each batch. Mix reference text, stories, instruction examples and reasoning tasks. Train first at 256, then 512 and finally 1,024 positions. Larger rotary-position tables add context without adding learned weights.
3. **Tune complete conversations.** Update all weights on English and Turkish dialogue, short conversational exchanges, correction examples, reading and some continued text. Inputs include system instructions, previous turns and natural follow-up questions. The fraction of raw text rows in supervised training is 12%.
4. **Tune two low-rank adapters.** Freeze the main weights. Fit a rank-32 dialogue adapter on assistant targets. Merge it and fit a rank-16 evidence adapter. The latter combines assistant loss with contrastive updates that compare correct answers to different answers from the same language and source.
5. **Learn from sampled replies.** Freeze the merged policy and attach a fresh rank-16 adapter. Sample six answers to each bilingual task using the model's ordinary token vocabulary. A deterministic verifier compares the generated text with the known answer. Groups with identical rewards give no relative signal and skip the policy update. Select a scale for the trained adapter on separate validation examples, then merge it for inference.
6. **Export the released checkpoint.** Store the merged model in FP16. Pack a four-bit and an eight-bit variant, retaining FP32 norm parameters and one scale per 32 matrix entries. Save a SHA-256 digest, tensor shapes and byte offsets for each file.

## The objectives

With correct token $y$ and output scores $z$, the probability of $y$ is $p_y=\exp(z_y)/\sum_j\exp(z_j)$. A training position contributes $-\log p_y$. Giving the observed token probability 0.5 incurs a loss of about 0.69 natural-log units. At probability 0.1 it incurs about 2.30. AdamW follows the gradient of the average loss on unmasked positions. Stage learning rates warm up and then decay on a cosine schedule.

For a matching or mismatched response, let $s$ be its mean log probability per answer token. The preference batches minimize

$$-s_{good} - 0.45\log\sigma(2(s_{good}-s_{foil})-0.35).$$

The foil generator samples from the same data source and favors a similar length. These are synthetic preferences. Some otherwise valid responses can be used as foils.

A low-rank update replaces $W$ with $W+(\alpha/r)BA$. Here $A$ has $r$ output features and $B$ maps them back to the projected width. Each layer adapts the attention input/output and SwiGLU up/down projections. All eight feed-forward banks receive the same low-rank delta for a given projection in a layer. Their original weights remain independent. The release includes the pre-adapter base and the three adapters in application order.

For reinforcement, let $A_i=(R_i-\bar R)/(\mathrm{std}(R)+10^{-4})$ be the reward advantage of completion $i$ in its group. For each generated token let $u=\exp(\log p_\theta-\log p_{old})$ and $d=\log p_{ref}-\log p_\theta$. Minimize the mean over real output tokens of

$$-\min(uA_i,\mathrm{clip}(u,0.8,1.2)A_i)+0.06(\exp(d)-d-1).$$

The reference policy has the reinforcement adapter disabled. The initial pretraining validation mix differs from the dedicated conversational mix used after pretraining. The post-training process repeats validation on its starting checkpoint to establish a comparable baseline. Exact normalized answers score 1. Partial word overlap scores up to 0.35, and a numeric positional score gives limited credit on some same-length numbers. This verifier measures its specified tasks. Each rollout group receives a single update, so ratio clipping is present in the objective but is normally inactive at the start of that update. The KL penalty constrains drift from the frozen pre-RL policy over successive updates.

## Reproduce or inspect a run

The [Colab notebook](../training/Franok.ipynb) builds the entire lineage. To repeat just the extended stages from the project's published first checkpoint, start with a fresh checkout and run the following from its root.

```sh
python -m pip install -r requirements.txt
python -m training.extended.pipeline --lm-minutes 63 --post-minutes 34 --rl-minutes 11
```

The same entry point accepts `--scratch-run work/run --tokenizer work/data/tokenizer.json` after the initial from-scratch commands. Keep separate clean checkouts for different tokenizer runs. Stage budgets are wall-clock limits. Hardware speed, data loading and scheduling affect the step counts. Initialize and train on a BF16-capable CUDA GPU. Native inference and the WASM exporter also run on CPU.

The final checkpoint, the pre-adapter base, tokenizer, adapters and raw training receipts are on the [model hub](https://huggingface.co/yunusemrejr/Franok/tree/main). Copy the `adapters/` directory with its subdirectories into a working folder. Reassemble the final FP16 weights with `python -m franok.merge FOLDER merged.safetensors`. The [evaluation report](evaluation/v2/README.md) records the released run and direct conversation tests.

## Third Colab run

The third run begins with Franok's [v0.2 weights and original tokenizer](https://huggingface.co/yunusemrejr/Franok/tree/db3e8a10bd3fcefde40acdc25ba7c6dc59cde32b). The main decoder keeps its 12 layers, eight banks and 1,024-position window. The new [training sources](data.md#third-run-sources) include educational and general web pages, short stories, passage questions and dialogues with ranked replies.

**Pretraining.** Each batch contains the same number of English and Turkish positions. A random window is sampled uniformly over valid start offsets in the prepared language shards. The run starts at 256 positions, then mixes 512- and 1,024-position sequences. The peak learning rate is 0.00023 with a 120-step warm-up. Periodic checkpoint comparisons measure both languages on fixed held-out windows.

**Instruction tuning.** A row contains one complete dialogue prefix and its answer. Padding fills the rest of the row. The loss starts at the assistant marker and includes the final stop token. A long response and a brief response contribute through two averages. Half the supervised loss is the mean over all answer tokens. Half is the mean of the separate loss for each reply. This gives short answers a say in the update. Occasional raw-text batches help preserve next-token prediction.

**Learning relevance.** Preference batches compare two responses to exactly the same prompt. Some alternatives are ranked OpenAssistant replies. Others are answers to different questions about the same passage, or an earlier reply from the same dialogue. If $L_+$ and $L_-$ are the mean answer-token losses for the preferred and rejected replies, the loss is

$$L_+ - 0.4\log\sigma\big(2(L_--L_+)-0.25\big).$$

A rank-32 dialogue LoRA and a rank-16 evidence LoRA follow full-weight tuning. Held-out instruction loss and pairwise ranking select the scale of each adapter before merging. The from-scratch decoder's tensors keep the same shapes as the original release.

**Following system instructions.** Some minibatches reuse a real passage question and two candidate answers. The system asks for a letter, the answer text, or the whole labeled option. Both the option order and the wording vary. The rejected response either chooses the wrong answer or gives the right answer in the wrong format. The application passes the user's session-level system prompt into the same role position. [Choice sampler](../training/third/controls.py)

**Learning to quote a passage.** A separate 96-feature readout learns from answer offsets and impossible questions in the English and Turkish SQuAD training articles. Every fifth supervised step trains the readout on detached decoder states with its own optimizer. After the reinforcement merge, the readout adapts to the frozen final checkpoint. Development questions tune a single cutoff for the difference between the best span and the no-answer score. The cutoff must still return an answer on at least 70% of the answerable development examples in each language. A separate XQuAD test measures whether the resulting quotations are correct on other articles. [Reader objective](../training/third/reader.py) · [Independent test](../tools/span_transfer.py)

**Reinforcement.** A fresh rank-16 adapter samples groups of six ordinary completions. Prompts include short reasoning questions, passage extraction and answer choices in both languages. Rewards come from the generated text. Equal-reward groups skip the optimizer update. The group-relative objective uses a KL coefficient of 0.075. Every fourth successful update adds a small supervised dialogue term. Validation includes the pre-reinforcement policy as a scale candidate.

The recipe can continue the pinned parent or the weights from the previous cells of the Colab notebook. Wall-clock budgets include data preparation. Different hardware and network speeds produce different counts.

```sh
python -m training.third.pipeline --minutes 240
# Or continue your own completed v0.2 run with its own tokenizer.
python -m training.third.pipeline --parent work/extended/final/model.safetensors --tokenizer work/data/tokenizer.json --minutes 240
```

The [evaluation protocol and sample identifiers](evaluation/v3/README.md) were published while pretraining was still running.

**Dialogue reward.** During supervised training, a small separate network learns to distinguish real answers from lower-ranked answers or misplaced earlier replies. It reads detached decoder states at the assistant boundary and over the response. Training includes complete replies and short openings. A held-out test checks both languages. The RL trainer includes natural conversation prompts when the head distinguishes at least 58% of preferred replies in each language across at least 32 pairs. For those prompts the reward combines the frozen preference score (80%) with reference overlap (20%). This test measures ranking and does not establish that a generated answer is correct. [Preference head](../franok/preference.py) · [Training and validation](../training/third/preference.py)

**Bilingual session search.** A CPU training run projected Franok v0.2's own token embeddings to 96 dimensions and trained them for 9,000 contrastive steps. Training used 59,660 aligned SQuAD question/passage records, 43,884 English–Turkish instruction pairs and 214,429 translation pairs from the Tatoeba community. The translations came from the Sentence Transformers distribution via OPUS. Training and validation use separate article or sentence groups. The released 8-bit table comes from step 8,700, which had the lowest held-out loss. [Training source](../experiments/recall/train.py) · [Tatoeba contributor attribution and licensing](https://tatoeba.org/en/downloads) · [Pinned parallel corpus](https://huggingface.co/datasets/sentence-transformers/parallel-sentences/tree/767e169873643dc102321cbb83bf5798efdc7d0f/Tatoeba)
