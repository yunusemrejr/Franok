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
