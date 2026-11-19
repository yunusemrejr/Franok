# Training Franok

Franok began with an empty tokenizer vocabulary and randomly initialized decoder weights. The first Colab run fitted the byte-pair vocabulary to English and Turkish training text, trained a 256-position decoder and taught it the user/assistant turn format. The extended run resumed that exact Franok checkpoint and its tokenizer at [revision 7633d06](https://huggingface.co/yunusemrejr/Franok/tree/7633d06180ebbc836af02c590f96f8ac28a4829a). In the first three runs, a third-party model supplied neither starting parameters nor runtime answers. Several open corpora contain previously generated synthetic examples. The fourth run trains on conversations written by open teacher models, described in [its section](#fourth-run-teacher-written-conversations).

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

**Dialogue reward.** During supervised training, a small separate network learns to distinguish real answers from lower-ranked answers or misplaced earlier replies. It reads detached decoder states at the assistant boundary and over the response. Training includes complete replies and short openings. A held-out test checks both languages. The RL trainer includes natural conversation prompts when the head distinguishes at least 58% of preferred replies in each language across at least 32 pairs. The preferred replies must include at least eight shorter and eight longer examples per language. Accuracy averaged equally across those two groups must reach 55%, which guards against using length alone to win the comparison. For those prompts the reward combines the frozen preference score (80%) with reference overlap (20%). This test measures ranking and does not establish that a generated answer is correct. [Preference head](../franok/preference.py) · [Training and validation](../training/third/preference.py)

**Bilingual session search.** A CPU training run projected Franok v0.2's own token embeddings to 96 dimensions and trained them for 9,000 contrastive steps. Training used 59,660 aligned SQuAD question/passage records, 43,884 English–Turkish instruction pairs and 214,429 translation pairs from the Tatoeba community. The translations came from the Sentence Transformers distribution via OPUS. Training and validation use separate article or sentence groups. The released 8-bit table comes from step 8,700, which had the lowest held-out loss. [Training source](../experiments/recall/train.py) · [Tatoeba contributor attribution and licensing](https://tatoeba.org/en/downloads) · [Pinned parallel corpus](https://huggingface.co/datasets/sentence-transformers/parallel-sentences/tree/767e169873643dc102321cbb83bf5798efdc7d0f/Tatoeba)

During response tuning, the vocabulary projection and cross-entropy run only at supervised answer positions. Every prefix token still contributes through causal attention. During reinforcement sampling, six candidate replies can share the prefix key/value state and retain completed groups of eight tokens. Each reply keeps its own anchor as soon as the candidates diverge. The gradient pass uses teacher forcing over the full prompt and sampled reply. [Training sampler](../franok/rollout.py) · [Cache equivalence test](../tests/rollout.py)

The released run used 236.6 L4 minutes. Supervised and low-rank training consumed 35,864,087 supervised target positions after continued pretraining. Reinforcement learning sampled 7,338 replies, made 825 optimizer updates and selected merge scale 1.0. [Third-run measurements and receipts](evaluation/v3/final.md)

## Fourth run: teacher-written conversations

Release 0.4 starts from the FP32 merge of 0.3, SHA-256 `b7d97329…`, and trains every weight. The architecture and tensor shapes do not change. The stage has four parts.

1. **Write.** Qwen3-8B (revision `b968826d9c46dd6066d109eabc6255188de91218`) and Qwen3-14B (revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, FP8) write one conversation per seed. Each seed fixes a topic, a task, a persona, a style and a twist. Thinking is disabled. Sampling uses temperature 0.8, top-p 0.95 and top-k 50.
2. **Grade.** The same teacher reads each conversation with a general check and a stricter one. A conversation is kept only when both return GOOD.
3. **Filter.** Malformed output, wrong-language turns, duplicates and overlong turns are dropped. A conversation is also dropped when one of its user turns has a content-word Jaccard similarity of at least 0.3 with a held-out test turn, or shares any 6-gram with one. Programmatic dialogues are computed from code templates and involve no model.
4. **Tune.** Full-parameter AdamW trains on a mixture by target tokens: 55% new conversations, 20% reading examples from SQuAD and SQuAD-TR, 10% replay of 0.3 conversations and 15% raw English and Turkish text. The peak learning rate is 4e-5. Loss is taken on assistant tokens only.

The checkpoint is the one with the lowest development conversation loss among those whose raw-text loss stays within 0.02 nats of the starting model in both languages. The run selected step 768 of 3,940. It took 1,400 seconds. Commands, counts and results are in the [evaluation report](evaluation/v4/final.md).

## Fifth run: everyday conversation

Release 0.5 starts from the 0.4 FP32 weights, SHA-256 `8259f95e…`, and trains every weight. The architecture, tokenizer, passage-reader head and search encoder are unchanged. Release 0.4 was trained only on practical-advice dialogues. In real use it failed at greetings and small talk, questions about itself, casually typed messages, "what do you mean" repair turns, and requests for a poem, a short Python example or help with maths. The fifth run is built around those cases. The stage has five parts.

1. **Write.** Seeds from `training/fifth/seeds.py` cover thirteen families: smalltalk, facts, repair, explain, howto, feelings, opinions, followup, writing, unknowns, create, code and math. Create asks for four-line poems, two-sentence stories, jokes, riddles and short messages. Code asks for one to four lines of plain-text Python plus one sentence. Math asks for an area explained in one or two sentences with one small worked example. Each seed fixes a topic, a persona, a situation and one of seven user typing styles: clean, lowercase, typos, abbreviations, Turkish typed without Turkish letters, very short, and mixed. Assistant replies are always clean. Turkish prompts are written in Turkish, and a dialogue has two to five turns. google/gemma-4-26B-A4B-it (revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, Apache-2.0) runs under vLLM 0.34.0 on one A100 80 GB and was asked for 63,000 dialogues in three batches: 24,000 from the ten general families, 15,000 from create, code and math, and 24,000 more from the ten general families. The third batch was requested as 60,000 and stopped by its 24-minute cap at 24,000. A further 1,500 Turkish dialogues come from google/gemma-4-31B-it (revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, Apache-2.0, FP8). The wrapper is `training/fifth/teacher.py`.
2. **Grade.** The 26B model reads every parsed dialogue twice, once with a standard prompt and once with a strict one, and a dialogue is kept only when both answer GOOD. Of the 63,000 dialogues, 1,939 were malformed and the grader flagged 8,479. The 1,500 dialogues from the 31B model are rule-filtered and not graded; 1,377 were kept.
3. **Filter.** `training/fifth/filter.py` drops a dialogue when a user turn overlaps a held-out test file (the six files in `TEST_FILES`, including the two new v5 sets), when a reply contains markdown or an emoji, when a reply is longer than 60 words or three sentences, when it duplicates another dialogue, or when a reply is in the wrong language. Poems and code use a line-based rule instead: up to six lines, and every code line must parse as Python and may not use `import`, `open(` or `while True`. The filter compared against 4,368 held-out test turns, including the XQuAD and Belebele question lists. Of the 63,000 dialogues from the 26B model, 12,387 were duplicates and 1,582 overlapped a test turn (884 exactly, 640 by Jaccard similarity and 58 by six-gram), and 37,145 were kept: 18,451 English and 18,694 Turkish. With the 12,778 programmatic dialogues and the 1,377 from the 31B model, the set holds 51,300 dialogues. About 3% of them, chosen by a hash of their identifier, are held out as development dialogues, which leaves 49,650 training rows and 1,650 development rows.
4. **Programmatic.** Three scripts compute 13,500 dialogues in code with no model involved, and 12,778 were kept: 6,393 English and 6,385 Turkish. `training/fifth/identity.py` writes 6,000 dialogues about the model's name, its maker Yunus Emre Vurgun, what it is, its languages, its limits, greetings and thanks. `training/fifth/affirm.py` writes 1,500 in which the user insists on, doubts or repeats Franok's name and Franok confirms it. `training/fifth/basics.py` writes 6,000 about calendar facts, number order, small sums and everyday knowledge. It leaves out any question close to a held-out test turn and any sum whose two numbers appear together in a test turn, so facts the tests ask for are not in it. Typing noise is applied to user turns only.
5. **Tune.** Full-parameter AdamW (betas 0.9 and 0.95, weight decay 0.01, gradient clip 1.0) runs on one A100 80 GB with 32,768 padded positions per micro-batch and two micro-batches per step. The cosine schedule is set by wall-clock time and decays to 5% of the peak. The 35,672 round-one conversations from 0.4 stay in as a secondary stream, 18,385 rows after preparation, and the passage-reading set has 138,493 rows. 150 million tokens of SODA dialogues (allenai/soda, CC BY 4.0, Kim et al. 2023) are written by `training/fifth/soda.py` as plain `Speaker: utterance` transcripts into the English raw-text stream. They are raw text and never appear as assistant turns.

Training has two stages. The mixture is by target tokens.

| Stage | Loss on | Peak learning rate | Time | Steps | Fifth-run dialogues | Round one | Passage reading | Raw text |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | every token of a dialogue, user and assistant | 1.5e-4 | 15 minutes | 3,401 | 47% | 10% | 6% | 37% |
| B | assistant tokens only | 3e-5 | 5 minutes | 1,130 | 60% | 10% | 6% | 24% |

A first attempt ran stage A for 68 minutes on a set of 30,323 dialogues. Held-out conversation loss was lowest at the first checkpoint, 1.471 at nine minutes, and rose afterwards to 1.486 and then 1.640, so the model was memorising the set. That run was discarded. The set was enlarged to 51,300 dialogues and the schedule shortened to the times above.

In both stages the checkpoint with the lowest assistant-token loss on the held-out development dialogues is kept, so the two stages and their learning rates are compared on one scale. Stage B starts from the best stage-A checkpoint with a 50-step warm-up. Stage A evaluated a candidate every 2.5 minutes, and the loss went 2.948 at the start, then 1.919, 1.702, 1.598, 1.554, 1.534 and 1.531. Stage B evaluated every 1.25 minutes: 1.531, 1.508, 1.492, 1.490 and 1.489. The released checkpoint is step 1,130 of stage B, with loss 1.489 (English 1.416, Turkish 1.553) and SHA-256 `419f3ad443df455110a8e0bbfbd080c040fb5ee5321f85d0e8a0f89def2f620f`. Over both stages the held-out passage-reading loss went from 1.065 to 1.060, and raw-text loss from 3.590 to 3.578 in English and from 3.749 to 3.673 in Turkish, all in nats per token. Commands, counts and results are in the [evaluation report](evaluation/v5/final.md).

Reproduce the stage with these commands. Each step reads and writes under `data/`.

```sh
# Teacher dialogues: three batches from the 26B model, each graded twice
python -m training.fifth.teacher --model google/gemma-4-26B-A4B-it --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --count 24000 --seed 21 --id-prefix ta --grade --out data/general-a.jsonl
python -m training.fifth.teacher --tasks create code math --model google/gemma-4-26B-A4B-it --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --count 15000 --seed 33 --id-prefix tx --grade --out data/tasks.jsonl
python -m training.fifth.teacher --model google/gemma-4-26B-A4B-it --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --count 60000 --seed 4404 --id-prefix tb --max-minutes 24 --grade --out data/general-b.jsonl

# Turkish dialogues from the 31B model, rule-filtered but not graded
python -m training.fifth.teacher --lang tr --model google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --quantization fp8 --count 1500 --seed 11 --id-prefix tg --out data/turkish.jsonl

# Programmatic dialogues: identity, name confirmation and basic facts
python -m training.fifth.identity --count 6000 --seed 9 --prefix j --out data/identity.jsonl
python -m training.fifth.affirm --count 1500 --out data/affirm.jsonl
python -m training.fifth.basics --count 6000 --out data/basics.jsonl

# Filter: graded dialogues with the programmatic ones, then the ungraded Turkish file on its own
python -m training.fifth.filter data/general-a.jsonl data/tasks.jsonl data/general-b.jsonl --grades data/general-a.grades.jsonl data/tasks.grades.jsonl data/general-b.grades.jsonl --identity data/identity.jsonl data/affirm.jsonl data/basics.jsonl --out data/graded.jsonl --report data/graded-report.json
python -m training.fifth.filter data/turkish.jsonl --out data/turkish-clean.jsonl --report data/turkish-report.json
cat data/graded.jsonl data/turkish-clean.jsonl > data/clean.jsonl

# SODA transcripts into the English raw-text stream
python -m training.fifth.soda --out-dir work/third/data/en --tokens 150000000

# Stage A: loss on every token
python -m training.fifth.train --weights weights/fp32/model.safetensors --expect-sha 8259f95e933c5e021dc29258bb21982532229e574bd7bc29cc83417a14b67738 --clean data/clean.jsonl --bulk data/round-one.jsonl --reading data/reading.jsonl --every-token --shares clean=0.47,bulk=0.10,reading=0.06,raw=0.37 --peak-lr 1.5e-4 --minutes 15 --eval-minutes 2.5 --out run-a

# Stage B: loss on assistant tokens only, from the best stage-A checkpoint
python -m training.fifth.train --weights run-a/best/model.safetensors --clean data/clean.jsonl --bulk data/round-one.jsonl --reading data/reading.jsonl --shares clean=0.60,bulk=0.10,reading=0.06,raw=0.24 --peak-lr 3e-5 --warmup 50 --minutes 5 --eval-minutes 1.25 --out run-b
```

`data/round-one.jsonl` holds the 35,672 round-one conversations from the [fourth run](#fourth-run-teacher-written-conversations), and `data/reading.jsonl` is the short-span reading set from `python -m training.fourth.reading`. Each `--grade` run writes its verdicts next to its output as `<name>.grades.jsonl`.
