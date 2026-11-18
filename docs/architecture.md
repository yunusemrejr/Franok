# Franok

Franok is a decoder-only language model for English and Turkish. Its anchor-rotation rule assigns work to eight independent SwiGLU networks at each of twelve layers. It uses a byte-level bilingual vocabulary fitted from scratch, grouped-query attention, rotary position embeddings, RMS normalization and tied input/output embeddings.

## Routing

Number layers and positions from zero. For position $t$, let $g=\lfloor t/8\rfloor$ and $s=t\bmod 8$. Define $a_g=0$ for the first group and $a_g=x_{8g-1}$ for later groups, where $x$ is the input token ID. Layer $l$ chooses

$$b(l,t)=\left[s+l+\left\lfloor a_g/2^{l\bmod4}\right\rfloor\right]\bmod8.$$

The anchor always precedes the group. Adding a constant modulo eight permutes its eight slots, so every complete group assigns exactly one token to each bank. This holds for repeated tokens and for either language. All assignments can be calculated before the first layer. Training gathers positions in bank order, computes two batched matrix products, and restores sequence order. In autoregressive decoding a token reads only its assigned bank. A partial last group uses a subset. Right-padding to eight in the reference forward pass has no effect on earlier logits.

The dependence on the previous group's last token and the fixed cyclic permutation are the design choices specific to this implementation. Content-based deterministic expert selection and balanced expert training have prior art. The permutation trades content-adaptive specialization for predictable load. [Hash Layers](https://arxiv.org/abs/2106.04426) · [BASE Layers](https://arxiv.org/abs/2103.16716)

## Decoder

Write $R(x)=g\odot x / \sqrt{\operatorname{mean}(x^2)+10^{-5}}$. Each block first computes $u=R(x)$ and projects $u$ into eight query heads and two key/value heads of dimension 64. Four query heads share each KV head. Rotary embeddings rotate adjacent component pairs by angle $t/10000^{2j/64}$. For each head, multiply the softmax of causally masked $QK^T/8$ by $V$. Concatenate heads, project, and add the residual to produce $z$.

For bank $b(l,t)$, the second sublayer is

$$x'=z+W_{d,b}\Big(\operatorname{SiLU}(W_{a,b}R(z))\odot W_{g,b}R(z)\Big).$$

Each bank expands 512 features into 2,048. All projections omit bias. The final norm and token embedding transpose yield 8,192 logits. Pretraining minimizes mean next-token cross entropy. The final supervised stage masks prompt tokens and padding.

| Component | Parameter count |
|---|---:|
| Tied token embeddings | 4,194,304 |
| Attention projections in 12 layers | 7,864,320 |
| Feed-forward banks in 12 layers | 301,989,888 |
| 25 RMS scale vectors | 12,800 |
| **Total** | **314,061,312** |
| **Parameters on one token's path, including the vocabulary head** | **49,820,160** |

The trained context is 256 positions. The application keeps complete saved sessions, retrieves earlier messages by term overlap weighted by inverse document frequency, and bounds their contribution to each prompt. Search returns the original saved turn and lets the user attach it to a new message. An attachment gets prompt budget first and its local citation survives a restart. The source turn stays available in full even when only its first tokens fit the neural window. This application feature can retrieve text older than the neural window. Approximate keyword retrieval can miss paraphrases.

## Browser implementation

The worker streams one checked weight file, verifies SHA-256, and reuses a cached copy where available. Each group of 32 weights has a float32 scale $s=\max(|w|)/7$ and sixteen bytes of packed integer weights. Entries $j$ and $j+16$ share a byte. Each group therefore uses 20 bytes. The WebAssembly SIMD kernel quantizes input activations to signed eight-bit values per group, forms integer dot products and accumulates the rescaled products in float32. Norm scales and KV caches stay in float32. At 256 tokens the twelve-layer KV cache occupies 3 MiB. First-generation latency includes prompt processing. Device throughput varies.

The reference parity check compares every position of a small deterministic PyTorch example with the incremental WASM decoder, including anchor changes and partial groups. The bilingual tokenizer round-trip test includes Turkish characters, punctuation, literal role-marker text, combining marks and emoji.
