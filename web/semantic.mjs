export async function loadRecallEncoder(tokenizer, spec) {
  const response = await fetch(spec.url);
  if (!response.ok) throw Error('Search weights unavailable.');
  const bytes = await response.arrayBuffer();
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  if (Array.from(digest, b => b.toString(16).padStart(2, '0')).join('') !== spec.sha256)
    throw Error('Search weights failed verification.');
  return new RecallEncoder(tokenizer, spec, bytes);
}

export class RecallEncoder {
  constructor(tokenizer, config, buffer) {
    this.tk = tokenizer;
    this.c = config;
    const bytes = new Uint8Array(buffer),
      { vocab, width } = config;
    if (config.format !== 'franok-recall-i8-v1' || bytes.length !== vocab * (4 + width))
      throw Error('Invalid semantic index weights.');
    this.scales = new Float32Array(bytes.buffer, 0, vocab);
    this.weights = new Int8Array(bytes.buffer, vocab * 4);
    this.cache = new WeakMap();
  }
  encode(text) {
    const s = text.normalize('NFKC').replaceAll('İ', 'i').toLowerCase().trim();
    let ids = this.tk.encode(s);
    if (ids.length > 160) ids = [...ids.slice(0, 96), ...ids.slice(-64)];
    const v = new Float32Array(this.c.width);
    for (const id of ids) {
      const scale = this.scales[id],
        at = id * v.length;
      for (let j = 0; j < v.length; j++) v[j] += scale * this.weights[at + j];
    }
    let size = 0;
    for (const f of v) size += f * f;
    size = Math.sqrt(size) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= size;
    return v;
  }
  turn(t) {
    const old = this.cache.get(t);
    if (old?.text === t.text) return old.v;
    const v = this.encode(t.text);
    this.cache.set(t, { text: t.text, v });
    return v;
  }
  score(query, turn) {
    const v = this.turn(turn);
    let s = 0;
    for (let i = 0; i < v.length; i++) s += query[i] * v[i];
    return s;
  }
}
