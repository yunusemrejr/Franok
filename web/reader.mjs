// A separate learned head scores contiguous source spans using Franok's normalized decoder states.
const utf8 = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true });

export async function createReadout(wasm, buffer, config) {
  const bytes = new Uint8Array(buffer);
  const size = Number(new DataView(bytes.buffer, bytes.byteOffset).getBigUint64(0, true));
  if (!Number.isSafeInteger(size) || size < 2 || size + 8 > bytes.length)
    throw Error('Invalid passage-reader weights.');
  const header = JSON.parse(decoder.decode(bytes.subarray(8, 8 + size)));
  const { instance } = await WebAssembly.instantiate(wasm, { math: { exp: Math.exp } });
  const w = instance.exports;
  const base = Number(w.__heap_base.value);
  const raw = bytes.subarray(8 + size);
  const inp = Math.ceil((base + raw.length) / 16) * 16;
  const dimension = Math.max(config.width, config.hidden * 2, config.max_span);
  const out = inp + dimension * 4;
  const end = out + dimension * 4;
  w.memory.grow(Math.max(0, Math.ceil((end - w.memory.buffer.byteLength) / 65536)));
  new Uint8Array(w.memory.buffer, base).set(raw);
  const floats = new Float32Array(w.memory.buffer);
  const tensors = new Map();
  for (const [name, tensor] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [a, b] = tensor.data_offsets;
    if (
      tensor.dtype !== 'F32' ||
      !Number.isSafeInteger(a) ||
      !Number.isSafeInteger(b) ||
      a % 4 ||
      a < 0 ||
      b > raw.length ||
      b - a !== tensor.shape.reduce((n, d) => n * d, 4)
    )
      throw Error('Invalid passage-reader tensor ' + name);
    tensors.set(name, {
      shape: tensor.shape,
      ptr: base + a,
      values: floats.subarray((base + a) / 4, (base + b) / 4),
    });
  }
  const tensor = (name) => {
    const x = tensors.get(name);
    if (!x) throw Error('Missing passage-reader tensor ' + name);
    return x;
  };
  function linear(name, x) {
    const mat = tensor(name + '.weight'),
      [rows, cols] = mat.shape;
    if (cols !== x.length || cols > dimension || rows > dimension)
      throw Error('Passage-reader dimensions disagree.');
    floats.set(x, inp / 4);
    w.linear_f32(mat.ptr, inp, tensors.get(name + '.bias')?.ptr || 0, out, rows, cols);
    return floats.slice(out / 4, out / 4 + rows);
  }
  return { config, linear, values: (name) => tensor(name).values };
}

function similarity(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export function scoreSpan(readout, states, offset, legal) {
  const { linear, values, config } = readout,
    { hidden, max_span: maxSpan } = config;
  const q = states[offset - 1];
  const question = linear('question', q);
  const keys = states.slice(3, offset - 2).map((h) => linear('query_token', h).map(Math.tanh));
  const features = states.map((h) => {
    const p = linear('passage_token', h).map(Math.tanh);
    const att = keys.map((k) => similarity(p, k) / Math.sqrt(hidden));
    const peak = Math.max(...att);
    const mass = att.map((v) => Math.exp(v - peak)),
      total = mass.reduce((a, b) => a + b, 0);
    const pair = new Float32Array(hidden * 2);
    for (let i = 0; i < keys.length; i++) {
      const weight = mass[i] / total;
      for (let j = 0; j < hidden; j++) pair[j] += weight * keys[i][j];
    }
    for (let j = 0; j < hidden; j++) pair[hidden + j] = pair[j] * p[j];
    const aligned = linear('fuse', pair),
      product = linear(
        'product',
        h.map((v, j) => v * q[j]),
      );
    return linear('text', h).map((v, j) => Math.tanh(v + question[j] + product[j] + aligned[j]));
  });
  const wide = values('wide.weight'),
    near = values('near.weight');
  const nearBias = values('near.bias'),
    wideBias = values('wide.bias');
  const context = features.map((f, i) => {
    const local = new Float32Array(hidden);
    for (let d = 0; d < hidden; d++) {
      let x = nearBias[d] + wideBias[d];
      for (let k = 0; k < 5; k++) if (features[i + k - 2]) x += near[5 * d + k] * features[i + k - 2][d];
      for (let k = 0; k < 3; k++)
        if (features[i + 4 * (k - 1)]) x += wide[3 * d + k] * features[i + 4 * (k - 1)][d];
      local[d] = x;
    }
    return linear('mix', local).map((v, d) => f[d] + Math.tanh(v));
  });
  const source = context.slice(offset, -1),
    mean = new Float32Array(hidden);
  const starts = source.map((h) => linear('start', h)[0]);
  let best = -Infinity,
    from = -1,
    through = -1;
  source.forEach((h, j) => {
    const end = linear('end', h)[0],
      length = linear('length', h);
    for (let d = 0; d < hidden; d++) mean[d] += h[d] / source.length;
    if (!legal[j + 1]) return;
    for (let delta = 0; delta < Math.min(maxSpan, j + 1); delta++) {
      const i = j - delta;
      if (!legal[i]) continue;
      const val = end + starts[i] + length[delta];
      if (val > best) {
        best = val;
        from = i;
        through = j;
      }
    }
  });
  const none = linear('null', context.at(-1))[0] + linear('absence', mean)[0];
  return { from, through, margin: best - none };
}

export function readPassage(model, tok, head, question, passage, lang, progress = () => {}) {
  const questionIds = tok.encode(question.trim()),
    ids = tok.encode(passage);
  if (!questionIds.length || !ids.length) throw Error('Supply a question and a passage.');
  const prefix = [1, lang === 'tr' ? 6 : 5, 3, ...questionIds, 2, 3];
  const room = model.c.window - prefix.length - 1,
    overlap = head.config.max_span;
  if (room < overlap + 8) throw Error('Shorten the question to leave room for the passage.');
  const raw = utf8.encode(passage),
    offsets = [0];
  for (const id of ids) offsets.push(offsets.at(-1) + tok.pieces[id].length);
  if (offsets.at(-1) !== raw.length) throw Error('Passage offsets do not match the source text.');
  const legal = offsets.map((n) => n === raw.length || (raw[n] & 192) !== 128);
  let start = 0,
    windows = 0,
    best = null;
  while (start < ids.length) {
    let stop = Math.min(ids.length, start + room);
    while (!legal[stop]) stop--;
    model.reset();
    const states = [];
    for (const id of [...prefix, ...ids.slice(start, stop), 4]) {
      model.step(id, false);
      states.push(model.hidden());
      if (states.length % 32 === 0)
        progress(Math.min(ids.length, start + Math.max(0, states.length - prefix.length)), ids.length);
    }
    const { from, through, margin } = scoreSpan(head, states, prefix.length, legal.slice(start, stop + 1));
    if (from >= 0 && (!best || margin > best.margin)) {
      const a = offsets[start + from],
        b = offsets[start + through + 1];
      const text = decoder.decode(raw.subarray(a, b)),
        quote = text.trim();
      if (quote) {
        const startByte = a + utf8.encode(text.slice(0, text.length - text.trimStart().length)).length;
        best = { quote, startByte, endByte: startByte + utf8.encode(quote).length, margin };
      }
    }
    windows++;
    if (stop === ids.length) break;
    start = stop - overlap;
    while (!legal[start]) start--;
  }
  const accepted = !!best && best.margin >= head.config.threshold;
  return {
    accepted,
    windows,
    quote: accepted ? best.quote : '',
    startByte: accepted ? best.startByte : null,
    endByte: accepted ? best.endByte : null,
    margin: best?.margin ?? null,
    tokens: ids.length,
  };
}
