import { createEngine } from './engine.mjs';
import { Tokenizer } from './tokenizer.mjs';
import { buildPrompt } from './session.mjs';
import { settings } from './config.mjs';
import { sampleToken } from './sampling.mjs';
let model, tok, release;
const tell = (kind, values = {}) => postMessage({ kind, ...values });
async function read(url, label) {
  const r = await fetch(url);
  if (!r.ok) throw Error(label + ' returned HTTP ' + r.status + '.');
  return r;
}
async function load() {
  if (model) return;
  release = await (await read('./release.json', 'Release manifest')).json();
  const format = release.default || 'q4';
  const asset = release.formats?.[format] || { url: release.weights, manifest: './model.json' };
  const [manifest, spec, wasm] = await Promise.all([
    read(asset.manifest, 'Model manifest').then((r) => r.json()),
    read('./tokenizer.json', 'Tokenizer').then((r) => r.json()),
    read('./franok.wasm', 'Runtime').then((r) => r.arrayBuffer()),
  ]);
  const url = new URL(asset.url, location.href).href;
  const cache =
    release.native || typeof caches === 'undefined'
      ? null
      : await caches.open('franok-weights-v1').catch(() => null);
  const stored = await cache?.match(url);
  let bytes;
  if (stored) bytes = await stored.arrayBuffer();
  if (!bytes) {
    const response = await read(url, 'Model download'),
      reader = response.body.getReader();
    bytes = new ArrayBuffer(manifest.bytes);
    const buf = new Uint8Array(bytes);
    let at = 0,
      last = -1;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (at + value.length > buf.length) throw Error('Unexpected model length.');
      buf.set(value, at);
      at += value.length;
      const pct = Math.floor((100 * at) / buf.length);
      if (pct !== last) {
        tell('download', { pct });
        last = pct;
      }
    }
    if (at !== buf.length) throw Error('Incomplete model download. Please retry.');
  }
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (x) =>
    x.toString(16).padStart(2, '0'),
  ).join('');
  if (hash !== manifest.sha256) {
    await cache?.delete(url);
    throw Error('Weight checksum mismatch. Please retry.');
  }
  if (!stored && cache) await cache.put(url, new Response(bytes)).catch(() => {});
  model = await createEngine(wasm, manifest, bytes);
  tok = new Tokenizer(spec);
  tell('ready');
}
onmessage = async ({ data }) => {
  try {
    await load();
    if (data.type === 'load') return;
    model.reset();
    const options = settings(data.options, model.c.window),
      limit = options.maxNew;
    const prompt = buildPrompt(
      tok,
      data.turns,
      data.lang,
      model.c.window - limit,
      options,
      release.systemConditioned !== false,
      data.pins || [],
    );
    tell('prefill', { count: prompt.ids.length, truncated: prompt.truncated });
    let logits;
    for (let i = 0; i < prompt.ids.length; i++)
      logits = model.step(prompt.ids[i], i === prompt.ids.length - 1);
    const generated = [],
      start = performance.now();
    for (let i = 0; i < limit; i++) {
      const token = sampleToken(logits, generated, options);
      if (token === 2) break;
      generated.push(token);
      tell('part', { text: tok.decode(generated).replace(/\uFFFD$/, '') });
      if (i + 1 < limit) logits = model.step(token);
    }
    tell('done', {
      text: tok.decode(generated).trim(),
      tokens: generated.length,
      seconds: (performance.now() - start) / 1000,
      prompt: prompt.ids.length,
      used: prompt.used,
      shortened: prompt.shortened,
      systemTokens: prompt.systemTokens,
      truncated: prompt.truncated,
    });
  } catch (e) {
    tell('error', { message: e.message || String(e) });
  }
};
