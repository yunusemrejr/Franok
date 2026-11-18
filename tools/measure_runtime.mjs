// Evaluate the actual WebAssembly kernel, shared prompt builder and application sampler.
import fs from 'node:fs';
import path from 'node:path';
import { createEngine } from '../web/engine.mjs';
import { Tokenizer } from '../web/tokenizer.mjs';
import { buildPrompt } from '../web/session.mjs';
import { settings } from '../web/config.mjs';
import { sampleToken } from '../web/sampling.mjs';

const [dir, fmt, numdir, out, mode = 'all', dialogueFile = 'tests/conversations.json'] =
  process.argv.slice(2);
if (!out || !['q4', 'q8'].includes(fmt))
  throw Error('node tools/measure_runtime.mjs EXPORT q4|q8 REFERENCE OUTPUT [all|dialogues|numeric]');
const json = (p) => JSON.parse(fs.readFileSync(p));
const spec = json(path.join(dir, fmt === 'q4' ? 'model.json' : 'model-q8.json'));
const engine = await createEngine(
  fs.readFileSync('web/franok.wasm'),
  spec,
  fs.readFileSync(path.join(dir, 'model.' + fmt)),
);
const tok = new Tokenizer(json(path.join(dir, 'tokenizer.json')));
const report = { format: fmt, weight_sha256: spec.sha256, window: spec.config.window };
const clampStr = (s) =>
  s
    .normalize('NFKC')
    .toLocaleLowerCase('tr')
    .replace(/[^\p{L}\p{N}:.]+/gu, ' ')
    .trim()
    .replace(/[.]$/, '');
function scores(a, target) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  const peak = a[best];
  let total = 0;
  for (const v of a) total += Math.exp(v - peak);
  return [best, peak + Math.log(total) - a[target]];
}
function prefill(ids) {
  engine.reset();
  for (let i = 0; i < ids.length - 1; i++) engine.step(ids[i], false);
  return engine.step(ids.at(-1));
}
function rng(seed) {
  let a = seed;
  return () => {
    a = (Math.imul(a, 1664525) + 1013904223) >>> 0;
    return a / 4294967296;
  };
}
if (mode === 'all' || mode === 'dialogues') {
  const conversations = [];
  for (const sc of json(dialogueFile)) {
    const opts = settings({ system: sc.system || '', maxNew: sc.maxNew ?? 64, temperature: sc.temperature }, spec.config.window),
      turns = [],
      checks = [];
    const random = rng(17941);
    for (const [i, text] of sc.turns.entries()) {
      turns.push({ role: 'user', text });
      const prompt = buildPrompt(
        tok,
        turns,
        sc.lang,
        spec.config.window - opts.maxNew,
        opts,
        spec.config.window >= 1024,
      );
      let logits = prefill(prompt.ids);
      const gen = [];
      const start = performance.now();
      for (let g = 0; g < opts.maxNew; g++) {
        const id = sampleToken(logits, gen, opts, random);
        if (id === 2) break;
        gen.push(id);
        if (g + 1 < opts.maxNew) logits = engine.step(id);
      }
      const answer = tok.decode(gen).trim();
      turns.push({ role: 'assistant', text: answer });
      const match = sc.expect[i],
        norm = clampStr(answer);
      const passed =
        match === null
          ? null
          : match.some((x) => sc.verbatim ? answer === x : (sc.exact ? norm === clampStr(x) : norm.includes(clampStr(x))));
      checks.push({
        tokens: gen.length,
        prompt_tokens: prompt.ids.length,
        decode_ms: Math.round(performance.now() - start),
        criterion: match,
        passed,
      });
    }
    conversations.push({ id: sc.case, lang: sc.lang, turns, checks });
  }
  report.conversations = conversations;
}
if (mode === 'all') {
  const items = [];
  const summary = {};
  for (const c of json('work/measurement/cases.json')) {
    const header = [1, c.lang === 'en' ? 5 : 6, 3, ...tok.encode(c.q), 4];
    const first = Float32Array.from(prefill(header));
    // Causal KV entries below the branch position are immutable. Each candidate overwrites its tail.
    const state = { pos: engine.pos, anchor: engine.anchor, last: engine.last };
    const vals = c.choices.map((choice) => {
      Object.assign(engine, state);
      const ids = tok.encode(choice);
      let logp = 0,
        logits = first;
      for (let n = 0; n < ids.length; n++) {
        logp -= scores(logits, ids[n])[1];
        if (n + 1 < ids.length) logits = engine.step(ids[n]);
      }
      return logp / (c.suite === 'reading' ? [...choice].length : ids.length);
    });
    const pred = vals.indexOf(Math.max(...vals)),
      key = c.suite + '-' + c.lang;
    const s = (summary[key] ??= { correct: 0, count: 0 });
    s.count++;
    if (pred === c.gold) s.correct++;
    items.push({
      suite: c.suite,
      lang: c.lang,
      key: c.key,
      gold: c.gold,
      pred,
      scores: vals.map((x) => +x.toFixed(5)),
    });
  }
  report.choices = { summary, items };
}
if (mode === 'all' || mode === 'numeric') {
  const rows = json(path.join(numdir, 'numeric.json')),
    bytes = fs.readFileSync(path.join(numdir, 'reference.f32'));
  const fp = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
  let offset = 0;
  const slices = {};
  for (const r of rows) {
    engine.reset();
    const s = (slices[r.lang + '-' + r.source] ??= {
      tokens: 0,
      native_nll: 0,
      wasm_nll: 0,
      agreement: 0,
      cosine: 0,
    });
    for (let j = 0; j + 1 < r.ids.length; j++) {
      const a = engine.step(r.ids[j]),
        b = fp.subarray(offset, offset + a.length);
      offset += a.length;
      const [amax, lossA] = scores(a, r.ids[j + 1]),
        [bmax, lossB] = scores(b, r.ids[j + 1]);
      let dot = 0,
        aa = 0,
        bb = 0;
      for (let k = 0; k < b.length; k++) {
        dot += a[k] * b[k];
        aa += a[k] ** 2;
        bb += b[k] ** 2;
      }
      s.tokens++;
      s.native_nll += lossB;
      s.wasm_nll += lossA;
      s.agreement += Number(amax === bmax);
      s.cosine += dot / Math.sqrt(aa * bb);
    }
  }
  if (offset !== fp.length) throw Error('Reference shape mismatch');
  for (const s of Object.values(slices))
    for (const k of ['native_nll', 'wasm_nll', 'agreement', 'cosine']) s[k] /= s.tokens;
  report.numeric = slices;
}
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(report, null, 2) + '\n');
console.log(out, report.choices?.summary, report.numeric);
