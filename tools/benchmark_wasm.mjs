// Interleave two builds on fixed token streams, with identical weights and prefix lengths.
import fs from 'node:fs';
import path from 'node:path';
import { createEngine } from '../web/engine.mjs';

const [before, after, directory, format, output] = process.argv.slice(2);
if (!output || !['q4', 'q8'].includes(format))
  throw Error('Pass old.wasm new.wasm export-directory q4|q8 report.json');
const weights = fs.readFileSync(path.join(directory, 'model.' + format));
const manifest = JSON.parse(
  fs.readFileSync(path.join(directory, format === 'q8' ? 'model-q8.json' : 'model.json')),
);
const builds = await Promise.all(
  [before, after].map((p) => createEngine(fs.readFileSync(p), manifest, weights)),
);
const texts = JSON.parse(fs.readFileSync('web/tokenizer.json'));
const { Tokenizer } = await import('../web/tokenizer.mjs');
const tok = new Tokenizer(texts);
const fragments = [
  ...tok.encode('Birds have feathers. Kuşların tüyleri vardır. How do small birds keep warm at night? '),
  ...tok.encode('Geceleri soğuktan korunabilmek için hangi özelliklerden yararlanırlar? '),
];
const ids = [
  1,
  5,
  3,
  ...Array.from({ length: manifest.config.window - 3 }, (_, i) => fragments[i % fragments.length]),
];
const lengths = [32, 256, Math.min(896, manifest.config.window - 32)];
const rows = [];
for (const m of builds) for (const token of ids.slice(0, 24)) m.step(token);
for (let repeat = 0; repeat < 3; repeat++) {
  for (const seq of lengths) {
    const entries = [];
    for (const id of repeat % 2 ? [1, 0] : [0, 1]) {
      const m = builds[id];
      m.reset();
      const t = performance.now();
      for (let i = 0; i < seq; i++) m.step(ids[i], false);
      const t2 = performance.now();
      const logits = [];
      for (let i = 0; i < 32; i++) logits.push(m.step(ids[seq + i]).slice());
      entries[id] = { prefillMs: t2 - t, decodeMs: performance.now() - t2, logits };
    }
    let common = 0,
      maximum = 0,
      squares = 0;
    for (let j = 0; j < 32; j++) {
      const a = entries[0].logits[j],
        b = entries[1].logits[j];
      let ix = 0,
        iy = 0;
      for (let k = 0; k < a.length; k++) {
        if (a[k] > a[ix]) ix = k;
        if (b[k] > b[iy]) iy = k;
        const d = a[k] - b[k];
        maximum = Math.max(maximum, Math.abs(d));
        squares += d * d;
      }
      common += Number(ix === iy);
    }
    const row = {
      repeat,
      seq,
      agreement: common,
      tokens: 32,
      maxError: maximum,
      rmse: Math.sqrt(squares / (32 * manifest.config.vocab)),
      before: entries[0].prefillMs,
      after: entries[1].prefillMs,
      decodeBefore: entries[0].decodeMs,
      decodeAfter: entries[1].decodeMs,
    };
    rows.push(row);
    console.log(JSON.stringify(row));
  }
}
fs.writeFileSync(
  output,
  JSON.stringify({ node: process.version, format, weights: manifest.sha256, rows }, null, 2) + '\n',
);
