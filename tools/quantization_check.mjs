// Check actual WASM logits against FP32 on fixed validation prefixes.
import fs from 'node:fs';
import path from 'node:path';
import { createEngine } from '../web/engine.mjs';

const [directory, bits, casesFile, report] = process.argv.slice(2);
if (!report || !['q4', 'q8'].includes(bits))
  throw Error('Pass an export directory, q4 or q8, cases and report.');
const spec = JSON.parse(
  fs.readFileSync(path.join(directory, bits === 'q4' ? 'model.json' : 'model-q8.json')),
);
const model = await createEngine(
  fs.readFileSync('web/franok.wasm'),
  spec,
  fs.readFileSync(path.join(directory, 'model.' + bits)),
);
const cases = JSON.parse(fs.readFileSync(casesFile));
const metrics = {};
const rows = [];
for (const c of cases) {
  model.reset();
  let nll = 0,
    full = 0,
    kl = 0,
    agree = 0;
  for (let t = 0; t < c.ids.length - 1; t++) {
    const j = t + 1 - c.at;
    const scores = model.step(c.ids[t], j >= 0);
    if (j < 0) continue;
    let peak = 0;
    for (let k = 1; k < scores.length; k++) if (scores[k] > scores[peak]) peak = k;
    let z = 0;
    for (const s of scores) z += Math.exp(s - scores[peak]);
    const logz = Math.log(z) + scores[peak];
    nll += logz - scores[c.ids[t + 1]];
    full -= c.gold[j];
    agree += Number(peak === c.top[j][0]);
    let baseTail = 1,
      quantTail = 1;
    for (let k = 0; k < c.top[j].length; k++) {
      const p = c.probs[j][k],
        lq = scores[c.top[j][k]] - logz;
      kl += p * (Math.log(p) - lq);
      baseTail -= p;
      quantTail -= Math.exp(lq);
    }
    if (baseTail > 1e-9) kl += baseTail * Math.log(baseTail / Math.max(1e-10, quantTail));
  }
  const n = c.gold.length;
  rows.push({ key: c.key, lang: c.lang, group: c.group, source: c.source, n, nll, full, kl, agree });
  for (const group of ['all', c.group, c.lang]) {
    const tab = (metrics[group] ||= { n: 0, modelNll: 0, fullNll: 0, coarseKl: 0, top1: 0 });
    tab.n += n;
    tab.modelNll += nll;
    tab.fullNll += full;
    tab.coarseKl += kl;
    tab.top1 += agree;
  }
}
for (const tab of Object.values(metrics)) {
  for (const key of ['modelNll', 'fullNll', 'coarseKl', 'top1']) tab[key] /= tab.n;
}
fs.writeFileSync(
  report,
  JSON.stringify(
    { format: spec.format, sha256: spec.sha256, scale: spec.scale_method, metrics, rows },
    null,
    2,
  ) + '\n',
);
console.log(metrics);
