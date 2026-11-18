import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createEngine } from '../web/engine.mjs';
const dir = 'work/parity/export';
const args = [
  fs.readFileSync('web/franok.wasm'),
  JSON.parse(fs.readFileSync(dir + '/model.json')),
  fs.readFileSync(dir + '/model.q4'),
];
const [cold, warm] = await Promise.all([createEngine(...args), createEngine(...args)]);
const base = Array.from({ length: 26 }, (_, i) => (51 * i + 7) % 380);
const prefill = (m, ids, start) => {
  let logits;
  for (let i = start; i < ids.length; i++) logits = m.step(ids[i], i === ids.length - 1);
  return logits.slice();
};
for (const n of [0, 1, 7, 8, 9, 15, 16, 17, 24, 26]) {
  warm.reset();
  prefill(warm, base, 0);
  const variant = [...base.slice(0, n), 491, 209, 210].slice(0, 30);
  const reused = warm.reusePrefix(variant, variant.length - 1);
  assert.equal(reused, n);
  cold.reset();
  assert.deepEqual(prefill(warm, variant, reused), prefill(cold, variant, 0));
  assert.deepEqual(warm.step(72), cold.step(72));
  const same = warm.reusePrefix(variant, variant.length - 1);
  assert.equal(same, variant.length - 1);
  cold.reset();
  assert.deepEqual(prefill(warm, variant, same), prefill(cold, variant, 0));
}
console.log(
  'Cached and fresh WASM logits match exactly across 10 prefix lengths, changed branches and identical prompts.',
);
