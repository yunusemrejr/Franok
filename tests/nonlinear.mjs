import assert from 'node:assert/strict';
import fs from 'node:fs';
const { instance } = await WebAssembly.instantiate(fs.readFileSync('web/franok.wasm'), {
  math: { exp: Math.exp },
});
const w = instance.exports,
  base = Number(w.__heap_base.value);
w.memory.grow(3);
const mem = new Float32Array(w.memory.buffer);
const n = 1003,
  start = base / 4;
const xs = Array.from({ length: n }, (_, i) => Math.fround(-104 + (208 * i) / (n - 1)));
const gates = xs.map((_, i) => Math.fround(Math.cos(i) + 0.125));
mem.set(xs, start);
mem.set(gates, start + n);
w.swiglu(base, n);
let error = 0;
for (let i = 0; i < n; i++) {
  const expected = (xs[i] * gates[i]) / (1 + Math.exp(-xs[i]));
  const d = Math.abs(mem[start + i] - expected) / Math.max(1, Math.abs(expected));
  error = Math.max(error, d);
  assert.ok(d < 3e-6);
}
const heads = 4,
  groups = 2,
  dim = 6;
for (const steps of [1, 3, 19, 133]) {
  const q = start,
    key = q + heads * dim,
    value = key + steps * groups * dim,
    out = value + steps * groups * dim,
    scores = out + heads * dim;
  for (let i = q; i < out; i++) mem[i] = Math.fround(Math.sin(i * 0.71) * Math.cos(i * 0.13) * 3);
  w.attend(q * 4, key * 4, value * 4, out * 4, scores * 4, steps, heads, groups, dim);
  for (let h = 0; h < heads; h++) {
    const kh = Math.floor(h / (heads / groups)),
      dots = [];
    for (let t = 0; t < steps; t++) {
      let dot = 0;
      for (let j = 0; j < dim; j++) dot += mem[q + h * dim + j] * mem[key + (t * groups + kh) * dim + j];
      dots.push(dot / Math.sqrt(dim));
    }
    const peak = Math.max(...dots),
      mass = dots.map((x) => Math.exp(x - peak)),
      z = mass.reduce((a, b) => a + b);
    for (let j = 0; j < dim; j++) {
      const want = mass.reduce((s, m, t) => s + (m / z) * mem[value + (t * groups + kh) * dim + j], 0);
      assert.ok(Math.abs(want - mem[out + h * dim + j]) < 5e-6);
    }
  }
}
console.log('SwiGLU max scaled error', error, 'and grouped attention passed at four lengths.');
