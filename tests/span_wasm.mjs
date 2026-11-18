import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createReadout, scoreSpan } from '../web/reader.mjs';
const folder = process.argv[2],
  config = { width: 32, hidden: 16, max_span: 7 };
const head = await createReadout(
  fs.readFileSync('web/franok.wasm'),
  fs.readFileSync(path.join(folder, 'head.safetensors')),
  config,
);
for (const { states, offset, legal, want } of JSON.parse(fs.readFileSync(path.join(folder, 'test.json')))) {
  const out = scoreSpan(
    head,
    states.map((row) => Float32Array.from(row)),
    offset,
    legal,
  );
  assert.equal(out.from, want.start);
  assert.equal(out.through, want.end);
  assert.ok(Math.abs(out.margin - want.margin) < 2e-5, JSON.stringify({ out, want }));
}
console.log('WASM readout matched PyTorch span boundaries and confidence margins.');
