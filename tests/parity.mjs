import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createEngine} from '../web/engine.mjs';
import {Tokenizer} from '../web/tokenizer.mjs';
const read=p=>fs.readFileSync(p);
const json=p=>JSON.parse(read(p));
const dir='work/parity/';
const tok=new Tokenizer(json(dir+'export/tokenizer.json'));
for(const [text,ids] of json(dir+'token-tests.json')) {
  assert.deepEqual(tok.encode(text),ids);assert.equal(tok.decode(ids),text);
}
const model=await createEngine(read('web/franok.wasm'),json(dir+'export/model.json'),read(dir+'export/model.q4'));
const ref=json(dir+'reference.json');
let worst=1;
for(const [pos,id] of ref.ids.entries()) {
  const pred=model.step(id),truth=ref.logits[pos];
  let aa=0,bb=0,ab=0;
  pred.forEach((a,i)=>{let b=truth[i];ab+=a*b;aa+=a*a;bb+=b*b;});
  worst=Math.min(worst,ab/Math.sqrt(aa*bb));
}
console.log('Worst FP32 / WASM Q4 logit cosine',worst);
assert(worst>.98);
