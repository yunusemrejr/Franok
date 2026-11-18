import fs from 'node:fs';
import {strict as assert} from 'node:assert';
import {createEngine} from '../web/engine.mjs';
const json=path=>JSON.parse(fs.readFileSync(path));
const manifest=json('work/export/model.json');
const ids=json('work/release-parity/ids.json');
const buf=fs.readFileSync('work/release-parity/logits.f32');
const ref=new Float32Array(buf.buffer,buf.byteOffset,buf.length/4);
const model=await createEngine(fs.readFileSync('web/franok.wasm'),manifest,fs.readFileSync('work/export/model.q4'));
const evals=[];let offset=0;
for(const row of ids){
  model.reset();let fp=0,q4=0,agree=0,cos=0;
  for(let t=0;t<row.length-1;t++){
    const actual=model.step(row[t]),expected=ref.subarray(offset,offset+actual.length);offset+=actual.length;
    function stats(logits){
      let arg=0;for(let i=1;i<logits.length;i++)if(logits[i]>logits[arg])arg=i;
      const max=logits[arg];let z=0;for(const x of logits)z+=Math.exp(x-max);
      return [arg,max+Math.log(z)-logits[row[t+1]]];
    }
    const [a,la]=stats(actual),[b,lb]=stats(expected);
    agree+=Number(a===b);fp+=lb;q4+=la;
    let ab=0,aa=0,bb=0;for(let i=0;i<actual.length;i++){ab+=actual[i]*expected[i];aa+=actual[i]**2;bb+=expected[i]**2;}
    cos+=ab/Math.sqrt(aa*bb);
  }
  const tokens=row.length-1;evals.push({tokens,fp_nll:fp/tokens,q4_nll:q4/tokens,top1_agreement:agree/tokens,logit_cosine:cos/tokens});
}
assert(evals.every(e=>e.logit_cosine>.95));
const out={slices:['English text','English chat','Turkish text','Turkish chat'],results:evals};
fs.writeFileSync('docs/evaluation/wasm-parity.json',JSON.stringify(out,null,2)+'\n');console.log(out);
