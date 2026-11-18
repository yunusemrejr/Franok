import {createEngine} from './engine.mjs';
import {Tokenizer} from './tokenizer.mjs';
import {promptTokens} from './session.mjs';
let model, tok;
const tell=(kind,more={})=>postMessage({kind,...more});
async function read(url,what) {
  const r=await fetch(url);if(!r.ok)throw Error(what+' returned HTTP '+r.status);return r;
}
async function load() {
  if(model)return;
  const release=await (await read('./release.json','Release manifest')).json();
  const [manifest,spec,wasm]=await Promise.all([
    read('./model.json','Model manifest').then(r=>r.json()),read('./tokenizer.json','Tokenizer').then(r=>r.json()),read('./franok.wasm','Runtime').then(r=>r.arrayBuffer())]);
  const url=new URL(release.weights,location.href).href;
  const cache=typeof caches==='undefined'?null:await caches.open('franok-weights-v1').catch(()=>null);
  let bytes;
  const cached=await cache?.match(url);
  if(cached)bytes=await cached.arrayBuffer();
  if(!bytes){
    const response=await read(url,'Model download'), reader=response.body.getReader();
    bytes=new ArrayBuffer(manifest.bytes);const buf=new Uint8Array(bytes);let at=0,last=-1;
    while(true){
      const {done,value}=await reader.read();if(done)break;
      if(at+value.length>buf.length)throw Error('Unexpected model length.');
      buf.set(value,at);at+=value.length;
      const pct=Math.floor(100*at/buf.length);if(pct!==last){tell('download',{pct});last=pct;}
    }
    if(at!==buf.length)throw Error('Incomplete model download.');
  }
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==manifest.sha256){await cache?.delete(url);throw Error('Weight checksum mismatch. Please retry.');}
  if(!cached&&cache)await cache.put(url,new Response(bytes)).catch(()=>{});
  model=await createEngine(wasm,manifest,bytes); tok=new Tokenizer(spec);tell('ready');
}
function nextToken(logits,ids){
  const past=new Set(ids.slice(-24));
  const top=[];
  for(let i=2;i<logits.length;i++){
    if(i<7&&i!==2)continue;
    let v=logits[i]-(past.has(i)?.75:0);
    if(top.length===8&&v<=top[7][1])continue;
    top.push([i,v]);top.sort((a,b)=>b[1]-a[1]);if(top.length>8)top.pop();
  }
  const weights=top.map(x=>Math.exp((x[1]-top[0][1])/.63));
  let n=Math.random()*weights.reduce((a,b)=>a+b,0);
  for(let i=0;i<weights.length;i++)if((n-=weights[i])<=0)return top[i][0];
  return top[0][0];
}
onmessage=async evt=>{
  try{
    await load();
    if(evt.data.type==='load')return;
    model.reset();
    const ids=promptTokens(tok,evt.data.turns,evt.data.lang,model.c.window-72);
    let logits;
    tell('prefill');
    for(const id of ids)logits=model.step(id);
    const generated=[];const start=performance.now();
    for(let i=0;i<72;i++){
      const token=nextToken(logits,generated);if(token===2)break;
      generated.push(token);
      tell('part',{text:tok.decode(generated).replace(/\uFFFD$/,'')});
      if(i+1<72)logits=model.step(token);
    }
    tell('done',{text:tok.decode(generated).trim(),tokens:generated.length,seconds:(performance.now()-start)/1000});
  } catch(e){tell('error',{message:e.message||String(e)});}
};
