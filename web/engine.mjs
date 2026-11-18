// Incremental decoder. The same module runs in a Worker and under Node for parity tests.
export class Engine {
  constructor(instance, manifest, bytes) {
    const w = this.w = instance.exports;
    const m = this.c = manifest.config;
    if (manifest.format !== 'franok-q4-v1' || bytes.byteLength !== manifest.bytes) throw Error('Incomplete model file.');
    const dim = m.width, hd = dim / m.heads, kv = hd * m.kv_heads;
    this.base = Number(w.__heap_base.value);
    this.t = manifest.tensors;
    let next = this.base + bytes.byteLength;
    const alloc = length => {
      next = Math.ceil(next / 16) * 16;
      const start = next; next += 4 * length; return start;
    };
    this.x = alloc(dim); this.norm = alloc(dim); this.qkv = alloc(dim + kv * 2);
    this.a = alloc(dim); this.h = alloc(m.hidden * 2); this.scores = alloc(m.window);
    this.logits = alloc(m.vocab); this.cos = alloc(m.window * hd / 2); this.sin = alloc(m.window * hd / 2);
    this.act = alloc(Math.max(dim,m.hidden) / 4); this.scales = alloc(Math.max(dim,m.hidden) / 32);
    this.keys = alloc(m.layers * m.window * kv); this.values = alloc(m.layers * m.window * kv);
    w.memory.grow(Math.max(0,Math.ceil((next - w.memory.buffer.byteLength)/65536)));
    this.f = new Float32Array(w.memory.buffer);
    new Uint8Array(w.memory.buffer,this.base,bytes.byteLength).set(new Uint8Array(bytes));
    for(let pos=0; pos<m.window; pos++) for(let i=0; i<hd/2; i++) {
      const theta=pos * 10000 ** (-2*i/hd), j=pos*hd/2+i;
      this.f[this.cos/4+j]=Math.cos(theta); this.f[this.sin/4+j]=Math.sin(theta);
    }
    this.reset();
  }
  reset() { this.pos=0; this.anchor=0; this.last=0; }
  ptr(name) { return this.base+this.t[name].offset; }
  step(token) {
    const {w,c,f,pos} = this, d=c.width, hd=d/c.heads, kv=c.kv_heads*hd;
    if(pos>=c.window || !Number.isInteger(token) || token<0 || token>=c.vocab) throw Error('Decoder input outside its context.');
    if(pos%c.banks===0) this.anchor=this.last;
    this.last=token;
    const mul=(p,x,y,rows,cols)=>w.gemv(p,x,y,rows,cols,this.act,this.scales);
    const tbl=this.ptr('embed.weight'), rowBytes=d/32*20;
    w.unpack(tbl+rowBytes*token,this.x,d);
    for(let layer=0;layer<c.layers;layer++) {
      const p=name=>this.ptr('blocks.'+layer+'.'+name);
      w.rms(this.x,p('pre.weight'),this.norm,d);
      mul(p('qkv.weight'),this.norm,this.qkv,d+2*kv,d);
      const k=this.qkv+4*d, v=k+4*kv;
      const cos=this.cos+2*pos*hd, sin=this.sin+2*pos*hd;
      w.rotary(this.qkv,cos,sin,c.heads,hd); w.rotary(k,cos,sin,c.kv_heads,hd);
      const cacheOffset=4*layer*c.window*kv;
      const keys=this.keys+cacheOffset, vals=this.values+cacheOffset;
      f.copyWithin(keys/4+pos*kv,k/4,k/4+kv);
      f.copyWithin(vals/4+pos*kv,v/4,v/4+kv);
      w.attend(this.qkv,keys,vals,this.a,this.scores,pos+1,c.heads,c.kv_heads,hd);
      mul(p('out.weight'),this.a,this.norm,d,d); w.add(this.x,this.norm,d);
      w.rms(this.x,p('post.weight'),this.norm,d);
      const bank=(pos%c.banks+(this.anchor>>(layer%4))+layer)%c.banks;
      mul(p('up')+bank*this.t['blocks.'+layer+'.up'].bytes/c.banks,this.norm,this.h,2*c.hidden,d);
      w.swiglu(this.h,c.hidden);
      mul(p('down')+bank*this.t['blocks.'+layer+'.down'].bytes/c.banks,this.h,this.norm,d,c.hidden);
      w.add(this.x,this.norm,d);
    }
    w.rms(this.x,this.ptr('norm.weight'),this.norm,d);
    mul(tbl,this.norm,this.logits,c.vocab,d);
    this.pos++;
    return f.subarray(this.logits/4,this.logits/4+c.vocab);
  }
}

export async function createEngine(wasm,manifest,bytes) {
  const {instance} = await WebAssembly.instantiate(wasm,{math:{exp:Math.exp}});
  return new Engine(instance,manifest,bytes);
}
