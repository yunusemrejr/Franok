const encoder = new TextEncoder(), decoder = new TextDecoder();
const pattern = / ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s/gu;

export class Tokenizer {
  constructor(spec) {
    const alphabet = [];
    for(let i=0;i<256;i++) if((i>=33&&i<=126)||(i>=161&&i<=172)||i>=174) alphabet.push(i);
    const chars=[...alphabet]; let extra=256;
    for(let i=0;i<256;i++) if(!alphabet.includes(i)){alphabet.push(i);chars.push(extra++);}
    this.byteChar=[];const reverse=new Map();
    alphabet.forEach((byte,i)=>{const ch=String.fromCharCode(chars[i]);this.byteChar[byte]=ch;reverse.set(ch,byte);});
    this.vocab=spec.model.vocab;this.pieces=[];
    for(const [str,id] of Object.entries(this.vocab)) this.pieces[id]=Uint8Array.from(str,ch=>reverse.get(ch));
    this.ranks=new Map(spec.model.merges.map((pair,i)=>[Array.isArray(pair)?pair.join(' '):pair,i]));
  }
  encode(text) {
    const output=[];
    for(const match of text.matchAll(pattern)) {
      let symbols=Array.from(encoder.encode(match[0]),b=>this.byteChar[b]);
      while(symbols.length>1) {
        let min=Infinity,left,right;
        for(let i=1;i<symbols.length;i++) {
          const rank=this.ranks.get(symbols[i-1]+' '+symbols[i]);
          if(rank<min){min=rank;left=symbols[i-1];right=symbols[i];}
        }
        if(min===Infinity)break;
        const next=[];
        for(let i=0;i<symbols.length;i++) {
          if(symbols[i]===left&&symbols[i+1]===right){next.push(left+right);i++;}
          else next.push(symbols[i]);
        }
        symbols=next;
      }
      for(const s of symbols)output.push(this.vocab[s]);
    }
    return output;
  }
  decode(ids) {
    const bytes=[];
    for(const id of ids)if(id>=7)bytes.push(...this.pieces[id]);
    return decoder.decode(Uint8Array.from(bytes));
  }
}
