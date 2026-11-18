export const terms = text => (text.toLocaleLowerCase('tr').normalize('NFKC').match(/[\p{L}\p{N}]+/gu)||[]);
// FTS-style mixed-language overlap, with a local turn number as provenance.
export function retrieve(turns, query, count=3) {
  const needles=[...new Set(terms(query))];
  const docs=turns.map((turn,index)=>({turn,index,words:new Set(terms(turn.text))}));
  const weights=new Map(needles.map(k=>[k,Math.log(1+(docs.length+.5)/(.5+docs.filter(d=>d.words.has(k)).length))]));
  return docs.map(d=>({...d,score:needles.reduce((s,k)=>s+(d.words.has(k)?weights.get(k):0),0)}))
    .filter(d=>d.score>0).sort((a,b)=>b.score-a.score||b.index-a.index).slice(0,count);
}
export function promptTokens(tok,turns,lang,limit) {
  const prefix=[1,lang==='tr'?6:5], current=turns.at(-1);
  const query=tok.encode(current.text), tail=[3,...query.slice(-Math.min(limit-4,120)),4];
  let budget=limit-prefix.length-tail.length;
  const snippets=[];
  const attached=current.quote;
  if(Number.isInteger(attached)&&attached>=0&&attached<turns.length-1&&budget>4){
    const source=turns[attached];
    const body=tok.encode(source.text).slice(0,Math.min(68,budget-2));
    snippets.push({index:attached,ids:[source.role==='user'?3:4,...body,2]});
    budget-=body.length+2;
  }
  // Prioritize the last exchange. Spend spare room on older, matching turns.
  const history=turns.slice(0,-1), last=history.slice(-2);
  const recalled=retrieve(history.slice(0,Math.max(0,history.length-2)),current.text,2).map(h=>h.index);
  const indices=[...last.map((_,i)=>history.length-last.length+i).reverse(),...recalled];
  for(const index of indices) {
    if(budget<6)break;
    if(snippets.some(s=>s.index===index))continue;
    const t=history[index], body=tok.encode(t.text).slice(0,Math.min(44,budget-2));
    snippets.push({index,ids:[t.role==='user'?3:4,...body,...(t.role==='assistant'?[2]:[])]});
    budget-=snippets.at(-1).ids.length;
  }
  return [...prefix,...snippets.sort((a,b)=>a.index-b.index).flatMap(s=>s.ids),...tail];
}
