"""Paired, context-limited Belebele reading evaluation. No test text in training."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer
from franok.model import Config,Franok

REV="7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
BASE="a10cc1512eabd3dde888204e902eca88bddb4951"
PEERS = {
    "smollm2": ("HuggingFaceTB/SmolLM2-360M-Instruct", BASE),
    "lfm2": ("LiquidAI/LFM2-350M", "f37d3f5c8c5484bc01dad379a595cf4c68c4e70e"),
}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",action="store_true")
    p.add_argument("--run",type=Path,default=Path("work/run"))
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    from transformers import AutoTokenizer,AutoModelForCausalLM
    base_id="HuggingFaceTB/SmolLM2-360M-Instruct"
    bt=AutoTokenizer.from_pretrained(base_id,revision=BASE)
    ft=Tokenizer.from_file("work/data/tokenizer.json")
    if args.baseline:
        m=AutoModelForCausalLM.from_pretrained(base_id,revision=BASE,local_files_only=True).eval()
    else:
        m=Franok(Config(**json.loads((args.run/"config.json").read_text()))).eval()
        m.load_state_dict(load_file(str(args.run/"model.safetensors")))
    def encoded(row,lang,baseline):
        prompt=row["flores_passage"]+"\n\n"+row["question"]
        if baseline:
            head=bt.apply_chat_template([dict(role="user",content=prompt)],tokenize=True,add_generation_prompt=True)
        else:head=[1,5 if lang=="en" else 6,3,*ft.encode(prompt).ids,4]
        tails=[]
        for i in range(1,5):
            text=row["mc_answer"+str(i)]
            ids=bt.encode(text,add_special_tokens=False) if baseline else ft.encode(text).ids
            tails.append((ids,len(text)))
        return head,tails
    sets={}
    for lang,dialect in [("en","eng_Latn"),("tr","tur_Latn")]:
        loc=hf_hub_download("facebook/belebele","data/"+dialect+".jsonl",revision=REV,repo_type="dataset")
        rows=map(json.loads,Path(loc).read_text().splitlines())
        sets[lang]={(r['link'],r['question_number']):r for r in rows}
    keys=sorted(sets["en"].keys() & sets["tr"].keys(),key=lambda k:hashlib.sha256(repr(k).encode()).digest())
    results=[]
    for key in keys:
        pair={lang:sets[lang][key] for lang in ["en","tr"]}
        both={(lang,base):encoded(row,lang,base) for lang,row in pair.items() for base in [True,False]}
        if any(len(h)+max(len(t) for t,n in ts)>256 for h,ts in both.values()):continue
        for lang,row in pair.items():
            head,tails=both[lang,args.baseline]
            seq=[head+ids[:-1] for ids,n in tails];width=max(map(len,seq))
            batch=torch.tensor([s+[0]*(width-len(s)) for s in seq])
            out=m(batch,use_cache=False).logits if args.baseline else m(batch)
            scores=[]
            for idx,(ids,n) in enumerate(tails):
                lp=out[idx,len(head)-1:len(head)-1+len(ids)].float().log_softmax(-1)
                score=lp.gather(1,torch.tensor(ids)[:,None]).sum().item()/n
                scores.append(round(score,6))
            pred=1+max(range(4),key=scores.__getitem__)
            results.append(dict(language=lang,link=key[0],question=key[1],pred=pred,gold=int(row["correct_answer_num"]),scores=scores))
        if len(results)==96:break
    summary={lang:sum(r['pred']==r['gold'] for r in results if r['language']==lang) for lang in ["en","tr"]}
    assert len(results)==96
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(dataset_revision=REV,per_language=48,summary=summary,items=results),indent=2))
    print(summary,flush=True)
if __name__=="__main__":main()
