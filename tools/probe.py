"""Score candidate continuations under native chat templates. No benchmark model is used in training."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer,decoders
from franok.model import Franok,Config

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",help="Local directory of the comparison model")
    p.add_argument("--run",type=Path,default=Path("work/run"))
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.baseline:
        from transformers import AutoTokenizer,AutoModelForCausalLM
        t=AutoTokenizer.from_pretrained(args.baseline,local_files_only=True)
        m=AutoModelForCausalLM.from_pretrained(args.baseline,local_files_only=True).eval()
        def prompt(q,lang):return t.apply_chat_template([{"role":"user","content":q}],tokenize=True,add_generation_prompt=True)
        def encode(s):return t.encode(s,add_special_tokens=False)
        def logits(i):return m(i,use_cache=False).logits
        def decode(ids):return t.decode(ids,skip_special_tokens=True)
        end=t.eos_token_id
    else:
        t=Tokenizer.from_file("work/data/tokenizer.json");t.decoder=decoders.ByteLevel()
        m=Franok(Config(**json.loads((args.run/"config.json").read_text()))).eval()
        m.load_state_dict(load_file(str(args.run/"model.safetensors")))
        def encode(s):return t.encode(s).ids
        def prompt(q,lang):return [1,5 if lang=="en" else 6,3,*encode(q),4]
        def logits(i):return m(i)
        def decode(ids):return t.decode(ids)
        end=2
    rows=[]
    for lang,q,good,bad in json.loads(Path("tests/dialogue.json").read_text()):
        prefix=prompt(q,lang)
        def score(s):
            tail=encode(s)
            logp=logits(torch.tensor([prefix+tail[:-1]]))[0,len(prefix)-1:].float()
            return round(-F.cross_entropy(logp,torch.tensor(tail)).item(),5)
        a,b=score(good),score(bad)
        rows.append(dict(lang=lang,q=q,correct=good,foil=bad,score_correct=a,score_foil=b,win=a>b))
    generations=[]
    for lang,q in [("en","Hello! Tell me something about coffee."),("tr","Merhaba! Bana kahveden bahseder misin?"),("en","Name the capital of France."),("tr","Bir haftada kaç gün vardır?")]:
        ids=prompt(q,lang); generated=[]
        for _ in range(32):
            token=logits(torch.tensor([ids+generated]))[0,-1].argmax().item()
            if token==end:break
            generated.append(token)
        generations.append(dict(lang=lang,prompt=q,output=decode(generated)))
    summary={lang:sum(r["win"] for r in rows if r["lang"]==lang) for lang in ["en","tr"]}
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(json.dumps(dict(scores=summary,out_of_per_language=16,rows=rows,generations=generations),ensure_ascii=False,indent=2))
    print(summary,generations,flush=True)
if __name__=="__main__":main()
