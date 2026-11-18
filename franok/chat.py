"""Interactive CPU or GPU reference chat with your downloaded Franok files."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer,decoders
from .model import Config,Franok

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path,help="Directory with config, tokenizer and model.safetensors")
    p.add_argument("--lang",choices=["en","tr"],default="en")
    p.add_argument("--max-new",type=int,default=60)
    args=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    tok=Tokenizer.from_file(str(args.directory/"tokenizer.json"));tok.decoder=decoders.ByteLevel()
    cfg=Config(**json.loads((args.directory/"config.json").read_text()))
    m=Franok(cfg).eval().to(device);m.load_state_dict(load_file(str(args.directory/"model.safetensors")))
    limit=max(1,min(cfg.window//2,args.max_new));history=[]
    while True:
        try:line=input("You › ").strip()
        except (EOFError,KeyboardInterrupt):break
        if not line:continue
        prompt=[3,*tok.encode(line).ids[-(cfg.window-limit-4):],4]
        room=cfg.window-limit-len(prompt)-2
        context=[1,5 if args.lang=="en" else 6,*(history[-room:] if room>0 else []),*prompt]
        output=[]
        for _ in range(limit):
            logits=m(torch.tensor([context+output],device=device))[0,-1]
            logits[:2]=-torch.inf;logits[3:7]=-torch.inf
            if output:logits[list(set(output[-24:]))]-=.75
            values,indices=logits.topk(8)
            token=indices[torch.multinomial((values/.63).softmax(0),1)].item()
            if token==2:break
            output.append(token)
        print("Franok ›",tok.decode(output).strip())
        history=(history+prompt+output+[2])[-cfg.window:]

if __name__=="__main__":main()
