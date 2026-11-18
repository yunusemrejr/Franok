"""Continue pretraining on balanced texts, then supervise assistant spans."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F
from franok.model import Franok,Config

DATA=Path("work/data")
OUT=Path("work/run")

class Streams:
    def __init__(self,split):
        self.text={}
        self.chat={}
        for lang in ("en","tr"):
            for kind in ("text","chat"):
                a=np.memmap(DATA/f"{split}-{lang}-{kind}.bin",mode="r",dtype="<u2")
                if kind=="text":
                    self.text[lang]=a
                    continue
                ends=np.flatnonzero(a==2); starts=np.r_[0,ends[:-1]+1]
                roles=np.flatnonzero(a==4)
                assert len(ends)==len(roles) and np.all(a[starts]==1)
                self.chat[lang]=(a,starts,roles,ends)
    def batch(self,rng,n,seq,supervised):
        x=np.zeros((n,seq+1),dtype="int64")
        y=np.full((n,seq),-100,dtype="int64")
        for i in range(n):
            lang=("en","tr")[i%2]
            if supervised:
                data,starts,roles,ends=self.chat[lang]
                at=0
                while seq-at>=22:
                    idx=rng.integers(len(ends))
                    q=data[starts[idx]+3:roles[idx]]
                    a=data[roles[idx]+1:ends[idx]+1]
                    room=seq+1-at
                    # Shorten the context first and preserve the end of the question.
                    nq=min(len(q),max(8,room//2))
                    na=min(len(a),room-4-nq)
                    turn=np.r_[data[starts[idx]:starts[idx]+3],q[-nq:] if nq else q,4,a[:na]]
                    x[i,at:at+len(turn)]=turn
                    first=at+3+nq
                    y[i,first:first+na]=a[:na]
                    at+=len(turn)
            else:
                a=self.text[lang] if rng.random()<.80 else self.chat[lang][0]
                pos=rng.integers(len(a)-seq)
                x[i]=a[pos:pos+seq+1];y[i]=x[i,1:]
        return torch.tensor(x[:,:-1],device="cuda"),torch.tensor(y,device="cuda")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pretrain",type=int,default=4800)
    p.add_argument("--sft",type=int,default=1200)
    p.add_argument("--batch",type=int,default=24)
    p.add_argument("--minutes",type=float,default=35)
    args=p.parse_args()
    rng=np.random.default_rng(4401);torch.manual_seed(4401)
    cfg=Config(**json.loads((OUT/"config.json").read_text()))
    m=Franok(cfg).cuda()
    m.load_state_dict(load_file(str(OUT/"model.safetensors")))
    train,val=Streams("train"),Streams("val")
    progress=json.loads((OUT/"progress.json").read_text())
    optim=torch.optim.AdamW(m.parameters(),lr=.0004,betas=(.9,.95),weight_decay=.06,fused=True)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    def run(stream,chooser,batch,supervised):
        x,y=stream.batch(chooser,batch,cfg.window,supervised)
        with torch.autocast("cuda",dtype=dtype):
            loss=F.cross_entropy(m(x).flatten(0,1),y.flatten())
        return loss,int((y!=-100).sum())
    def log(**entry):
        print(entry,flush=True)
        with (OUT/"refinement.jsonl").open("a") as f:f.write(json.dumps(entry)+"\n")
    @torch.no_grad()
    def evaluate():
        metrics={}
        for mode in [False,True]:
            vrng=np.random.default_rng(3521)
            metrics["assistant_nll" if mode else "mixed_nll"]=sum(run(val,vrng,8,mode)[0].item() for _ in range(8))/8
        return metrics
    consumed=targets=0
    log(initial=evaluate(),options=vars(args),seed=4401)
    start=time.monotonic()
    for mode,steps in [(False,args.pretrain),(True,args.sft)]:
        for idx in range(steps):
            limit=args.minutes*60*(1 if mode else .79)
            lr=(.00032 if not mode else .00013)*min(1,(idx+1)/60)*(.24+.76*(1+math.cos(math.pi*idx/max(1,steps)))/2)
            optim.param_groups[0]["lr"]=lr; optim.zero_grad(set_to_none=True)
            loss,valid=run(train,rng,args.batch,mode)
            scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(m.parameters(),1)
            scaler.step(optim);scaler.update()
            consumed+=args.batch*cfg.window;targets+=valid
            if idx%100==0:log(phase="assistant" if mode else "continuation",step=idx+1,loss=loss.item(),target_tokens=targets)
            if time.monotonic()-start>limit:break
        name="assistant" if mode else "continuation"
        save_file({k:v.detach().to("cpu",torch.float16).contiguous() for k,v in m.state_dict().items()},str(OUT/"refined.tmp"))
        (OUT/"refined.tmp").replace(OUT/"model.safetensors")
        metrics=evaluate()
        log(finished=name,steps=idx+1,elapsed=time.monotonic()-start,metrics=metrics)
        (OUT/"refined.json").write_text(json.dumps(dict(stage=name,pretraining=progress,additional_positions=consumed,additional_targets=targets,metrics=metrics),indent=2))
if __name__=="__main__":main()
